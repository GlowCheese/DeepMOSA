####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_13 = 2
    var_14 = 3
    var_15 = 4
    var_16 = [var_14, var_15]
    var_17 = [var_13, var_16]
    var_18 = [var_3, var_17]
    var_19 = 0
    var_20 = [var_3, var_3, var_19]
    var_21 = module_0.get_in(var_20, var_18)
    assert var_21 == 3
    var_22 = [var_3, var_3]
    var_23 = module_0.get_in(var_22, var_18)
    var_24 = [var_3]
    var_25 = module_0.get_in(var_24, var_18)
    var_26 = 'x'
    var_27 = [var_26]
    var_28 = 'not found'
    var_29 = module_0.get_in(var_27, var_18, var_28)
    assert var_29 == 'not found'
    var_30 = 'y'
    var_31 = [var_26, var_30]
    var_32 = module_0.get_in(var_31, var_18, var_28)
    assert var_32 == 'not found'
    var_33 = 5
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_18, var_28)
    assert var_35 == 'not found'
    var_36 = 'x'
    var_37 = [var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_18, no_default=var_38)
    var_40 = 5
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_18, no_default=var_42)
    var_44 = []
    var_45 = module_0.get_in(var_44, var_18)
    var_46 = {var_41: var_13}
    var_47 = [var_43, var_46]
    var_48 = {var_40: var_47}
    var_49 = [var_40, var_43, var_41]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 2
    var_51 = [var_40, var_43]
    var_52 = module_0.get_in(var_51, var_48)
    var_53 = [var_40]
    var_54 = module_0.get_in(var_53, var_48)
    var_55 = [var_26]
    var_56 = module_0.get_in(var_55, var_48)
    assert var_56 is None
    var_57 = [var_40, var_33]
    var_58 = module_0.get_in(var_57, var_48)
    assert var_58 is None



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 0
    var_19 = 2
    var_20 = [var_3, var_19]
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = 5
    var_26 = 6
    var_27 = [var_25, var_26]
    var_28 = 7
    var_29 = 8
    var_30 = [var_28, var_29]
    var_31 = [var_27, var_30]
    var_32 = [var_24, var_31]
    var_33 = [var_17, var_3, var_3]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 4
    var_35 = [var_3, var_17]
    var_36 = module_0.get_in(var_35, var_32)
    var_37 = [var_19]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = [var_19]
    var_40 = module_0.get_in(var_39, var_32, var_17)
    assert var_40 == 0
    var_41 = {var_1: var_21}
    var_42 = [var_3, var_19, var_41]
    var_43 = {var_0: var_42}
    var_44 = [var_0, var_19, var_1]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 3
    var_46 = [var_0, var_3]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 == 2
    var_48 = [var_0, var_21]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_0, var_21]
    var_51 = module_0.get_in(var_50, var_43, var_17)
    assert var_51 == 0
    var_52 = {var_1: var_3}
    var_53 = {var_0: var_52}
    var_54 = 'x'
    var_55 = [var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_53, no_default=var_56)
    var_58 = 'a'
    var_59 = 'x'
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_53, no_default=var_61)
    var_63 = {var_58: var_61}
    var_64 = []
    var_65 = module_0.get_in(var_64, var_63)
    var_66 = {var_59: var_61}
    var_67 = {var_58: var_66}
    var_68 = 'y'
    var_69 = 'z'
    var_70 = [var_13, var_68, var_69]
    var_71 = module_0.get_in(var_70, var_67)
    assert var_71 is None
    var_72 = [var_58, var_13, var_68]
    var_73 = module_0.get_in(var_72, var_67)
    assert var_73 is None
    var_74 = {var_58: var_61}
    var_75 = [var_58, var_59]
    var_76 = module_0.get_in(var_75, var_74)
    assert var_76 is None
    var_77 = [var_58, var_59]
    var_78 = module_0.get_in(var_77, var_74, var_17)
    assert var_78 == 0



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'f'
    var_2 = 'b'
    var_3 = 'e'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 3
    var_10 = 4
    var_11 = 5
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_2: var_8, var_3: var_12}
    var_14 = 'value'
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = [var_0, var_2, var_4]
    var_17 = module_0.get_in(var_16, var_15)
    assert var_17 == 1
    var_18 = [var_0, var_2, var_5]
    var_19 = module_0.get_in(var_18, var_15)
    assert var_19 == 2
    var_20 = [var_0, var_3, var_6]
    var_21 = module_0.get_in(var_20, var_15)
    assert var_21 == 4
    var_22 = [var_1]
    var_23 = module_0.get_in(var_22, var_15)
    assert var_23 == 'value'
    var_24 = 'x'
    var_25 = [var_0, var_2, var_24]
    var_26 = module_0.get_in(var_25, var_15)
    assert var_26 is None
    var_27 = [var_0, var_2, var_24]
    var_28 = 'default'
    var_29 = module_0.get_in(var_27, var_15, var_28)
    assert var_29 == 'default'
    var_30 = 'y'
    var_31 = [var_0, var_24, var_30]
    var_32 = 0
    var_33 = module_0.get_in(var_31, var_15, var_32)
    assert var_33 == 0
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 'x'
    var_37 = [var_34, var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_15, no_default=var_38)
    var_40 = 'a'
    var_41 = 'e'
    var_42 = 10
    var_43 = [var_40, var_41, var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_15, no_default=var_44)
    var_46 = []
    var_47 = module_0.get_in(var_46, var_15)
    var_48 = [var_6, var_7]
    var_49 = [var_9, var_10]
    var_50 = 6
    var_51 = [var_11, var_50]
    var_52 = [var_48, var_49, var_51]
    var_53 = [var_6, var_6]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 == 4
    var_55 = [var_7]
    var_56 = module_0.get_in(var_55, var_52)
    var_57 = [var_9]
    var_58 = module_0.get_in(var_57, var_52)
    assert var_58 is None
    var_59 = [var_9]
    var_60 = 'missing'
    var_61 = module_0.get_in(var_59, var_52, var_60)
    assert var_61 == 'missing'
    var_62 = 'items'
    var_63 = 'prices'
    var_64 = 'apple'
    var_65 = 'banana'
    var_66 = 'cherry'
    var_67 = [var_64, var_65, var_66]
    var_68 = 0.5
    var_69 = {var_64: var_6, var_65: var_68}
    var_70 = {var_62: var_67, var_63: var_69}
    var_71 = [var_62, var_6]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 'banana'
    var_73 = [var_63, var_64]
    var_74 = module_0.get_in(var_73, var_70)
    var_75 = [var_63, var_66]
    var_76 = module_0.get_in(var_75, var_70)
    assert var_76 is None
    var_77 = [var_63, var_66]
    var_78 = module_0.get_in(var_77, var_70, var_32)
    var_79 = [var_40, var_42, var_44, var_45]
    var_80 = module_0.get_in(var_79, var_15)
    assert var_80 is None
    var_81 = [var_40, var_42, var_44, var_45]
    var_82 = 'error'
    var_83 = module_0.get_in(var_81, var_15, var_82)
    assert var_83 == 'error'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_17]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = {var_44: var_46}
    var_51 = {var_43: var_50}
    var_52 = [var_43, var_45]
    var_53 = module_0.get_in(var_52, var_51, var_15)
    assert var_53 == 0
    var_54 = [var_43, var_45]
    var_55 = None
    var_56 = module_0.get_in(var_54, var_51, var_55)
    assert var_56 is None
    var_57 = [var_43, var_45]
    var_58 = 'default'
    var_59 = module_0.get_in(var_57, var_51, var_58)
    assert var_59 == 'default'



# Parsed testcases at query #5
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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = {var_1: var_17}
    var_30 = [var_3, var_29]
    var_31 = {var_0: var_30}
    var_32 = [var_0, var_3, var_1]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 2
    var_34 = [var_0, var_15]
    var_35 = module_0.get_in(var_34, var_31)
    assert var_35 == 1
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_31)
    var_38 = {var_0: var_3}
    var_39 = 'b'
    var_40 = [var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_38, no_default=var_41)
    var_43 = 'y'
    var_44 = 'z'
    var_45 = [var_11, var_43, var_44]
    var_46 = {}
    var_47 = 'default'
    var_48 = module_0.get_in(var_45, var_46, var_47)
    assert var_48 == 'default'
    var_49 = [var_11, var_43, var_44]
    var_50 = {}
    var_51 = None
    var_52 = module_0.get_in(var_49, var_50, var_51)
    assert var_52 is None
    var_53 = {var_39: var_42}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = 'string'
    var_57 = {var_39: var_56}
    var_58 = [var_39, var_40]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_39, var_40]
    var_61 = 'error'
    var_62 = module_0.get_in(var_60, var_57, var_61)
    assert var_62 == 'error'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = [var_17]
    var_31 = module_0.get_in(var_30, var_22, var_23)
    assert var_31 == 0
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_34)
    var_39 = [var_0, var_17]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'b'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = {var_42: var_45}
    var_47 = []
    var_48 = module_0.get_in(var_47, var_46)
    var_49 = {var_43: var_45}
    var_50 = {var_42: var_49}
    var_51 = [var_42, var_44]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_42, var_44]
    var_54 = module_0.get_in(var_53, var_50, var_23)
    assert var_54 == 0
    var_55 = {var_42: var_45}
    var_56 = [var_42, var_43]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None



# Parsed testcases at query #7
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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 99
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 99
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9]
    var_28 = {var_23: var_27}
    var_29 = 'a'
    var_30 = 5
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_28, no_default=var_32)
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = [var_38, var_9, var_10]
    var_44 = {var_36: var_43}
    var_45 = 4
    var_46 = {var_37: var_45}
    var_47 = [var_44, var_46]
    var_48 = {var_35: var_47}
    var_49 = 0
    var_50 = [var_35, var_49, var_36, var_38]
    var_51 = module_0.get_in(var_50, var_48)
    assert var_51 == 2
    var_52 = [var_35, var_38, var_37]
    var_53 = module_0.get_in(var_52, var_48)
    assert var_53 == 4
    var_54 = 'name'
    var_55 = 'purchase'
    var_56 = 'credit card'
    var_57 = 'Alice'
    var_58 = 'items'
    var_59 = 'costs'
    var_60 = 'Apple'
    var_61 = 'Orange'
    var_62 = [var_60, var_61]
    var_63 = 0.5
    var_64 = 1.25
    var_65 = [var_63, var_64]
    var_66 = {var_58: var_62, var_59: var_65}
    var_67 = '5555-1234-1234-1234'
    var_68 = {var_54: var_57, var_55: var_66, var_56: var_67}
    var_69 = [var_55, var_58, var_49]
    var_70 = [var_54]
    var_71 = 'total'
    var_72 = [var_55, var_71]
    var_73 = [var_55, var_71]



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 'x'
    var_18 = [var_17]
    var_19 = True
    var_20 = module_0.get_in(var_18, var_6, no_default=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = {var_17: var_23}
    var_25 = [var_17, var_20]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 2
    var_27 = 10
    var_28 = [var_17, var_27]
    var_29 = module_0.get_in(var_28, var_24)
    assert var_29 is None
    var_30 = {var_19: var_22}
    var_31 = [var_20, var_21, var_30]
    var_32 = {var_18: var_31}
    var_33 = {var_17: var_32}
    var_34 = [var_17, var_18, var_21, var_19]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 3
    var_36 = 'd'
    var_37 = [var_17, var_18, var_21, var_36]
    var_38 = module_0.get_in(var_37, var_33)
    assert var_38 is None
    var_39 = []
    var_40 = module_0.get_in(var_39, var_33)
    var_41 = {var_17: var_20}
    var_42 = [var_17, var_18]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 is None



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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_17]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = {var_43: var_46}
    var_51 = [var_43, var_44]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_43, var_44]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0



# Parsed testcases at query #10
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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3]
    var_27 = module_0.get_in(var_26, var_22)
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = [var_17]
    var_31 = module_0.get_in(var_30, var_22, var_15)
    assert var_31 == 'default'
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_34)
    var_39 = [var_0, var_17]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = [var_0, var_17]
    var_42 = module_0.get_in(var_41, var_34, var_15)
    assert var_42 == 'default'
    var_43 = {var_0: var_3}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = {var_44: var_47}
    var_49 = []
    var_50 = module_0.get_in(var_49, var_48)
    var_51 = None
    var_52 = {var_45: var_51}
    var_53 = {var_44: var_52}
    var_54 = [var_44, var_45]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 is None
    var_56 = [var_44, var_45]
    var_57 = module_0.get_in(var_56, var_53, var_15)
    assert var_57 is None
    var_58 = [var_44, var_46]
    var_59 = module_0.get_in(var_58, var_53, var_15)
    assert var_59 == 'default'



# Parsed testcases at query #11
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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = [var_18, var_19]
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_15, var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 4
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_15]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 == 1
    var_38 = [var_0, var_3, var_2]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_3, var_2]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = [var_43, var_44]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 is None
    var_50 = [var_43, var_44]
    var_51 = module_0.get_in(var_50, var_47, var_15)
    assert var_51 == 0
    var_52 = {var_43: var_46}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = None
    var_56 = {var_44: var_55}
    var_57 = {var_43: var_56}
    var_58 = [var_43, var_44]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_43, var_44]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 is None
    var_62 = [var_43, var_45]
    var_63 = module_0.get_in(var_62, var_57, var_15)
    assert var_63 == 0



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 0
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 0
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = 'a'
    var_23 = 'c'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_16, no_default=var_25)
    var_27 = {var_23: var_25}
    var_28 = {var_23: var_9}
    var_29 = [var_27, var_28]
    var_30 = {var_22: var_29}
    var_31 = [var_22, var_25, var_23]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 2
    var_33 = [var_25, var_9, var_10]
    var_34 = {var_22: var_33}
    var_35 = 10
    var_36 = [var_22, var_35]
    var_37 = module_0.get_in(var_36, var_34, var_18)
    assert var_37 == 0
    var_38 = {var_22: var_25}
    var_39 = [var_22, var_23]
    var_40 = module_0.get_in(var_39, var_38, var_18)
    assert var_40 == 0
    var_41 = {var_22: var_25}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_22: var_25}
    var_45 = [var_22]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 1



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'not found'
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = 'a'
    var_23 = 'c'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_16, no_default=var_25)
    var_27 = [var_25, var_9, var_10]
    var_28 = {var_22: var_27}
    var_29 = 10
    var_30 = [var_22, var_29]
    var_31 = 'out of bounds'
    var_32 = module_0.get_in(var_30, var_28, var_31)
    assert var_32 == 'out of bounds'
    var_33 = 'a'
    var_34 = 10
    var_35 = [var_33, var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_28, no_default=var_36)
    var_38 = {var_33: var_36}
    var_39 = [var_33, var_34]
    var_40 = 'type error'
    var_41 = module_0.get_in(var_39, var_38, var_40)
    assert var_41 == 'type error'
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_38, no_default=var_45)
    var_47 = {var_42: var_45}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = 'a'
    var_51 = 'b'
    var_52 = 1
    var_53 = {var_51: var_52}
    var_54 = {var_50: var_53}
    var_55 = [var_50, var_51]
    var_56 = module_0.get_in(var_55, var_47)
    assert var_56 == 1



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 is None
    var_37 = {var_0: var_3}
    var_38 = 'b'
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_37, no_default=var_40)
    var_42 = 'y'
    var_43 = [var_11, var_42]
    var_44 = {}
    var_45 = 'not_found'
    var_46 = module_0.get_in(var_43, var_44, var_45)
    assert var_46 == 'not_found'
    var_47 = {var_38: var_41}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = 'string'
    var_51 = [var_23]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 == 's'
    var_53 = [var_41]
    var_54 = module_0.get_in(var_53, var_50)
    assert var_54 == 't'
    var_55 = [var_11]
    var_56 = module_0.get_in(var_55, var_50)
    assert var_56 is None



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = [var_0, var_1, var_11]
    var_18 = module_0.get_in(var_17, var_6, var_15)
    assert var_18 == 0
    var_19 = 2
    var_20 = [var_3, var_19]
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = [var_15, var_3]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 2
    var_27 = [var_3, var_15]
    var_28 = module_0.get_in(var_27, var_24)
    assert var_28 == 3
    var_29 = [var_19]
    var_30 = module_0.get_in(var_29, var_24)
    assert var_30 is None
    var_31 = [var_19]
    var_32 = module_0.get_in(var_31, var_24, var_15)
    assert var_32 == 0
    var_33 = {var_1: var_19}
    var_34 = [var_3, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_3, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 2
    var_38 = [var_0, var_3, var_11]
    var_39 = module_0.get_in(var_38, var_35)
    assert var_39 is None
    var_40 = [var_0, var_3, var_11]
    var_41 = module_0.get_in(var_40, var_35, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'x'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = 0
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_42, no_default=var_49)
    var_51 = {var_47: var_50}
    var_52 = []
    var_53 = module_0.get_in(var_52, var_51)
    var_54 = None
    var_55 = {var_47: var_54}
    var_56 = [var_47]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_47]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 is None
    var_60 = [var_11]
    var_61 = module_0.get_in(var_60, var_55, var_15)
    assert var_61 == 0



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = [var_0, var_1, var_11]
    var_18 = module_0.get_in(var_17, var_6)
    assert var_18 is None
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_3, var_19, var_22]
    var_24 = [var_19, var_3]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 4
    var_26 = 0
    var_27 = [var_26]
    var_28 = module_0.get_in(var_27, var_23)
    assert var_28 == 1
    var_29 = 5
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_23)
    assert var_31 is None
    var_32 = [var_19, var_29]
    var_33 = module_0.get_in(var_32, var_23)
    assert var_33 is None
    var_34 = {var_1: var_19}
    var_35 = [var_3, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_3, var_1]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 2
    var_39 = [var_0, var_3, var_11]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = 5
    var_47 = [var_46]
    var_48 = 1
    var_49 = 2
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_47, var_50, no_default=var_51)
    var_53 = [var_11]
    var_54 = {}
    var_55 = module_0.get_in(var_53, var_54, var_26)
    assert var_55 == 0
    var_56 = 'y'
    var_57 = [var_11, var_56]
    var_58 = {}
    var_59 = []
    var_60 = module_0.get_in(var_57, var_58, var_59)
    var_61 = [var_29]
    var_62 = [var_49, var_19]
    var_63 = 'out_of_bounds'
    var_64 = module_0.get_in(var_61, var_62, var_63)
    assert var_64 == 'out_of_bounds'
    var_65 = {var_46: var_49}
    var_66 = []
    var_67 = module_0.get_in(var_66, var_65)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_65, var_15)
    var_70 = 123
    var_71 = [var_11]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 is None
    var_73 = [var_11]
    var_74 = 'not_subscriptable'
    var_75 = module_0.get_in(var_73, var_70, var_74)
    assert var_75 == 'not_subscriptable'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 'default'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'default'
    var_19 = [var_0, var_1, var_13]
    var_20 = module_0.get_in(var_19, var_6)
    assert var_20 is None
    var_21 = [var_0, var_1, var_13]
    var_22 = 0
    var_23 = module_0.get_in(var_21, var_6, var_22)
    assert var_23 == 0
    var_24 = 2
    var_25 = [var_3, var_24]
    var_26 = 3
    var_27 = 4
    var_28 = [var_26, var_27]
    var_29 = [var_25, var_28]
    var_30 = [var_22, var_3]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 2
    var_32 = [var_3, var_22]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 3
    var_34 = [var_24]
    var_35 = module_0.get_in(var_34, var_29)
    assert var_35 is None
    var_36 = [var_24]
    var_37 = 'out_of_bounds'
    var_38 = module_0.get_in(var_36, var_29, var_37)
    assert var_38 == 'out_of_bounds'
    var_39 = {var_1: var_24}
    var_40 = [var_3, var_39]
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_3, var_1]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 2
    var_44 = [var_0, var_3]
    var_45 = module_0.get_in(var_44, var_41)
    var_46 = [var_0, var_22]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 == 1
    var_48 = [var_0, var_24]
    var_49 = module_0.get_in(var_48, var_41)
    assert var_49 is None
    var_50 = {var_0: var_3}
    var_51 = 'b'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_50, no_default=var_53)
    var_55 = 0
    var_56 = [var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_50, no_default=var_57)
    var_59 = []
    var_60 = module_0.get_in(var_59, var_50)
    var_61 = 'y'
    var_62 = 'z'
    var_63 = [var_13, var_61, var_62]
    var_64 = {}
    var_65 = 'not_found'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'not_found'



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 'default'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 'default'
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 10
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = 'a'
    var_33 = 10
    var_34 = [var_32, var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_28, no_default=var_35)
    var_37 = {var_32: var_35}
    var_38 = [var_32, var_33]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 is None
    var_40 = 'a'
    var_41 = 'b'
    var_42 = [var_40, var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_37, no_default=var_43)
    var_45 = {var_40: var_43}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = 'name'
    var_49 = 'purchase'
    var_50 = 'credit card'
    var_51 = 'Alice'
    var_52 = 'items'
    var_53 = 'costs'
    var_54 = 'Apple'
    var_55 = 'Orange'
    var_56 = [var_54, var_55]
    var_57 = 0.5
    var_58 = 1.25
    var_59 = [var_57, var_58]
    var_60 = {var_52: var_56, var_53: var_59}
    var_61 = '5555-1234-1234-1234'
    var_62 = {var_48: var_51, var_49: var_60, var_50: var_61}
    var_63 = 0
    var_64 = [var_49, var_52, var_63]
    var_65 = [var_48]
    var_66 = 'total'
    var_67 = [var_49, var_66]
    var_68 = [var_49, var_66]



# Parsed testcases at query #19
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
    var_13 = 2
    var_14 = [var_3, var_13]
    var_15 = 3
    var_16 = 4
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = 0
    var_20 = [var_19, var_3]
    var_21 = module_0.get_in(var_20, var_18)
    assert var_21 == 2
    var_22 = [var_3, var_19]
    var_23 = module_0.get_in(var_22, var_18)
    assert var_23 == 3
    var_24 = 'x'
    var_25 = [var_24]
    var_26 = 'not found'
    var_27 = module_0.get_in(var_25, var_18, var_26)
    assert var_27 == 'not found'
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.get_in(var_28, var_18, var_19)
    assert var_29 == 0
    var_30 = 'x'
    var_31 = [var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_18, no_default=var_32)
    var_34 = {var_31: var_13}
    var_35 = [var_33, var_34]
    var_36 = {var_30: var_35}
    var_37 = [var_30, var_33, var_31]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 2
    var_39 = []
    var_40 = module_0.get_in(var_39, var_36)
    var_41 = {var_30: var_33}
    var_42 = [var_30, var_31]
    var_43 = 'error'
    var_44 = module_0.get_in(var_42, var_41, var_43)
    assert var_44 == 'error'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 0
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = [var_3, var_23]
    var_25 = [var_3, var_3, var_17]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 3
    var_27 = [var_3, var_3]
    var_28 = module_0.get_in(var_27, var_24)
    var_29 = [var_3]
    var_30 = module_0.get_in(var_29, var_24)
    var_31 = [var_19]
    var_32 = module_0.get_in(var_31, var_24)
    assert var_32 is None
    var_33 = [var_19]
    var_34 = module_0.get_in(var_33, var_24, var_17)
    assert var_34 == 0
    var_35 = {var_1: var_19}
    var_36 = [var_3, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_3, var_1]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_0, var_3]
    var_41 = module_0.get_in(var_40, var_37)
    var_42 = [var_0, var_17]
    var_43 = module_0.get_in(var_42, var_37)
    assert var_43 == 1
    var_44 = [var_0, var_19]
    var_45 = module_0.get_in(var_44, var_37)
    assert var_45 is None
    var_46 = [var_0, var_19]
    var_47 = module_0.get_in(var_46, var_37, var_17)
    assert var_47 == 0
    var_48 = {var_0: var_3}
    var_49 = 'b'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_52}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {var_49: var_52}
    var_57 = [var_50]
    var_58 = 'default'
    var_59 = module_0.get_in(var_57, var_56, var_58)
    assert var_59 == 'default'



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 'default'
    assert var_20 == 'Apple'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 'default'
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    assert var_27 == 'Alice'
    var_28 = {var_23: var_27}
    var_29 = 5
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = 'a'
    var_33 = 5
    var_34 = [var_32, var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_28, no_default=var_35)
    var_37 = {var_32: var_35}
    var_38 = [var_32, var_33]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 is None
    assert var_39 == 0
    var_40 = 'a'
    var_41 = 'b'
    var_42 = [var_40, var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_37, no_default=var_43)
    var_45 = {var_40: var_43}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = 'name'
    var_49 = 'purchase'
    var_50 = 'credit card'
    var_51 = 'Alice'
    var_52 = 'items'
    var_53 = 'costs'
    var_54 = 'Apple'
    var_55 = 'Orange'
    var_56 = [var_54, var_55]
    var_57 = 0.5
    var_58 = 1.25
    var_59 = [var_57, var_58]
    var_60 = {var_52: var_56, var_53: var_59}
    var_61 = '5555-1234-1234-1234'
    var_62 = {var_48: var_51, var_49: var_60, var_50: var_61}
    var_63 = 0
    var_64 = [var_49, var_52, var_63]
    var_65 = [var_48]
    var_66 = 'total'
    var_67 = [var_49, var_66]
    var_68 = [var_49, var_66]



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 'default'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 'default'
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 10
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = 'a'
    var_33 = 10
    var_34 = [var_32, var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_28, no_default=var_35)
    var_37 = {var_32: var_35}
    var_38 = [var_32, var_33]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 is None
    var_40 = 'a'
    var_41 = 'b'
    var_42 = [var_40, var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_37, no_default=var_43)
    var_45 = {var_40: var_43}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = [var_43, var_9, var_10]
    var_49 = {var_41: var_48}
    var_50 = [var_49]
    var_51 = {var_40: var_50}
    var_52 = 0
    var_53 = [var_40, var_52, var_41, var_43]
    var_54 = module_0.get_in(var_53, var_51)
    assert var_54 == 2



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'default'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'default'
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_1: var_3}
    var_23 = {var_0: var_22}
    var_24 = 'a'
    var_25 = 'c'
    var_26 = [var_24, var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_23, no_default=var_27)
    var_29 = [var_27, var_9, var_10]
    var_30 = {var_24: var_29}
    var_31 = 10
    var_32 = [var_24, var_31]
    var_33 = module_0.get_in(var_32, var_30, var_18)
    assert var_33 == 'default'
    var_34 = [var_24, var_31]
    var_35 = module_0.get_in(var_34, var_30)
    assert var_35 is None
    var_36 = [var_27, var_9, var_10]
    var_37 = {var_24: var_36}
    var_38 = 'a'
    var_39 = 10
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_37, no_default=var_41)
    var_43 = {var_38: var_41}
    var_44 = [var_38, var_39]
    var_45 = module_0.get_in(var_44, var_43, var_18)
    assert var_45 == 'default'
    var_46 = [var_38, var_39]
    var_47 = module_0.get_in(var_46, var_43)
    assert var_47 is None
    var_48 = {var_38: var_41}
    var_49 = []
    var_50 = module_0.get_in(var_49, var_48)
    var_51 = {var_39: var_41}
    var_52 = {var_40: var_9}
    var_53 = [var_51, var_52]
    var_54 = {var_38: var_53}
    var_55 = [var_38, var_41, var_40]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 == 2



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = [var_0, var_1, var_11]
    var_18 = 0
    var_19 = module_0.get_in(var_17, var_6, var_18)
    assert var_19 == 0
    var_20 = 2
    var_21 = [var_3, var_20]
    var_22 = 3
    var_23 = 4
    var_24 = [var_22, var_23]
    var_25 = [var_21, var_24]
    var_26 = [var_18, var_3]
    var_27 = module_0.get_in(var_26, var_25)
    assert var_27 == 2
    var_28 = [var_3]
    var_29 = module_0.get_in(var_28, var_25)
    var_30 = [var_20]
    var_31 = module_0.get_in(var_30, var_25)
    assert var_31 is None
    var_32 = [var_20]
    var_33 = module_0.get_in(var_32, var_25, var_15)
    assert var_33 == 'default'
    var_34 = {var_1: var_20}
    var_35 = [var_3, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_3, var_1]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 2
    var_39 = [var_0, var_3]
    var_40 = module_0.get_in(var_39, var_36)
    var_41 = [var_0, var_3, var_2]
    var_42 = module_0.get_in(var_41, var_36)
    assert var_42 is None
    var_43 = [var_0, var_3, var_2]
    var_44 = module_0.get_in(var_43, var_36, var_15)
    assert var_44 == 'default'
    var_45 = {var_0: var_3}
    var_46 = 'b'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_45, no_default=var_48)
    var_50 = {var_46: var_49}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = {var_46: var_49}
    var_54 = [var_47]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 is None
    var_56 = [var_46, var_47]
    var_57 = module_0.get_in(var_56, var_53)
    assert var_57 is None
    var_58 = {var_46: var_49}
    var_59 = [var_46, var_47]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_46, var_47]
    var_62 = module_0.get_in(var_61, var_58, var_15)
    assert var_62 == 'default'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 'default'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'default'
    var_19 = [var_0, var_1, var_13]
    var_20 = 0
    var_21 = module_0.get_in(var_19, var_6, var_20)
    assert var_21 == 0
    var_22 = 2
    var_23 = [var_3, var_22]
    var_24 = 3
    var_25 = 4
    var_26 = [var_24, var_25]
    var_27 = [var_23, var_26]
    var_28 = [var_20, var_3]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 2
    var_30 = [var_3, var_20]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 3
    var_32 = [var_20]
    var_33 = module_0.get_in(var_32, var_27)
    var_34 = [var_22]
    var_35 = module_0.get_in(var_34, var_27)
    assert var_35 is None
    var_36 = [var_22]
    var_37 = module_0.get_in(var_36, var_27, var_17)
    assert var_37 == 'default'
    var_38 = {var_1: var_22}
    var_39 = [var_3, var_38]
    var_40 = {var_0: var_39}
    var_41 = [var_0, var_3, var_1]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 == 2
    var_43 = [var_0, var_3]
    var_44 = module_0.get_in(var_43, var_40)
    var_45 = [var_0, var_20]
    var_46 = module_0.get_in(var_45, var_40)
    assert var_46 == 1
    var_47 = [var_0, var_22]
    var_48 = module_0.get_in(var_47, var_40)
    assert var_48 is None
    var_49 = {var_0: var_3}
    var_50 = 'b'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_49, no_default=var_52)
    var_54 = 0
    var_55 = [var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_49, no_default=var_56)
    var_58 = {var_54: var_57}
    var_59 = []
    var_60 = module_0.get_in(var_59, var_58)
    var_61 = None
    var_62 = {var_55: var_61}
    var_63 = {var_54: var_62}
    var_64 = [var_54, var_55]
    var_65 = module_0.get_in(var_64, var_63)
    assert var_65 is None
    var_66 = [var_54, var_55]
    var_67 = module_0.get_in(var_66, var_63, var_17)
    assert var_67 is None
    var_68 = [var_54, var_56]
    var_69 = module_0.get_in(var_68, var_63, var_17)
    assert var_69 == 'default'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_3, var_15]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 3
    var_25 = [var_15, var_3]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 2
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_15]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 == 1
    var_40 = [var_11]
    var_41 = module_0.get_in(var_40, var_33)
    assert var_41 is None
    var_42 = {var_0: var_3}
    var_43 = [var_0]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 1
    var_45 = 'x'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_42, no_default=var_47)
    var_49 = {var_45: var_48}
    var_50 = [var_45, var_46]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 is None
    var_52 = [var_45, var_46]
    var_53 = module_0.get_in(var_52, var_49, var_15)
    assert var_53 == 0
    var_54 = {var_45: var_48}
    var_55 = []
    var_56 = module_0.get_in(var_55, var_54)
    var_57 = None
    var_58 = {var_46: var_57}
    var_59 = {var_45: var_58}
    var_60 = [var_45, var_46]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 is None
    var_62 = [var_45, var_46]
    var_63 = module_0.get_in(var_62, var_59, var_15)
    assert var_63 is None
    var_64 = [var_45, var_47]
    var_65 = module_0.get_in(var_64, var_59, var_15)
    assert var_65 == 0



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3, var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 3
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_17]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = {var_0: var_3}
    var_41 = [var_0]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 == 1
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_40, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = [var_43, var_44]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 is None
    var_50 = [var_43, var_44]
    var_51 = module_0.get_in(var_50, var_47, var_15)
    assert var_51 == 0
    var_52 = {var_43: var_46}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = None
    var_56 = {var_44: var_55}
    var_57 = {var_43: var_56}
    var_58 = [var_43, var_44]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_43, var_44]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 is None
    var_62 = [var_43, var_45]
    var_63 = module_0.get_in(var_62, var_57, var_15)
    assert var_63 == 0



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'not found'
    var_20 = [var_3, var_9, var_10]
    assert var_20 == 'Apple'
    var_21 = {var_0: var_20}
    var_22 = 10
    var_23 = [var_0, var_22]
    assert var_23 == 'Alice'
    var_24 = 'out of range'
    var_25 = module_0.get_in(var_23, var_21, var_24)
    assert var_25 == 'out of range'
    var_26 = {var_1: var_3}
    assert var_26 is None
    var_27 = {var_0: var_26}
    var_28 = 'a'
    var_29 = 'c'
    var_30 = [var_28, var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_27, no_default=var_31)
    var_33 = [var_31, var_9, var_10]
    var_34 = {var_28: var_33}
    var_35 = 'a'
    var_36 = 10
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_36: var_38}
    assert var_40 == 0
    var_41 = {var_35: var_40}
    var_42 = [var_35, var_37]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 is None
    var_44 = {var_35: var_38}
    var_45 = []
    var_46 = module_0.get_in(var_45, var_44)
    var_47 = 'name'
    var_48 = 'purchase'
    var_49 = 'credit card'
    var_50 = 'Alice'
    var_51 = 'items'
    var_52 = 'costs'
    var_53 = 'Apple'
    var_54 = 'Orange'
    var_55 = [var_53, var_54]
    var_56 = 0.5
    var_57 = 1.25
    var_58 = [var_56, var_57]
    var_59 = {var_51: var_55, var_52: var_58}
    var_60 = '5555-1234-1234-1234'
    var_61 = {var_47: var_50, var_48: var_59, var_49: var_60}
    var_62 = 0
    var_63 = [var_48, var_51, var_62]
    var_64 = [var_47]
    var_65 = 'total'
    var_66 = [var_48, var_65]
    var_67 = [var_48, var_65]



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'not found'
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 10
    var_23 = [var_0, var_22]
    var_24 = 'out of range'
    var_25 = module_0.get_in(var_23, var_21, var_24)
    assert var_25 == 'out of range'
    var_26 = {var_0: var_3}
    var_27 = 'b'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_26, no_default=var_29)
    var_31 = [var_30, var_9, var_10]
    var_32 = 10
    var_33 = [var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_31, no_default=var_34)
    var_36 = {var_32: var_35}
    var_37 = [var_33]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 is None
    var_39 = {var_32: var_35}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_33: var_35}
    var_43 = {var_34: var_9}
    var_44 = [var_42, var_43]
    var_45 = {var_32: var_44}
    var_46 = [var_32, var_35, var_34]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 2



# Parsed testcases at query #5
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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 is None
    var_37 = {var_0: var_3}
    var_38 = 'b'
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_37, no_default=var_40)
    var_42 = 'y'
    var_43 = [var_11, var_42]
    var_44 = 'not_found'
    var_45 = module_0.get_in(var_43, var_37, var_44)
    assert var_45 == 'not_found'
    var_46 = []
    var_47 = module_0.get_in(var_46, var_37)
    var_48 = 'string'
    var_49 = {var_38: var_48}
    var_50 = [var_38, var_23]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 is None
    var_52 = [var_38, var_23]
    var_53 = module_0.get_in(var_52, var_49, var_15)
    assert var_53 == 'default'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_13]
    var_17 = module_0.get_in(var_16, var_6)
    assert var_17 is None
    var_18 = [var_0, var_1, var_13]
    var_19 = module_0.get_in(var_18, var_6)
    assert var_19 is None
    var_20 = 2
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = [var_3, var_24]
    var_26 = [var_3]
    var_27 = module_0.get_in(var_26, var_25)
    var_28 = [var_3, var_3]
    var_29 = module_0.get_in(var_28, var_25)
    var_30 = 0
    var_31 = [var_3, var_3, var_30]
    var_32 = module_0.get_in(var_31, var_25)
    assert var_32 == 3
    var_33 = [var_20]
    var_34 = module_0.get_in(var_33, var_25)
    assert var_34 is None
    var_35 = [var_3, var_20]
    var_36 = module_0.get_in(var_35, var_25)
    assert var_36 is None
    var_37 = [var_13]
    var_38 = {}
    var_39 = module_0.get_in(var_37, var_38, var_30)
    assert var_39 == 0
    var_40 = [var_0, var_13]
    var_41 = {var_0: var_3}
    var_42 = []
    var_43 = module_0.get_in(var_40, var_41, var_42)
    var_44 = [var_3]
    var_45 = [var_3, var_20]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_44, var_45, var_46)
    assert var_47 == 2
    var_48 = [var_20]
    var_49 = [var_3, var_20]
    var_50 = module_0.get_in(var_48, var_49, var_46)
    assert var_50 == 'missing'
    var_51 = 'x'
    var_52 = [var_51]
    var_53 = {}
    var_54 = True
    var_55 = module_0.get_in(var_52, var_53, no_default=var_54)
    var_56 = 2
    var_57 = [var_56]
    var_58 = 1
    var_59 = [var_58, var_56]
    var_60 = True
    var_61 = module_0.get_in(var_57, var_59, no_default=var_60)
    var_62 = {var_57: var_20}
    var_63 = [var_59, var_62]
    var_64 = {var_56: var_63}
    var_65 = [var_56, var_59, var_57]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 == 2
    var_67 = [var_56, var_59]
    var_68 = module_0.get_in(var_67, var_64)
    var_69 = [var_56, var_30]
    var_70 = module_0.get_in(var_69, var_64)
    assert var_70 == 1
    var_71 = [var_56, var_20]
    var_72 = module_0.get_in(var_71, var_64)
    assert var_72 is None
    var_73 = []
    var_74 = module_0.get_in(var_73, var_64)
    var_75 = []
    var_76 = {}
    var_77 = module_0.get_in(var_75, var_76, var_30)
    var_78 = []
    var_79 = [var_59, var_20, var_21]
    var_80 = module_0.get_in(var_78, var_79)
    var_81 = {var_56: var_59}
    var_82 = [var_56, var_57]
    var_83 = module_0.get_in(var_82, var_81)
    assert var_83 is None
    var_84 = [var_56, var_57]
    var_85 = module_0.get_in(var_84, var_81, var_30)
    assert var_85 == 0



# Parsed testcases at query #7
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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 0
    assert var_20 == 'Apple'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 0
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    assert var_27 == 'Alice'
    var_28 = {var_23: var_27}
    var_29 = 'a'
    var_30 = 10
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_28, no_default=var_32)
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = 'name'
    var_44 = 'purchase'
    var_45 = 'credit card'
    var_46 = 'Alice'
    var_47 = 'items'
    var_48 = 'costs'
    var_49 = 'Apple'
    var_50 = 'Orange'
    var_51 = [var_49, var_50]
    var_52 = 0.5
    var_53 = 1.25
    var_54 = [var_52, var_53]
    var_55 = {var_47: var_51, var_48: var_54}
    var_56 = '5555-1234-1234-1234'
    var_57 = {var_43: var_46, var_44: var_55, var_45: var_56}
    var_58 = 0
    var_59 = [var_44, var_47, var_58]
    var_60 = [var_43]
    var_61 = 'total'
    var_62 = [var_44, var_61]
    var_63 = [var_44, var_61]



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 0
    var_19 = 2
    var_20 = [var_3, var_19]
    var_21 = 3
    var_22 = [var_20, var_21]
    var_23 = 4
    var_24 = [var_22, var_23]
    var_25 = [var_17, var_17, var_3]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 2
    var_27 = [var_17, var_3]
    var_28 = module_0.get_in(var_27, var_24)
    assert var_28 == 3
    var_29 = [var_3]
    var_30 = module_0.get_in(var_29, var_24)
    assert var_30 == 4
    var_31 = [var_19]
    var_32 = module_0.get_in(var_31, var_24)
    assert var_32 is None
    var_33 = [var_19]
    var_34 = module_0.get_in(var_33, var_24, var_17)
    assert var_34 == 0
    var_35 = {var_1: var_19}
    var_36 = [var_3, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_3, var_1]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 1
    var_42 = [var_0, var_3]
    var_43 = module_0.get_in(var_42, var_37)
    var_44 = [var_13]
    var_45 = module_0.get_in(var_44, var_37)
    assert var_45 is None
    var_46 = [var_13]
    var_47 = module_0.get_in(var_46, var_37, var_17)
    assert var_47 == 0
    var_48 = {var_0: var_3}
    var_49 = 'x'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_52}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {var_49: var_52}
    var_57 = [var_49, var_50]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 is None
    var_59 = [var_49, var_50]
    var_60 = module_0.get_in(var_59, var_56, var_17)
    assert var_60 == 0



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
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_17]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = {var_43: var_46}
    var_51 = [var_43, var_44]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_43, var_44]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0



# Parsed testcases at query #10
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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 10
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 10
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 5
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 10
    var_25 = {var_1: var_3}
    var_26 = {var_0: var_25}
    var_27 = 'a'
    var_28 = 'c'
    var_29 = [var_27, var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_26, no_default=var_30)
    var_32 = [var_30, var_9, var_10]
    var_33 = {var_27: var_32}
    var_34 = 'a'
    var_35 = 5
    var_36 = [var_34, var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_33, no_default=var_37)
    var_39 = {var_35: var_37}
    var_40 = {var_36: var_9}
    var_41 = [var_39, var_40]
    var_42 = {var_34: var_41}
    var_43 = [var_34, var_37, var_36]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 2
    var_45 = 'name'
    var_46 = 'purchase'
    var_47 = 'credit card'
    var_48 = 'Alice'
    var_49 = 'items'
    var_50 = 'costs'
    var_51 = 'Apple'
    var_52 = 'Orange'
    var_53 = [var_51, var_52]
    var_54 = 0.5
    var_55 = 1.25
    var_56 = [var_54, var_55]
    var_57 = {var_49: var_53, var_50: var_56}
    var_58 = '5555-1234-1234-1234'
    var_59 = {var_45: var_48, var_46: var_57, var_47: var_58}
    var_60 = 0
    var_61 = [var_46, var_49, var_60]
    var_62 = [var_45]
    var_63 = 'total'
    var_64 = [var_46, var_63]
    var_65 = [var_46, var_63]
    var_66 = 'string'
    var_67 = {var_34: var_66}
    var_68 = [var_34, var_35]
    var_69 = module_0.get_in(var_68, var_67, var_18)
    assert var_69 == 10



# Parsed testcases at query #11
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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 'default'
    assert var_20 == 'Apple'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 'default'
    var_22 = {var_1: var_3}
    assert var_22 == 'Alice'
    var_23 = {var_0: var_22}
    var_24 = 'a'
    var_25 = 'c'
    var_26 = [var_24, var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_23, no_default=var_27)
    var_29 = [var_27, var_9, var_10]
    var_30 = {var_24: var_29}
    var_31 = 10
    var_32 = [var_24, var_31]
    assert var_32 is None
    var_33 = module_0.get_in(var_32, var_30)
    assert var_33 is None
    var_34 = 'a'
    var_35 = 10
    var_36 = [var_34, var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_30, no_default=var_37)
    var_39 = {var_34: var_37}
    var_40 = [var_34, var_35]
    assert var_40 == 0
    var_41 = module_0.get_in(var_40, var_39)
    assert var_41 is None
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_39, no_default=var_45)
    var_47 = {var_42: var_45}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = 'name'
    var_51 = 'purchase'
    var_52 = 'credit card'
    var_53 = 'Alice'
    var_54 = 'items'
    var_55 = 'costs'
    var_56 = 'Apple'
    var_57 = 'Orange'
    var_58 = [var_56, var_57]
    var_59 = 0.5
    var_60 = 1.25
    var_61 = [var_59, var_60]
    var_62 = {var_54: var_58, var_55: var_61}
    var_63 = '5555-1234-1234-1234'
    var_64 = {var_50: var_53, var_51: var_62, var_52: var_63}
    var_65 = 0
    var_66 = [var_51, var_54, var_65]
    var_67 = [var_50]
    var_68 = 'total'
    var_69 = [var_51, var_68]
    var_70 = [var_51, var_68]



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = [var_17]
    var_31 = module_0.get_in(var_30, var_22, var_15)
    assert var_31 == 'default'
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_34)
    var_39 = [var_0, var_17]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = [var_0, var_17]
    var_42 = module_0.get_in(var_41, var_34, var_15)
    assert var_42 == 'default'
    var_43 = {var_0: var_3}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = {var_44: var_47}
    var_49 = []
    var_50 = module_0.get_in(var_49, var_48)
    var_51 = {var_45: var_47}
    var_52 = {var_44: var_51}
    var_53 = [var_44, var_46]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = [var_44, var_46]
    var_56 = module_0.get_in(var_55, var_52, var_15)
    assert var_56 == 'default'
    var_57 = {var_44: var_47}
    var_58 = [var_44, var_45]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_44, var_45]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 == 'default'



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 42
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 42
    var_20 = [var_3, var_9]
    var_21 = {var_0: var_20}
    var_22 = 5
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 42
    var_25 = {var_0: var_3}
    var_26 = 'b'
    var_27 = [var_26]
    var_28 = True
    var_29 = module_0.get_in(var_27, var_25, no_default=var_28)
    var_30 = [var_29, var_9, var_10]
    var_31 = 5
    var_32 = [var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_30, no_default=var_33)
    var_35 = {var_32: var_34}
    var_36 = {var_32: var_9}
    var_37 = [var_35, var_36]
    var_38 = {var_31: var_37}
    var_39 = [var_31, var_34, var_32]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 2
    var_41 = {var_31: var_34}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_31: var_34}
    var_45 = [var_31, var_32]
    var_46 = module_0.get_in(var_45, var_44, var_18)
    assert var_46 == 42



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = [var_0, var_1, var_11]
    var_18 = 0
    var_19 = module_0.get_in(var_17, var_6, var_18)
    assert var_19 == 0
    var_20 = 2
    var_21 = [var_3, var_20]
    var_22 = 3
    var_23 = 4
    var_24 = [var_22, var_23]
    var_25 = [var_21, var_24]
    var_26 = [var_18, var_3]
    var_27 = module_0.get_in(var_26, var_25)
    assert var_27 == 2
    var_28 = [var_3, var_18]
    var_29 = module_0.get_in(var_28, var_25)
    assert var_29 == 3
    var_30 = [var_20]
    var_31 = module_0.get_in(var_30, var_25)
    assert var_31 is None
    var_32 = [var_20]
    var_33 = module_0.get_in(var_32, var_25, var_15)
    assert var_33 == 'default'
    var_34 = {var_1: var_20}
    var_35 = [var_3, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_3, var_1]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 2
    var_39 = [var_0, var_3, var_11]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = 0
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_41, no_default=var_48)
    var_50 = {var_46: var_49}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = None
    var_54 = {var_47: var_53}
    var_55 = {var_46: var_54}
    var_56 = [var_46, var_47]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_46, var_47]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 is None
    var_60 = [var_46, var_11]
    var_61 = module_0.get_in(var_60, var_55, var_15)
    assert var_61 == 'default'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = 'y'
    var_13 = [var_11, var_12]
    var_14 = module_0.get_in(var_13, var_6)
    assert var_14 is None
    var_15 = [var_11, var_12]
    var_16 = 'default'
    var_17 = module_0.get_in(var_15, var_6, var_16)
    assert var_17 == 'default'
    var_18 = 2
    var_19 = [var_3, var_18]
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = 0
    var_25 = [var_24, var_3]
    var_26 = module_0.get_in(var_25, var_23)
    assert var_26 == 2
    var_27 = [var_3, var_24]
    var_28 = module_0.get_in(var_27, var_23)
    assert var_28 == 3
    var_29 = [var_18, var_24]
    var_30 = module_0.get_in(var_29, var_23)
    assert var_30 is None
    var_31 = [var_18, var_24]
    var_32 = module_0.get_in(var_31, var_23, var_16)
    assert var_32 == 'default'
    var_33 = {var_1: var_18}
    var_34 = [var_3, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_3, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 2
    var_38 = [var_0, var_24]
    var_39 = module_0.get_in(var_38, var_35)
    assert var_39 == 1
    var_40 = [var_0, var_3, var_2]
    var_41 = module_0.get_in(var_40, var_35)
    assert var_41 is None
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = {var_43: var_46}
    var_51 = [var_43, var_44]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_43, var_44]
    var_54 = module_0.get_in(var_53, var_50, var_16)
    assert var_54 == 'default'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = [var_17]
    var_31 = module_0.get_in(var_30, var_22, var_23)
    assert var_31 == 0
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_34)
    var_39 = [var_0, var_17]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'b'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = [var_43]
    var_47 = module_0.get_in(var_46, var_41, var_15)
    assert var_47 == 'default'
    var_48 = []
    var_49 = module_0.get_in(var_48, var_41)
    var_50 = {var_42: var_45}
    var_51 = [var_42, var_43]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_42, var_43]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 'default'



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = [var_18, var_19]
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = 0
    var_24 = [var_23, var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 4
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_23]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 1
    var_37 = [var_0, var_3, var_2]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = {var_0: var_3}
    var_40 = 'b'
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_39, no_default=var_42)
    var_44 = [var_43, var_17, var_19]
    var_45 = 5
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_44, no_default=var_47)
    var_49 = 123
    var_50 = 0
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_49, no_default=var_52)
    var_54 = {var_50: var_53}
    var_55 = []
    var_56 = module_0.get_in(var_55, var_54)
    var_57 = {var_50: var_53}
    var_58 = [var_50]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 == 1
    var_60 = [var_51]
    var_61 = module_0.get_in(var_60, var_57)
    assert var_61 is None



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 'default'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'default'
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = 0
    var_24 = [var_23, var_3]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 2
    var_26 = [var_3, var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 3
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_22)
    assert var_29 is None
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_3, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 is None
    var_37 = {var_0: var_3}
    var_38 = 'b'
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_37, no_default=var_40)
    var_42 = 'name'
    var_43 = 'purchase'
    var_44 = 'credit card'
    var_45 = 'Alice'
    var_46 = 'items'
    var_47 = 'costs'
    var_48 = 'Apple'
    var_49 = 'Orange'
    var_50 = [var_48, var_49]
    var_51 = 0.5
    var_52 = 1.25
    var_53 = [var_51, var_52]
    var_54 = {var_46: var_50, var_47: var_53}
    var_55 = '5555-1234-1234-1234'
    var_56 = {var_42: var_45, var_43: var_54, var_44: var_55}
    var_57 = [var_43, var_46, var_23]
    var_58 = [var_42]
    var_59 = 'total'
    var_60 = [var_43, var_59]
    var_61 = [var_43, var_59]
    var_62 = 'string'
    var_63 = {var_38: var_62}
    var_64 = [var_38, var_23]
    var_65 = module_0.get_in(var_64, var_63)
    assert var_65 is None
    var_66 = [var_38, var_23]
    var_67 = module_0.get_in(var_66, var_63, var_15)
    assert var_67 == 'default'



# Parsed testcases at query #19
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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'default'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'default'
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9]
    var_28 = 4
    var_29 = [var_10, var_28]
    var_30 = [var_27, var_29]
    var_31 = {var_23: var_30}
    var_32 = 0
    var_33 = [var_23, var_26, var_32]
    var_34 = module_0.get_in(var_33, var_31)
    assert var_34 == 3
    var_35 = {var_23: var_26}
    var_36 = []
    var_37 = module_0.get_in(var_36, var_35)
    var_38 = {var_23: var_26}
    var_39 = [var_23, var_24]
    var_40 = 'error'
    var_41 = module_0.get_in(var_39, var_38, var_40)
    assert var_41 == 'error'



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 42
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 42
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 'a'
    var_30 = 10
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_28, no_default=var_32)
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = 'a'
    var_44 = 'b'
    var_45 = 2
    var_46 = {var_44: var_45}
    var_47 = {var_43: var_46}
    var_48 = [var_43, var_44]
    var_49 = module_0.get_in(var_48, var_40)
    assert var_49 == 2



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'default'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'default'
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 'a'
    var_30 = 10
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_28, no_default=var_32)
    var_34 = [var_32, var_9, var_10]
    var_35 = {var_30: var_34}
    var_36 = 4
    var_37 = {var_31: var_36}
    var_38 = [var_35, var_37]
    var_39 = {var_29: var_38}
    var_40 = 0
    var_41 = [var_29, var_40, var_30, var_32]
    var_42 = module_0.get_in(var_41, var_39)
    assert var_42 == 2
    var_43 = [var_29, var_32, var_31]
    var_44 = module_0.get_in(var_43, var_39)
    assert var_44 == 4
    var_45 = {var_29: var_32}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = {var_29: var_32}
    var_49 = [var_29, var_30]
    var_50 = module_0.get_in(var_49, var_48, var_18)
    assert var_50 == 'default'



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 'default'
    assert var_20 == 'Apple'
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 'default'
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    assert var_27 == 'Alice'
    var_28 = {var_23: var_27}
    var_29 = 5
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 'default'
    assert var_33 == 0
    var_34 = {var_23: var_26}
    var_35 = [var_23, var_24]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_23, var_24]
    var_38 = module_0.get_in(var_37, var_34, var_20)
    assert var_38 == 'default'
    var_39 = {var_23: var_26}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_24: var_26}
    var_43 = {var_25: var_9}
    var_44 = [var_42, var_43]
    var_45 = {var_23: var_44}
    var_46 = 0
    var_47 = [var_23, var_46, var_24]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 1
    var_49 = [var_23, var_26, var_25]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 == 2
    var_51 = 'name'
    var_52 = 'purchase'
    var_53 = 'credit card'
    var_54 = 'Alice'
    var_55 = 'items'
    var_56 = 'costs'
    var_57 = 'Apple'
    var_58 = 'Orange'
    var_59 = [var_57, var_58]
    var_60 = 0.5
    var_61 = 1.25
    var_62 = [var_60, var_61]
    var_63 = {var_55: var_59, var_56: var_62}
    var_64 = '5555-1234-1234-1234'
    var_65 = {var_51: var_54, var_52: var_63, var_53: var_64}
    var_66 = 0
    var_67 = [var_52, var_55, var_66]
    var_68 = [var_51]
    var_69 = 'total'
    var_70 = [var_52, var_69]
    var_71 = [var_52, var_69]



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
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = {var_1: var_3}
    var_16 = {var_0: var_15}
    var_17 = [var_0, var_2]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'not found'
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 'a'
    var_30 = 10
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_28, no_default=var_32)
    var_34 = {var_31: var_32}
    var_35 = {var_30: var_34}
    var_36 = {var_29: var_35}
    var_37 = 'd'
    var_38 = [var_29, var_30, var_37]
    var_39 = 0
    var_40 = module_0.get_in(var_38, var_36, var_39)
    assert var_40 == 0
    var_41 = {var_29: var_32}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_29: var_32}
    var_45 = [var_29, var_30]
    var_46 = 'error'
    var_47 = module_0.get_in(var_45, var_44, var_46)
    assert var_47 == 'error'
    var_48 = (var_32, var_9, var_10)
    var_49 = {var_30: var_48}
    var_50 = [var_49]
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_39, var_30, var_32]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 == 2



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = [var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_13]
    var_17 = 0
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 0
    var_19 = 2
    var_20 = [var_3, var_19]
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = 5
    var_26 = 6
    var_27 = [var_25, var_26]
    var_28 = 7
    var_29 = 8
    var_30 = [var_28, var_29]
    var_31 = [var_27, var_30]
    var_32 = [var_24, var_31]
    var_33 = [var_17, var_3, var_3]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 4
    var_35 = [var_3, var_17]
    var_36 = module_0.get_in(var_35, var_32)
    var_37 = [var_19]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = [var_19]
    var_40 = module_0.get_in(var_39, var_32, var_17)
    assert var_40 == 0
    var_41 = [var_3, var_19, var_21]
    var_42 = {var_1: var_41}
    var_43 = [var_22, var_25, var_26]
    var_44 = {var_2: var_43}
    var_45 = [var_42, var_44]
    var_46 = {var_0: var_45}
    var_47 = [var_0, var_17, var_1, var_3]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2
    var_49 = [var_0, var_3, var_2, var_17]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 4
    var_51 = [var_0, var_19]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_0, var_19]
    var_54 = module_0.get_in(var_53, var_46, var_17)
    assert var_54 == 0
    var_55 = {var_1: var_3}
    var_56 = {var_0: var_55}
    var_57 = 'x'
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_56, no_default=var_59)
    var_61 = 'a'
    var_62 = 'x'
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_56, no_default=var_64)
    var_66 = {var_61: var_64}
    var_67 = []
    var_68 = module_0.get_in(var_67, var_66)
    var_69 = {var_62: var_64}
    var_70 = {var_61: var_69}
    var_71 = [var_61, var_63]
    var_72 = module_0.get_in(var_71, var_70, var_17)
    assert var_72 == 0
    var_73 = 'y'
    var_74 = [var_13, var_73]
    var_75 = module_0.get_in(var_74, var_70, var_17)
    assert var_75 == 0
    var_76 = {var_61: var_64}
    var_77 = [var_61, var_62]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 is None
    var_79 = [var_61, var_62]
    var_80 = module_0.get_in(var_79, var_76, var_17)
    assert var_80 == 0



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
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = 'x'
    var_12 = [var_11]
    var_13 = module_0.get_in(var_12, var_6)
    assert var_13 is None
    var_14 = [var_11]
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = [var_3, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = [var_19, var_20]
    var_22 = [var_18, var_21]
    var_23 = [var_15, var_3]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 2
    var_25 = [var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_17]
    var_30 = module_0.get_in(var_29, var_22, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_17]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = {var_43: var_46}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = 123
    var_51 = [var_43]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_43]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0
    var_55 = 'a'
    var_56 = [var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_50, no_default=var_57)



