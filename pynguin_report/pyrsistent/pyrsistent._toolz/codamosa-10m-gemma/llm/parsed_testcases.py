####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'h'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = [var_3, var_18]
    var_20 = {var_0: var_3, var_1: var_15, var_2: var_19}
    var_21 = [var_0]
    var_22 = module_0.get_in(var_21, var_20)
    assert var_22 == 1
    var_23 = [var_1, var_4]
    var_24 = module_0.get_in(var_23, var_20)
    assert var_24 == 2
    var_25 = [var_1, var_5, var_3]
    var_26 = module_0.get_in(var_25, var_20)
    assert var_26 == 20
    var_27 = [var_1, var_6, var_12]
    var_28 = module_0.get_in(var_27, var_20)
    assert var_28 == 'hello'
    var_29 = [var_2, var_3, var_16]
    var_30 = module_0.get_in(var_29, var_20)
    assert var_30 == 5
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_20)
    assert var_33 is None
    var_34 = 'nonexistent'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_20)
    assert var_36 is None
    var_37 = [var_1, var_5, var_8]
    var_38 = module_0.get_in(var_37, var_20)
    assert var_38 is None
    var_39 = [var_1, var_6, var_34]
    var_40 = module_0.get_in(var_39, var_20)
    assert var_40 is None
    var_41 = [var_31]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_20, var_42)
    assert var_43 == 'missing'
    var_44 = [var_1, var_34]
    var_45 = 0
    var_46 = module_0.get_in(var_44, var_20, var_45)
    assert var_46 == 0
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_20, no_default=var_49)
    var_51 = 'b'
    var_52 = 'nonexistent'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_20, no_default=var_54)
    var_56 = 'b'
    var_57 = 'd'
    var_58 = 10
    var_59 = [var_56, var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_20, no_default=var_60)
    var_62 = 'a'
    var_63 = 'too_deep'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_20, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_20)
    var_69 = [var_62]
    var_70 = {}
    var_71 = 'fallback'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'fallback'
    var_73 = 'a'
    var_74 = [var_73]
    var_75 = {}
    var_76 = True
    var_77 = module_0.get_in(var_74, var_75, no_default=var_76)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = 'fallback'
    var_37 = module_0.get_in(var_35, var_21, var_36)
    assert var_37 == 'fallback'
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = [var_1, var_5, var_38]
    var_42 = module_0.get_in(var_41, var_21, var_36)
    assert var_42 == 'fallback'
    var_43 = 'not_an_index'
    var_44 = [var_0, var_43]
    var_45 = module_0.get_in(var_44, var_21)
    assert var_45 is None
    var_46 = [var_0, var_43]
    var_47 = module_0.get_in(var_46, var_21, var_36)
    assert var_47 == 'fallback'
    var_48 = 'b'
    var_49 = 'missing'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_21, no_default=var_51)
    var_53 = 'b'
    var_54 = 'd'
    var_55 = 5
    var_56 = [var_53, var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_21, no_default=var_57)
    var_59 = 'a'
    var_60 = 'not_an_index'
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_21, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_21)
    var_66 = 'any'
    var_67 = [var_66]
    var_68 = {}
    var_69 = 'empty'
    var_70 = module_0.get_in(var_67, var_68, var_69)
    assert var_70 == 'empty'
    var_71 = 'any'
    var_72 = [var_71]
    var_73 = {}
    var_74 = True
    var_75 = module_0.get_in(var_72, var_73, no_default=var_74)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = 'z'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'missing'
    var_31 = [var_1, var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = [var_1, var_4, var_6]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'nonexistent'
    var_36 = [var_1, var_4, var_20, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_27]
    var_39 = module_0.get_in(var_38, var_17, var_30)
    assert var_39 == 'missing'
    var_40 = [var_1, var_30]
    var_41 = module_0.get_in(var_40, var_17, var_20)
    assert var_41 == 0
    var_42 = 'z'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_17, no_default=var_44)
    var_46 = 'b'
    var_47 = 'missing'
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'f'
    var_52 = 10
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'a'
    var_57 = 0
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = [var_57, var_5]
    var_64 = module_0.get_in(var_63, var_17)
    assert var_64 is None



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_18]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_18, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 is True
    var_32 = 'x'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = 'nonexistent'
    var_42 = [var_1, var_35, var_41]
    var_43 = module_0.get_in(var_42, var_21)
    assert var_43 is None
    var_44 = [var_32]
    var_45 = 'missing'
    var_46 = module_0.get_in(var_44, var_21, var_45)
    assert var_46 == 'missing'
    var_47 = [var_1, var_35]
    var_48 = 0
    var_49 = module_0.get_in(var_47, var_21, var_48)
    assert var_49 == 0
    var_50 = [var_1, var_5, var_38]
    var_51 = 'error'
    var_52 = module_0.get_in(var_50, var_21, var_51)
    assert var_52 == 'error'
    var_53 = 'x'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'z'
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_21, no_default=var_60)
    var_62 = 'b'
    var_63 = 'd'
    var_64 = 5
    var_65 = [var_62, var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_21, no_default=var_66)
    var_68 = 'a'
    var_69 = 'not_an_index'
    var_70 = [var_68, var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_21, no_default=var_71)
    var_73 = 'b'
    var_74 = 'c'
    var_75 = 'too_deep'
    var_76 = [var_73, var_74, var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_21, no_default=var_77)
    var_79 = []
    var_80 = module_0.get_in(var_79, var_21)
    var_81 = [var_73]
    var_82 = {var_73: var_16}
    var_83 = module_0.get_in(var_81, var_82)
    assert var_83 is None
    var_84 = [var_73]
    var_85 = {}
    var_86 = 'fallback'
    var_87 = module_0.get_in(var_84, var_85, var_86)
    assert var_87 == 'fallback'
    var_88 = [var_73]
    var_89 = 3
    var_90 = [var_18, var_7, var_89]
    var_91 = module_0.get_in(var_88, var_90, var_86)
    assert var_91 == 'fallback'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_18]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_18, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 is True
    var_32 = 0
    var_33 = [var_2, var_32]
    var_34 = module_0.get_in(var_33)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = 'non_existent'
    var_42 = 'key'
    var_43 = [var_41, var_42]
    var_44 = module_0.get_in(var_43, var_21)
    assert var_44 is None
    var_45 = 'too_deep'
    var_46 = [var_0, var_45]
    var_47 = module_0.get_in(var_46, var_21)
    assert var_47 is None
    var_48 = [var_1, var_35]
    var_49 = 'missing'
    var_50 = module_0.get_in(var_48, var_21, var_49)
    assert var_50 == 'missing'
    var_51 = [var_1, var_5, var_38]
    var_52 = module_0.get_in(var_51, var_21, var_32)
    assert var_52 == 0
    var_53 = 'b'
    var_54 = 'z'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_21, no_default=var_56)
    var_58 = 'b'
    var_59 = 'd'
    var_60 = 5
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_21, no_default=var_62)
    var_64 = 'a'
    var_65 = 0
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_21, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_21)
    var_71 = [var_64]
    var_72 = {}
    var_73 = 'empty'
    var_74 = module_0.get_in(var_71, var_72, var_73)
    assert var_74 == 'empty'
    var_75 = 'a'
    var_76 = [var_75]
    var_77 = {}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'non_existent'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = [var_1, var_29]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = [var_1, var_4, var_6]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 'invalid_key'
    var_37 = [var_1, var_4, var_20, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = 'x'
    var_40 = [var_39]
    var_41 = 'missing'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'missing'
    var_43 = 'z'
    var_44 = [var_1, var_43]
    var_45 = 42
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 42
    var_47 = 99
    var_48 = [var_1, var_4, var_47]
    var_49 = 'out of bounds'
    var_50 = module_0.get_in(var_48, var_17, var_49)
    assert var_50 == 'out of bounds'
    var_51 = 'non_existent'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'b'
    var_56 = 'non_existent'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'f'
    var_61 = 5
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = 'a'
    var_66 = 0
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_17, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_17)
    var_72 = [var_65]
    var_73 = {}
    var_74 = 'empty'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'empty'
    var_76 = 'a'
    var_77 = [var_76]
    var_78 = {}
    var_79 = True
    var_80 = module_0.get_in(var_77, var_78, no_default=var_79)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = [var_1, var_29]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 5
    var_35 = [var_1, var_4, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 'nonexistent'
    var_38 = [var_1, var_4, var_14, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_29]
    var_41 = 'missing'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'missing'
    var_43 = [var_1, var_29]
    var_44 = module_0.get_in(var_43, var_17, var_20)
    assert var_44 == 0
    var_45 = 'z'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'b'
    var_50 = 'z'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'f'
    var_55 = 10
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'a'
    var_60 = 'not_a_container'
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
    var_66 = [var_59]
    var_67 = {}
    var_68 = 'fallback'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'fallback'
    var_70 = 'a'
    var_71 = [var_70]
    var_72 = {}
    var_73 = True
    var_74 = module_0.get_in(var_71, var_72, no_default=var_73)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = [var_1, var_29]
    var_33 = 'missing'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing'
    var_35 = 5
    var_36 = [var_1, var_4, var_35]
    var_37 = module_0.get_in(var_36, var_17, var_33)
    assert var_37 == 'missing'
    var_38 = 'not_an_int'
    var_39 = [var_1, var_4, var_38]
    var_40 = module_0.get_in(var_39, var_17, var_33)
    assert var_40 == 'missing'
    var_41 = 'z'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_17, no_default=var_43)
    var_45 = 'b'
    var_46 = 'z'
    var_47 = [var_45, var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'c'
    var_52 = 99
    var_53 = [var_50, var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'a'
    var_57 = 0
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = [var_56]
    var_64 = {}
    var_65 = 'empty'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'empty'
    var_67 = 'a'
    var_68 = [var_67]
    var_69 = {}
    var_70 = True
    var_71 = module_0.get_in(var_68, var_69, no_default=var_70)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 2
    var_7 = 10
    var_8 = 20
    var_9 = 'e'
    var_10 = 30
    var_11 = {var_9: var_10}
    var_12 = [var_7, var_8, var_11]
    var_13 = {var_4: var_6, var_5: var_12}
    var_14 = None
    var_15 = False
    var_16 = ''
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_0: var_3, var_1: var_13, var_2: var_17}
    var_19 = [var_0]
    var_20 = module_0.get_in(var_19, var_18)
    assert var_20 == 1
    var_21 = [var_1, var_4]
    var_22 = module_0.get_in(var_21, var_18)
    assert var_22 == 2
    var_23 = [var_1, var_5, var_15]
    var_24 = module_0.get_in(var_23, var_18)
    assert var_24 == 10
    var_25 = [var_1, var_5, var_6, var_9]
    var_26 = module_0.get_in(var_25, var_18)
    assert var_26 == 30
    var_27 = [var_2, var_15]
    var_28 = module_0.get_in(var_27, var_18)
    assert var_28 is None
    var_29 = [var_2, var_3]
    var_30 = module_0.get_in(var_29, var_18)
    assert var_30 is False
    var_31 = [var_2, var_6]
    var_32 = module_0.get_in(var_31, var_18)
    assert var_32 == ''
    var_33 = 'non_existent'
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_18)
    assert var_35 is None
    var_36 = [var_1, var_33]
    var_37 = module_0.get_in(var_36, var_18)
    assert var_37 is None
    var_38 = 99
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_18)
    assert var_40 is None
    var_41 = 'wrong_key'
    var_42 = [var_1, var_5, var_6, var_41]
    var_43 = module_0.get_in(var_42, var_18)
    assert var_43 is None
    var_44 = 'x'
    var_45 = [var_44]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_45, var_18, var_46)
    assert var_47 == 'missing'
    var_48 = 'z'
    var_49 = [var_1, var_48]
    var_50 = module_0.get_in(var_49, var_18, var_15)
    assert var_50 == 0
    var_51 = 'x'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_18, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_18, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 10
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_18, no_default=var_64)
    var_66 = 'a'
    var_67 = 'not_an_index'
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_18, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_18)
    var_73 = [var_66]
    var_74 = {}
    var_75 = 'fallback'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'fallback'
    var_77 = 'a'
    var_78 = [var_77]
    var_79 = {}
    var_80 = True
    var_81 = module_0.get_in(var_78, var_79, no_default=var_80)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'nested'
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = 'fallback'
    var_37 = module_0.get_in(var_35, var_21, var_36)
    assert var_37 == 'fallback'
    var_38 = [var_1, var_5, var_8]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = [var_1, var_5, var_8]
    var_41 = module_0.get_in(var_40, var_21, var_16)
    assert var_41 == 0
    var_42 = 'not_a_subdict'
    var_43 = [var_0, var_42]
    var_44 = module_0.get_in(var_43, var_21)
    assert var_44 is None
    var_45 = 'b'
    var_46 = 'missing'
    var_47 = [var_45, var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_21, no_default=var_48)
    var_50 = 'b'
    var_51 = 'd'
    var_52 = 10
    var_53 = [var_50, var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_21, no_default=var_54)
    var_56 = 'a'
    var_57 = 'not_a_subdict'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_21, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_21)
    var_63 = [var_56]
    var_64 = {}
    var_65 = 'empty'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'empty'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'nested'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'missing'
    var_36 = [var_1, var_35]
    var_37 = 'fallback'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'fallback'
    var_39 = 99
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = [var_1, var_5, var_39]
    var_43 = module_0.get_in(var_42, var_21, var_37)
    assert var_43 == 'fallback'
    var_44 = 'not_subscriptable'
    var_45 = [var_0, var_44]
    var_46 = module_0.get_in(var_45, var_21)
    assert var_46 is None
    var_47 = [var_0, var_44]
    var_48 = 'error'
    var_49 = module_0.get_in(var_47, var_21, var_48)
    assert var_49 == 'error'
    var_50 = 'z'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_21, no_default=var_52)
    var_54 = 'b'
    var_55 = 'z'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_21, no_default=var_57)
    var_59 = 'b'
    var_60 = 'd'
    var_61 = 99
    var_62 = [var_59, var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_21, no_default=var_63)
    var_65 = 'a'
    var_66 = 'not_subscriptable'
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_21, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_21)
    var_72 = 'any'
    var_73 = [var_72]
    var_74 = {}
    var_75 = 'empty'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'empty'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'h'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = [var_3, var_18]
    var_20 = {var_0: var_3, var_1: var_15, var_2: var_19}
    var_21 = [var_0]
    var_22 = module_0.get_in(var_21, var_20)
    assert var_22 == 1
    var_23 = [var_1, var_4]
    var_24 = module_0.get_in(var_23, var_20)
    assert var_24 == 2
    var_25 = [var_1, var_5, var_3]
    var_26 = module_0.get_in(var_25, var_20)
    assert var_26 == 20
    var_27 = [var_1, var_6, var_12]
    var_28 = module_0.get_in(var_27, var_20)
    assert var_28 == 'hello'
    var_29 = [var_2, var_3, var_16]
    var_30 = module_0.get_in(var_29, var_20)
    assert var_30 == 5
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_20)
    assert var_33 is None
    var_34 = [var_1, var_31]
    var_35 = module_0.get_in(var_34, var_20)
    assert var_35 is None
    var_36 = [var_1, var_5, var_17]
    var_37 = module_0.get_in(var_36, var_20)
    assert var_37 is None
    var_38 = [var_1, var_6, var_12, var_2]
    var_39 = module_0.get_in(var_38, var_20)
    assert var_39 is None
    var_40 = 'non_existent'
    var_41 = [var_40]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_20, var_42)
    assert var_43 == 'missing'
    var_44 = [var_1, var_31]
    var_45 = 0
    var_46 = module_0.get_in(var_44, var_20, var_45)
    assert var_46 == 0
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_20, no_default=var_49)
    var_51 = 'b'
    var_52 = 'z'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_20, no_default=var_54)
    var_56 = 'b'
    var_57 = 'd'
    var_58 = 10
    var_59 = [var_56, var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_20, no_default=var_60)
    var_62 = 'a'
    var_63 = 0
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_20, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_20)
    var_69 = [var_62]
    var_70 = {}
    var_71 = 'fallback'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'fallback'
    var_73 = 'a'
    var_74 = [var_73]
    var_75 = {}
    var_76 = True
    var_77 = module_0.get_in(var_74, var_75, no_default=var_76)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = 'z'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'x'
    var_31 = [var_1, var_30]
    var_32 = 'missing'
    var_33 = module_0.get_in(var_31, var_17, var_32)
    assert var_33 == 'missing'
    var_34 = 5
    var_35 = [var_1, var_4, var_34]
    var_36 = module_0.get_in(var_35, var_17, var_20)
    assert var_36 == 0
    var_37 = 'not_an_int'
    var_38 = [var_1, var_4, var_37]
    var_39 = 'error'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'error'
    var_41 = 'z'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_17, no_default=var_43)
    var_45 = 'b'
    var_46 = 'x'
    var_47 = [var_45, var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'f'
    var_51 = 10
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'a'
    var_56 = 0
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = []
    var_61 = module_0.get_in(var_60, var_17)
    var_62 = [var_56, var_5]
    var_63 = module_0.get_in(var_62, var_17)
    assert var_63 is None



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = 'z'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'missing'
    var_31 = [var_1, var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = [var_1, var_4, var_6]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'nonexistent'
    var_36 = [var_1, var_4, var_20, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_27]
    var_39 = module_0.get_in(var_38, var_17, var_30)
    assert var_39 == 'missing'
    var_40 = [var_1, var_30]
    var_41 = module_0.get_in(var_40, var_17, var_20)
    assert var_41 == 0
    var_42 = 'z'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_17, no_default=var_44)
    var_46 = 'b'
    var_47 = 'missing'
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'f'
    var_52 = 10
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'a'
    var_57 = 0
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = [var_57, var_5]
    var_64 = module_0.get_in(var_63, var_17)
    assert var_64 is None



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'nested'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'nonexistent'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = [var_1, var_5, var_8]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = 'too_deep'
    var_41 = [var_1, var_4, var_40]
    var_42 = module_0.get_in(var_41, var_21)
    assert var_42 is None
    var_43 = [var_32]
    var_44 = 'missing'
    var_45 = module_0.get_in(var_43, var_21, var_44)
    assert var_45 == 'missing'
    var_46 = [var_1, var_35]
    var_47 = module_0.get_in(var_46, var_21, var_16)
    assert var_47 == 0
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_21, no_default=var_50)
    var_52 = 'b'
    var_53 = 'nonexistent'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'd'
    var_59 = 10
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_21, no_default=var_61)
    var_63 = 'a'
    var_64 = 'not_possible'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_21, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_21)
    var_70 = [var_63]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'
    var_74 = 'a'
    var_75 = [var_74]
    var_76 = {}
    var_77 = True
    var_78 = module_0.get_in(var_75, var_76, no_default=var_77)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = True
    var_15 = False
    var_16 = [var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_15]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 10
    var_22 = 2
    var_23 = [var_1, var_4, var_22, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_15]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 is True
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = 'missing_val'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing_val'
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_1, var_4, var_36]
    var_40 = 'not_found'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'not_found'
    var_42 = 'not_an_index'
    var_43 = [var_0, var_42]
    var_44 = module_0.get_in(var_43, var_17)
    assert var_44 is None
    var_45 = [var_0, var_42]
    var_46 = 'error'
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 'error'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'missing'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'b'
    var_58 = 'c'
    var_59 = 5
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = 'a'
    var_64 = 'not_an_index'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_17, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_17)
    var_70 = [var_63]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'
    var_74 = 'a'
    var_75 = [var_74]
    var_76 = {}
    var_77 = True
    var_78 = module_0.get_in(var_75, var_76, no_default=var_77)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'h'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = [var_3, var_18]
    var_20 = {var_0: var_3, var_1: var_15, var_2: var_19}
    var_21 = [var_0]
    var_22 = module_0.get_in(var_21, var_20)
    assert var_22 == 1
    var_23 = [var_1, var_4]
    var_24 = module_0.get_in(var_23, var_20)
    assert var_24 == 2
    var_25 = [var_1, var_5, var_3]
    var_26 = module_0.get_in(var_25, var_20)
    assert var_26 == 20
    var_27 = [var_1, var_6, var_12]
    var_28 = module_0.get_in(var_27, var_20)
    assert var_28 == 'hello'
    var_29 = [var_2, var_3, var_16]
    var_30 = module_0.get_in(var_29, var_20)
    assert var_30 == 5
    var_31 = 'non'
    var_32 = 'existent'
    var_33 = [var_31, var_32]
    var_34 = module_0.get_in(var_33, var_20)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_20)
    assert var_37 is None
    var_38 = [var_2, var_17]
    var_39 = module_0.get_in(var_38, var_20)
    assert var_39 is None
    var_40 = [var_1, var_5, var_8]
    var_41 = module_0.get_in(var_40, var_20)
    assert var_41 is None
    var_42 = [var_31, var_32]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_20, var_43)
    assert var_44 == 'missing'
    var_45 = [var_1, var_35]
    var_46 = 0
    var_47 = module_0.get_in(var_45, var_20, var_46)
    assert var_47 == 0
    var_48 = 'y'
    var_49 = [var_48]
    var_50 = {}
    var_51 = True
    var_52 = module_0.get_in(var_49, var_50, no_default=var_51)
    var_53 = 'b'
    var_54 = 'z'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_20, no_default=var_56)
    var_58 = 'b'
    var_59 = 'd'
    var_60 = 99
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_20, no_default=var_62)
    var_64 = 'a'
    var_65 = 0
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_20, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_20)
    var_71 = 3
    var_72 = [var_71]
    var_73 = [var_7, var_72]
    var_74 = [var_67, var_73]
    var_75 = '1'
    var_76 = [var_75, var_46]
    var_77 = [var_8, var_9]
    var_78 = {var_75: var_77}
    var_79 = module_0.get_in(var_76, var_78)
    assert var_79 == 10
    var_80 = [var_75, var_46]
    var_81 = [var_8, var_9]
    var_82 = {var_75: var_81}
    var_83 = module_0.get_in(var_80, var_82)
    assert var_83 == 10
    var_84 = [var_67, var_46, var_67]
    var_85 = module_0.get_in(var_84, var_74)
    assert var_85 == 2



# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = 5
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 5
    var_32 = 0
    var_33 = [var_2, var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = [var_1, var_35]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = [var_1, var_5, var_18]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = [var_1, var_6, var_12, var_2]
    var_43 = module_0.get_in(var_42, var_21)
    assert var_43 is None
    var_44 = 'x'
    var_45 = [var_44]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_45, var_21, var_46)
    assert var_47 == 'missing'
    var_48 = [var_1, var_35]
    var_49 = 404
    var_50 = module_0.get_in(var_48, var_21, var_49)
    assert var_50 == 404
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_21, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_21, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 10
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_21, no_default=var_64)
    var_66 = 'a'
    var_67 = 'not_an_index'
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_21, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_21)
    var_73 = [var_66]
    var_74 = {}
    var_75 = 'empty'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'empty'
    var_77 = 'a'
    var_78 = [var_77]
    var_79 = {}
    var_80 = True
    var_81 = module_0.get_in(var_78, var_79, no_default=var_80)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'nested'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_21, var_36)
    assert var_37 == 'missing'
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = [var_1, var_5, var_38]
    var_42 = module_0.get_in(var_41, var_21, var_36)
    assert var_42 == 'missing'
    var_43 = 'not_a_key'
    var_44 = [var_0, var_43]
    var_45 = module_0.get_in(var_44, var_21)
    assert var_45 is None
    var_46 = [var_0, var_43]
    var_47 = module_0.get_in(var_46, var_21, var_36)
    assert var_47 == 'missing'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_21, no_default=var_50)
    var_52 = 'b'
    var_53 = 'z'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'd'
    var_59 = 5
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_21, no_default=var_61)
    var_63 = 'a'
    var_64 = 'not_a_key'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_21, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_21)
    var_70 = [var_63]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'
    var_74 = 'a'
    var_75 = [var_74]
    var_76 = {}
    var_77 = True
    var_78 = module_0.get_in(var_75, var_76, no_default=var_77)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = 5
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 5
    var_32 = 'z'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_5, var_8]
    var_36 = module_0.get_in(var_35, var_21)
    assert var_36 is None
    var_37 = 'y'
    var_38 = [var_32, var_37]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = 'nonexistent'
    var_41 = 'nested'
    var_42 = [var_1, var_40, var_41]
    var_43 = module_0.get_in(var_42, var_21)
    assert var_43 is None
    var_44 = [var_1, var_32]
    var_45 = 'missing'
    var_46 = module_0.get_in(var_44, var_21, var_45)
    assert var_46 == 'missing'
    var_47 = [var_1, var_5, var_8]
    var_48 = 0
    var_49 = module_0.get_in(var_47, var_21, var_48)
    assert var_49 == 0
    var_50 = 'b'
    var_51 = 'z'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_21, no_default=var_53)
    var_55 = 'b'
    var_56 = 'd'
    var_57 = 10
    var_58 = [var_55, var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_21, no_default=var_59)
    var_61 = 'a'
    var_62 = 'not_an_index'
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_21, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_21)
    var_68 = [var_61]
    var_69 = {}
    var_70 = 'empty'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'empty'
    var_72 = 'a'
    var_73 = [var_72]
    var_74 = {}
    var_75 = True
    var_76 = module_0.get_in(var_73, var_74, no_default=var_75)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'x'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_1, var_35]
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'missing'
    var_39 = 5
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_21, var_16)
    assert var_41 == 0
    var_42 = [var_1, var_6, var_35]
    var_43 = 'N/A'
    var_44 = module_0.get_in(var_42, var_21, var_43)
    assert var_44 == 'N/A'
    var_45 = 'x'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_21, no_default=var_47)
    var_49 = 'b'
    var_50 = 'z'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_21, no_default=var_52)
    var_54 = 'b'
    var_55 = 'd'
    var_56 = 5
    var_57 = [var_54, var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_21, no_default=var_58)
    var_60 = 'a'
    var_61 = 'not_an_index'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_21, no_default=var_63)
    var_65 = [var_60]
    var_66 = {}
    var_67 = 'empty'
    var_68 = module_0.get_in(var_65, var_66, var_67)
    assert var_68 == 'empty'
    var_69 = 'a'
    var_70 = [var_69]
    var_71 = {}
    var_72 = True
    var_73 = module_0.get_in(var_70, var_71, no_default=var_72)
    var_74 = []
    var_75 = module_0.get_in(var_74, var_21)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_18]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_18, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 is True
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'missing'
    var_36 = [var_1, var_35]
    var_37 = 'missing_val'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'missing_val'
    var_39 = 5
    var_40 = [var_1, var_5, var_39]
    var_41 = 0
    var_42 = module_0.get_in(var_40, var_21, var_41)
    assert var_42 == 0
    var_43 = 'too_deep'
    var_44 = [var_1, var_4, var_43]
    var_45 = 'default'
    var_46 = module_0.get_in(var_44, var_21, var_45)
    assert var_46 == 'default'
    var_47 = 'not_a_container'
    var_48 = [var_0, var_47]
    var_49 = module_0.get_in(var_48, var_21)
    assert var_49 is None
    var_50 = [var_0, var_41]
    var_51 = 'error'
    var_52 = module_0.get_in(var_50, var_21, var_51)
    assert var_52 == 'error'
    var_53 = 'z'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'non_existent'
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_21, no_default=var_60)
    var_62 = 'b'
    var_63 = 'd'
    var_64 = 99
    var_65 = [var_62, var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_21, no_default=var_66)
    var_68 = 'a'
    var_69 = 0
    var_70 = [var_68, var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_21, no_default=var_71)
    var_73 = []
    var_74 = module_0.get_in(var_73, var_21)
    var_75 = 'any'
    var_76 = [var_75]
    var_77 = {}
    var_78 = 'empty'
    var_79 = module_0.get_in(var_76, var_77, var_78)
    assert var_79 == 'empty'
    var_80 = 'any'
    var_81 = [var_80]
    var_82 = {}
    var_83 = True
    var_84 = module_0.get_in(var_81, var_82, no_default=var_83)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = module_0.get_in(var_35, var_21)
    assert var_36 is None
    var_37 = [var_1, var_5, var_8]
    var_38 = module_0.get_in(var_37, var_21)
    assert var_38 is None
    var_39 = 5
    var_40 = [var_2, var_39]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = [var_32]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_21, var_43)
    assert var_44 == 'missing'
    var_45 = [var_1, var_32]
    var_46 = 0
    var_47 = module_0.get_in(var_45, var_21, var_46)
    assert var_47 == 0
    var_48 = [var_1, var_5, var_8]
    var_49 = 'not found'
    var_50 = module_0.get_in(var_48, var_21, var_49)
    assert var_50 == 'not and found'
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_21, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_21, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 10
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_21, no_default=var_64)
    var_66 = 'a'
    var_67 = 0
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_21, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_21)
    var_73 = {var_66: var_16}
    var_74 = [var_66]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 is None
    var_76 = [var_32]
    var_77 = 'fallback'
    var_78 = module_0.get_in(var_76, var_73, var_77)
    assert var_78 == 'fallback'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = 'missing_val'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing_val'
    var_36 = [var_2, var_6]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_2, var_6]
    var_39 = 'out_of_bounds'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'out_of_bounds'
    var_41 = 'sub_key'
    var_42 = [var_0, var_41]
    var_43 = module_0.get_in(var_42, var_17)
    assert var_43 is None
    var_44 = [var_0, var_41]
    var_45 = 'not_a_dict'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'not_a_dict'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'f'
    var_52 = 10
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'a'
    var_57 = 'sub_key'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = [var_56]
    var_64 = {}
    var_65 = 'empty'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'empty'
    var_67 = 'a'
    var_68 = [var_67]
    var_69 = {}
    var_70 = True
    var_71 = module_0.get_in(var_68, var_69, no_default=var_70)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = 'missing_val'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing_val'
    var_36 = [var_2, var_6]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_2, var_6]
    var_39 = 'out_of_bounds'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'out_of_bounds'
    var_41 = [var_0, var_20]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_20]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'z'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'f'
    var_56 = 10
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'a'
    var_61 = 0
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = [var_60]
    var_68 = {}
    var_69 = 'empty'
    var_70 = module_0.get_in(var_67, var_68, var_69)
    assert var_70 == 'empty'
    var_71 = 'a'
    var_72 = [var_71]
    var_73 = {}
    var_74 = True
    var_75 = module_0.get_in(var_72, var_73, no_default=var_74)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'missing'
    var_36 = [var_1, var_35]
    var_37 = 'missing_val'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'missing_val'
    var_39 = [var_1, var_5, var_8]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = [var_1, var_5, var_8]
    var_42 = 'out_of_bounds'
    var_43 = module_0.get_in(var_41, var_21, var_42)
    assert var_43 == 'out_of_bounds'
    var_44 = 'not_a_container'
    var_45 = [var_0, var_44]
    var_46 = module_0.get_in(var_45, var_21)
    assert var_46 is None
    var_47 = [var_0, var_44]
    var_48 = 'error'
    var_49 = module_0.get_in(var_47, var_21, var_48)
    assert var_49 == 'error'
    var_50 = 'z'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_21, no_default=var_52)
    var_54 = 'b'
    var_55 = 'nonexistent'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_21, no_default=var_57)
    var_59 = 'b'
    var_60 = 'd'
    var_61 = 99
    var_62 = [var_59, var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_21, no_default=var_63)
    var_65 = 'a'
    var_66 = 0
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_21, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_21)
    var_72 = 'any'
    var_73 = [var_72]
    var_74 = {}
    var_75 = 'empty'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'empty'
    var_77 = 'any'
    var_78 = [var_77]
    var_79 = {}
    var_80 = True
    var_81 = module_0.get_in(var_78, var_79, no_default=var_80)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_21, var_36)
    assert var_37 == 'missing'
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = [var_1, var_5, var_38]
    var_42 = module_0.get_in(var_41, var_21, var_36)
    assert var_42 == 'missing'
    var_43 = 'not_a_subdict'
    var_44 = [var_0, var_43]
    var_45 = module_0.get_in(var_44, var_21)
    assert var_45 is None
    var_46 = [var_0, var_43]
    var_47 = module_0.get_in(var_46, var_21, var_36)
    assert var_47 == 'missing'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_21, no_default=var_50)
    var_52 = 'b'
    var_53 = 'z'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'd'
    var_59 = 5
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_21, no_default=var_61)
    var_63 = 'a'
    var_64 = 'not_a_subdict'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_21, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_21)
    var_70 = [var_63]
    var_71 = {}
    var_72 = 'none'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'none'
    var_74 = 'a'
    var_75 = [var_74]
    var_76 = {}
    var_77 = True
    var_78 = module_0.get_in(var_75, var_76, no_default=var_77)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_14, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = 'z'
    var_27 = [var_26]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'missing'
    var_30 = [var_1, var_29]
    var_31 = 'missing_val'
    var_32 = module_0.get_in(var_30, var_17, var_31)
    assert var_32 == 'missing_val'
    var_33 = [var_2, var_6]
    var_34 = 0
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 0
    var_36 = 'not_a_list'
    var_37 = [var_0, var_36]
    var_38 = 'error'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'error'
    var_40 = 'z'
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_17, no_default=var_42)
    var_44 = 'f'
    var_45 = 10
    var_46 = [var_44, var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'a'
    var_50 = 0
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = []
    var_55 = module_0.get_in(var_54, var_17)
    var_56 = [var_49]
    var_57 = {}
    var_58 = 'empty'
    var_59 = module_0.get_in(var_56, var_57, var_58)
    assert var_59 == 'empty'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_18]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_18, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 is True
    var_32 = 0
    var_33 = [var_2, var_32]
    var_34 = module_0.get_in(var_33)
    assert var_34 is None
    var_35 = 'non'
    var_36 = 'existent'
    var_37 = [var_35, var_36]
    var_38 = module_0.get_in(var_37, var_21)
    assert var_38 is None
    var_39 = 'z'
    var_40 = [var_1, var_39]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = 99
    var_43 = [var_1, var_5, var_42]
    var_44 = module_0.get_in(var_43, var_21)
    assert var_44 is None
    var_45 = 'invalid_type'
    var_46 = [var_45]
    var_47 = 123
    var_48 = module_0.get_in(var_46, var_47)
    assert var_48 is None
    var_49 = 'missing'
    var_50 = [var_1, var_49]
    var_51 = 'missing_val'
    var_52 = module_0.get_in(var_50, var_21, var_51)
    assert var_52 == 'missing_val'
    var_53 = 'x'
    var_54 = 'y'
    var_55 = [var_53, var_54]
    var_56 = module_0.get_in(var_55, var_21, var_32)
    assert var_56 == 0
    var_57 = 'non'
    var_58 = 'existent'
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_21, no_default=var_60)
    var_62 = 'b'
    var_63 = 'z'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_21, no_default=var_65)
    var_67 = 'b'
    var_68 = 'd'
    var_69 = 99
    var_70 = [var_67, var_68, var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_21, no_default=var_71)
    var_73 = 'a'
    var_74 = 'not_an_index'
    var_75 = [var_73, var_74]
    var_76 = True
    var_77 = module_0.get_in(var_75, var_21, no_default=var_76)
    var_78 = []
    var_79 = module_0.get_in(var_78, var_21)
    var_80 = [var_73]
    var_81 = {}
    var_82 = 'empty'
    var_83 = module_0.get_in(var_80, var_81, var_82)
    assert var_83 == 'empty'
    var_84 = 'a'
    var_85 = [var_84]
    var_86 = {}
    var_87 = True
    var_88 = module_0.get_in(var_85, var_86, no_default=var_87)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = True
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_18]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_18, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 is True
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = module_0.get_in(var_35, var_21)
    assert var_36 is None
    var_37 = 5
    var_38 = [var_1, var_5, var_37]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = [var_2, var_37]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = [var_32]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_21, var_43)
    assert var_44 == 'missing'
    var_45 = [var_1, var_32]
    var_46 = 0
    var_47 = module_0.get_in(var_45, var_21, var_46)
    assert var_47 == 0
    var_48 = [var_1, var_5, var_37]
    var_49 = 'error'
    var_50 = module_0.get_in(var_48, var_21, var_49)
    assert var_50 == 'error'
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_21, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_21, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 10
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_21, no_default=var_64)
    var_66 = 'a'
    var_67 = 0
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_21, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_21)
    var_73 = [var_66]
    var_74 = {}
    var_75 = module_0.get_in(var_73, var_74, var_43)
    assert var_75 == 'missing'
    var_76 = 'a'
    var_77 = [var_76]
    var_78 = {}
    var_79 = True
    var_80 = module_0.get_in(var_77, var_78, no_default=var_79)



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = 0
    var_21 = [var_1, var_4, var_20]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 10
    var_23 = [var_1, var_4, var_14, var_8]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 'found'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 5
    var_36 = [var_1, var_4, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = 'invalid_index'
    var_39 = [var_1, var_4, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = 'non'
    var_42 = 'existent'
    var_43 = 'path'
    var_44 = [var_41, var_42, var_43]
    var_45 = module_0.get_in(var_44, var_17)
    assert var_45 is None
    var_46 = [var_29]
    var_47 = module_0.get_in(var_46, var_17, var_32)
    assert var_47 == 'missing'
    var_48 = [var_1, var_32]
    var_49 = 42
    var_50 = module_0.get_in(var_48, var_17, var_49)
    assert var_50 == 42
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'f'
    var_61 = 10
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = 'a'
    var_66 = 0
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_17, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_17)
    var_72 = [var_65]
    var_73 = {}
    var_74 = 'fallback'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'fallback'
    var_76 = 'a'
    var_77 = [var_76]
    var_78 = {}
    var_79 = True
    var_80 = module_0.get_in(var_77, var_78, no_default=var_79)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'h'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = [var_3, var_18]
    var_20 = {var_0: var_3, var_1: var_15, var_2: var_19}
    var_21 = [var_0]
    var_22 = module_0.get_in(var_21, var_20)
    assert var_22 == 1
    var_23 = [var_1, var_4]
    var_24 = module_0.get_in(var_23, var_20)
    assert var_24 == 2
    var_25 = [var_1, var_5, var_3]
    var_26 = module_0.get_in(var_25, var_20)
    assert var_26 == 20
    var_27 = [var_1, var_6, var_12]
    var_28 = module_0.get_in(var_27, var_20)
    assert var_28 == 'hello'
    var_29 = [var_2, var_3, var_16]
    var_30 = module_0.get_in(var_29, var_20)
    assert var_30 == 5
    var_31 = 'x'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_20)
    assert var_33 is None
    var_34 = 'z'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_20)
    assert var_36 is None
    var_37 = [var_1, var_5, var_17]
    var_38 = module_0.get_in(var_37, var_20)
    assert var_38 is None
    var_39 = [var_1, var_6, var_12, var_2]
    var_40 = module_0.get_in(var_39, var_20)
    assert var_40 is None
    var_41 = [var_31]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_20, var_42)
    assert var_43 == 'missing'
    var_44 = [var_1, var_34]
    var_45 = 0
    var_46 = module_0.get_in(var_44, var_20, var_45)
    assert var_46 == 0
    var_47 = [var_1, var_5, var_17]
    var_48 = 'error'
    var_49 = module_0.get_in(var_47, var_20, var_48)
    assert var_49 == 'error'
    var_50 = 'x'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_20, no_default=var_52)
    var_54 = 'b'
    var_55 = 'z'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_20, no_default=var_57)
    var_59 = 'b'
    var_60 = 'd'
    var_61 = 5
    var_62 = [var_59, var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_20, no_default=var_63)
    var_65 = 'a'
    var_66 = 'not_an_index'
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_20, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_20)
    var_72 = [var_65]
    var_73 = {}
    var_74 = 'empty'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'empty'
    var_76 = 'a'
    var_77 = [var_76]
    var_78 = {}
    var_79 = True
    var_80 = module_0.get_in(var_77, var_78, no_default=var_79)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'missing'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = 'non_existent'
    var_42 = 'key'
    var_43 = [var_1, var_41, var_42]
    var_44 = module_0.get_in(var_43, var_21)
    assert var_44 is None
    var_45 = 'x'
    var_46 = [var_45]
    var_47 = module_0.get_in(var_46, var_21, var_35)
    assert var_47 == 'missing'
    var_48 = [var_1, var_32]
    var_49 = 42
    var_50 = module_0.get_in(var_48, var_21, var_49)
    assert var_50 == 42
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_21, no_default=var_53)
    var_55 = 'b'
    var_56 = 'z'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_21, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 10
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_21, no_default=var_64)
    var_66 = 'a'
    var_67 = 'sub_key'
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_21, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_21)
    var_73 = [var_66]
    var_74 = {}
    var_75 = 'empty'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'empty'
    var_77 = 'a'
    var_78 = [var_77]
    var_79 = {}
    var_80 = True
    var_81 = module_0.get_in(var_78, var_79, no_default=var_80)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 'h'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = [var_3, var_18]
    var_20 = {var_0: var_3, var_1: var_15, var_2: var_19}
    var_21 = [var_0]
    var_22 = module_0.get_in(var_21, var_20)
    assert var_22 == 1
    var_23 = [var_1, var_4]
    var_24 = module_0.get_in(var_23, var_20)
    assert var_24 == 2
    var_25 = [var_1, var_5, var_3]
    var_26 = module_0.get_in(var_25, var_20)
    assert var_26 == 20
    var_27 = [var_1, var_6, var_12]
    var_28 = module_0.get_in(var_27, var_20)
    assert var_28 == 'found'
    var_29 = [var_2, var_3, var_16]
    var_30 = module_0.get_in(var_29, var_20)
    assert var_30 == 5
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_20)
    assert var_33 is None
    var_34 = 'nonexistent'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_20)
    assert var_36 is None
    var_37 = [var_1, var_5, var_8]
    var_38 = module_0.get_in(var_37, var_20)
    assert var_38 is None
    var_39 = 'string_index'
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_20)
    assert var_41 is None
    var_42 = [var_31]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_20, var_43)
    assert var_44 == 'missing'
    var_45 = [var_1, var_34]
    var_46 = 0
    var_47 = module_0.get_in(var_45, var_20, var_46)
    assert var_47 == 0
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_20, no_default=var_50)
    var_52 = 'b'
    var_53 = 'nonexistent'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_20, no_default=var_55)
    var_57 = 'b'
    var_58 = 'd'
    var_59 = 10
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_20, no_default=var_61)
    var_63 = 'a'
    var_64 = 'not_an_index'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_20, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_20)
    var_70 = [var_63]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'
    var_74 = 'a'
    var_75 = [var_74]
    var_76 = {}
    var_77 = True
    var_78 = module_0.get_in(var_75, var_76, no_default=var_77)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'e'
    var_6 = 10
    var_7 = 20
    var_8 = 'd'
    var_9 = 'found'
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = None
    var_13 = {var_4: var_11, var_5: var_12}
    var_14 = 2
    var_15 = 3
    var_16 = [var_3, var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_3]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 2
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = [var_1, var_31]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = 'nonexistent'
    var_40 = [var_1, var_4, var_22, var_39]
    var_41 = module_0.get_in(var_40, var_17)
    assert var_41 is None
    var_42 = 'not_a_dict'
    var_43 = [var_42]
    var_44 = 'missing'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'missing'
    var_46 = [var_1, var_4, var_36]
    var_47 = module_0.get_in(var_46, var_17, var_44)
    assert var_47 == 'missing'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'z'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'f'
    var_58 = 10
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = 'a'
    var_63 = 0
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_17, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_17)
    var_69 = [var_62]
    var_70 = {}
    var_71 = 'empty'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'empty'
    var_73 = 'a'
    var_74 = [var_73]
    var_75 = {}
    var_76 = True
    var_77 = module_0.get_in(var_74, var_75, no_default=var_76)



# Parsed testcases at query #17
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = [var_1, var_32]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_21, var_36)
    assert var_37 == 'missing'
    var_38 = 5
    var_39 = [var_1, var_5, var_38]
    var_40 = module_0.get_in(var_39, var_21)
    assert var_40 is None
    var_41 = [var_2, var_38]
    var_42 = module_0.get_in(var_41, var_21, var_36)
    assert var_42 == 'missing'
    var_43 = 'not_a_subdict'
    var_44 = [var_0, var_43]
    var_45 = module_0.get_in(var_44, var_21)
    assert var_45 is None
    var_46 = [var_0, var_43]
    var_47 = 'error'
    var_48 = module_0.get_in(var_46, var_21, var_47)
    assert var_48 == 'error'
    var_49 = 'z'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_21, no_default=var_51)
    var_53 = 'b'
    var_54 = 'z'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_21, no_default=var_56)
    var_58 = 'b'
    var_59 = 'd'
    var_60 = 5
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_21, no_default=var_62)
    var_64 = 'a'
    var_65 = 'not_a_subdict'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_21, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_21)
    var_71 = [var_64]
    var_72 = {}
    var_73 = 'empty'
    var_74 = module_0.get_in(var_71, var_72, var_73)
    assert var_74 == 'empty'
    var_75 = 'a'
    var_76 = [var_75]
    var_77 = {}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'found'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = None
    var_17 = 'h'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'found'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'nested'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'nonexistent'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_21)
    assert var_37 is None
    var_38 = [var_1, var_5, var_8]
    var_39 = module_0.get_in(var_38, var_21)
    assert var_39 is None
    var_40 = 'too_deep'
    var_41 = [var_1, var_4, var_40]
    var_42 = module_0.get_in(var_41, var_21)
    assert var_42 is None
    var_43 = [var_32]
    var_44 = 'missing'
    var_45 = module_0.get_in(var_43, var_21, var_44)
    assert var_45 == 'missing'
    var_46 = [var_1, var_35]
    var_47 = 0
    var_48 = module_0.get_in(var_46, var_21, var_47)
    assert var_48 == 0
    var_49 = 'z'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_21, no_default=var_51)
    var_53 = 'b'
    var_54 = 'nonexistent'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_21, no_default=var_56)
    var_58 = 'b'
    var_59 = 'd'
    var_60 = 10
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_21, no_default=var_62)
    var_64 = 'a'
    var_65 = 'sub_key'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_21, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_21)
    var_71 = [var_64]
    var_72 = {}
    var_73 = 'empty'
    var_74 = module_0.get_in(var_71, var_72, var_73)
    assert var_74 == 'empty'
    var_75 = 'a'
    var_76 = [var_75]
    var_77 = {}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'g'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 'e'
    var_7 = 2
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 'f'
    var_13 = 'hello'
    var_14 = {var_12: var_13}
    var_15 = {var_4: var_7, var_5: var_11, var_6: var_14}
    var_16 = 0
    var_17 = 'h'
    var_18 = 'world'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_0: var_3, var_1: var_15, var_2: var_20}
    var_22 = [var_0]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 1
    var_24 = [var_1, var_4]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 2
    var_26 = [var_1, var_5, var_3]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 20
    var_28 = [var_1, var_6, var_12]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 'hello'
    var_30 = [var_2, var_3, var_17]
    var_31 = module_0.get_in(var_30, var_21)
    assert var_31 == 'world'
    var_32 = 'z'
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_21)
    assert var_34 is None
    var_35 = 'missing'
    var_36 = [var_1, var_35]
    var_37 = 'N/A'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'N/A'
    var_39 = 99
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = 5
    var_43 = [var_2, var_42]
    var_44 = False
    var_45 = module_0.get_in(var_43, var_21, var_44)
    assert var_45 is False
    var_46 = 'not_a_container'
    var_47 = [var_0, var_46]
    var_48 = module_0.get_in(var_47, var_21)
    assert var_48 is None
    var_49 = [var_0, var_44]
    var_50 = 'error'
    var_51 = module_0.get_in(var_49, var_21, var_50)
    assert var_51 == 'error'
    var_52 = 'z'
    var_53 = [var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_21, no_default=var_54)
    var_56 = 'b'
    var_57 = 'missing'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_21, no_default=var_59)
    var_61 = 'b'
    var_62 = 'd'
    var_63 = 99
    var_64 = [var_61, var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_21, no_default=var_65)
    var_67 = 'a'
    var_68 = 0
    var_69 = [var_67, var_68]
    var_70 = True
    var_71 = module_0.get_in(var_69, var_21, no_default=var_70)
    var_72 = []
    var_73 = module_0.get_in(var_72, var_21)
    var_74 = [var_67]
    var_75 = {}
    var_76 = module_0.get_in(var_74, var_75, var_35)
    assert var_76 == 'missing'
    var_77 = 'a'
    var_78 = [var_77]
    var_79 = {}
    var_80 = True
    var_81 = module_0.get_in(var_78, var_79, no_default=var_80)



