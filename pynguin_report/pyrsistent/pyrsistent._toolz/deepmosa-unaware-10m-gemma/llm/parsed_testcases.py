####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_37 = module_0.get_in(var_36, var_17, var_20)
    assert var_37 == 0
    var_38 = 'too_deep'
    var_39 = [var_0, var_38]
    var_40 = 'error'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'error'
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
    var_57 = 'too_deep'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = 'any'
    var_64 = [var_63]
    var_65 = {}
    var_66 = 'empty'
    var_67 = module_0.get_in(var_64, var_65, var_66)
    assert var_67 == 'empty'
    var_68 = 'any'
    var_69 = [var_68]
    var_70 = {}
    var_71 = True
    var_72 = module_0.get_in(var_69, var_70, no_default=var_71)



# Parsed testcases at query #2
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
    var_15 = [var_3, var_14]
    var_16 = {var_0: var_3, var_1: var_13, var_2: var_15}
    var_17 = [var_0]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 == 1
    var_19 = 0
    var_20 = [var_1, var_4, var_19]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_14, var_8]
    var_23 = module_0.get_in(var_22, var_16)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_16)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_16)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_16)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = 'missing_val'
    var_34 = module_0.get_in(var_32, var_16, var_33)
    assert var_34 == 'missing_val'
    var_35 = 5
    var_36 = [var_2, var_35]
    var_37 = module_0.get_in(var_36, var_16)
    assert var_37 is None
    var_38 = [var_2, var_35]
    var_39 = 'not_found'
    var_40 = module_0.get_in(var_38, var_16, var_39)
    assert var_40 == 'not_found'
    var_41 = 'sub_key'
    var_42 = [var_0, var_41]
    var_43 = module_0.get_in(var_42, var_16)
    assert var_43 is None
    var_44 = [var_0, var_41]
    var_45 = 'error'
    var_46 = module_0.get_in(var_44, var_16, var_45)
    assert var_46 == 'error'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_16, no_default=var_49)
    var_51 = 'b'
    var_52 = 'missing'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_16, no_default=var_54)
    var_56 = 'f'
    var_57 = 5
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_16, no_default=var_59)
    var_61 = 'a'
    var_62 = 'sub_key'
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_16, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_16)
    var_68 = 'any'
    var_69 = [var_68]
    var_70 = {}
    var_71 = 'empty'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'empty'
    var_73 = 'any'
    var_74 = [var_73]
    var_75 = {}
    var_76 = True
    var_77 = module_0.get_in(var_74, var_75, no_default=var_76)



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
    var_29 = [var_26]
    var_30 = 'missing'
    var_31 = module_0.get_in(var_29, var_17, var_30)
    assert var_31 == 'missing'
    var_32 = [var_2, var_6]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = [var_2, var_6]
    var_35 = module_0.get_in(var_34, var_17, var_30)
    assert var_35 == 'missing'
    var_36 = 'not_an_index'
    var_37 = [var_0, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_0, var_36]
    var_40 = 'error'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'error'
    var_42 = 'z'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_17, no_default=var_44)
    var_46 = 'f'
    var_47 = 10
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'a'
    var_52 = 'not_an_index'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = []
    var_57 = module_0.get_in(var_56, var_17)
    var_58 = [var_52, var_5]
    var_59 = module_0.get_in(var_58, var_17)
    assert var_59 is None
    var_60 = [var_52, var_5]
    var_61 = 'wrong'
    var_62 = module_0.get_in(var_60, var_17, var_61)
    assert var_62 is None



# Parsed testcases at query #4
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
    var_33 = 'z'
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_18)
    assert var_35 is None
    var_36 = 'nonexistent'
    var_37 = [var_1, var_36]
    var_38 = module_0.get_in(var_37, var_18)
    assert var_38 is None
    var_39 = 99
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_18)
    assert var_41 is None
    var_42 = 'invalid_type'
    var_43 = [var_1, var_42, var_15]
    var_44 = module_0.get_in(var_43, var_18)
    assert var_44 is None
    var_45 = [var_33]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_45, var_18, var_46)
    assert var_47 == 'missing'
    var_48 = [var_1, var_36]
    var_49 = 42
    var_50 = module_0.get_in(var_48, var_18, var_49)
    assert var_50 == 42
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_18, no_default=var_53)
    var_55 = 'b'
    var_56 = 'nonexistent'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_18, no_default=var_58)
    var_60 = 'b'
    var_61 = 'd'
    var_62 = 99
    var_63 = [var_60, var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_18, no_default=var_64)
    var_66 = 'a'
    var_67 = 0
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_14]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_15]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'x'
    var_32 = [var_1, var_31]
    var_33 = 'missing'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing'
    var_35 = 5
    var_36 = [var_1, var_4, var_35]
    var_37 = 'out_of_bounds'
    var_38 = module_0.get_in(var_36, var_17, var_37)
    assert var_38 == 'out_of_bounds'
    var_39 = 'nonexistent'
    var_40 = [var_1, var_4, var_14, var_39]
    var_41 = module_0.get_in(var_40, var_17, var_14)
    assert var_41 == 0
    var_42 = 'not_indexable'
    var_43 = [var_0, var_42]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'x'
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
    var_67 = 'any'
    var_68 = [var_67]
    var_69 = {}
    var_70 = 'empty'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'empty'



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
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = [var_1, var_4, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 'wrong_key'
    var_38 = [var_1, var_4, var_22, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_29]
    var_41 = module_0.get_in(var_40, var_17, var_32)
    assert var_41 == 'missing'
    var_42 = 'x'
    var_43 = [var_1, var_42]
    var_44 = 42
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 42
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'missing'
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
    var_67 = [var_61, var_5]
    var_68 = module_0.get_in(var_67, var_17)
    assert var_68 is None
    var_69 = 'b'
    var_70 = 'e'
    var_71 = 'too_deep'
    var_72 = [var_69, var_70, var_71]
    var_73 = True
    var_74 = module_0.get_in(var_72, var_17, no_default=var_73)



# Parsed testcases at query #7
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
    var_33 = 'nonexistent'
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_18)
    assert var_35 is None
    var_36 = 'missing'
    var_37 = [var_1, var_36]
    var_38 = module_0.get_in(var_37, var_18)
    assert var_38 is None
    var_39 = [var_1, var_5, var_7]
    var_40 = module_0.get_in(var_39, var_18)
    assert var_40 is None
    var_41 = 'nested_error'
    var_42 = [var_1, var_4, var_41]
    var_43 = module_0.get_in(var_42, var_18)
    assert var_43 is None
    var_44 = [var_33]
    var_45 = module_0.get_in(var_44, var_18, var_36)
    assert var_45 == 'missing'
    var_46 = [var_1, var_36]
    var_47 = module_0.get_in(var_46, var_18, var_15)
    assert var_47 == 0
    var_48 = 'y'
    var_49 = [var_48]
    var_50 = {}
    var_51 = True
    var_52 = module_0.get_in(var_49, var_50, no_default=var_51)
    var_53 = 'b'
    var_54 = 'missing'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_18, no_default=var_56)
    var_58 = 'b'
    var_59 = 'd'
    var_60 = 99
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_18, no_default=var_62)
    var_64 = 'a'
    var_65 = 'too_deep'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_18, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_18)



# Parsed testcases at query #8
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
    var_15 = True
    var_16 = [var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 2
    var_22 = 0
    var_23 = [var_1, var_5, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_5, var_6, var_9]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 30
    var_27 = [var_2, var_15]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is True
    var_29 = 'nonexistent'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'missing'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 5
    var_36 = [var_1, var_5, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = 'wrong_key'
    var_39 = [var_1, var_5, var_22, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_29]
    var_42 = module_0.get_in(var_41, var_17, var_32)
    assert var_42 == 'missing'
    var_43 = [var_1, var_32]
    var_44 = module_0.get_in(var_43, var_17, var_22)
    assert var_44 == 0
    var_45 = [var_1, var_5, var_35]
    var_46 = 'out of bounds'
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 'out of bounds'
    var_48 = 'nonexistent'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'missing'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'b'
    var_58 = 'd'
    var_59 = 5
    var_60 = [var_57, var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = 'a'
    var_64 = 'too_deep'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_17, no_default=var_66)
    var_68 = 'too_deep'
    var_69 = [var_63, var_68]
    var_70 = module_0.get_in(var_69, var_17)
    assert var_70 is None
    var_71 = []
    var_72 = module_0.get_in(var_71, var_17)
    var_73 = {}
    var_74 = 'any'
    var_75 = [var_74]
    var_76 = module_0.get_in(var_75, var_73)
    assert var_76 is None
    var_77 = [var_74]
    var_78 = 'fallback'
    var_79 = module_0.get_in(var_77, var_73, var_78)
    assert var_79 == 'fallback'
    var_80 = 'any'
    var_81 = [var_80]
    var_82 = True
    var_83 = module_0.get_in(var_81, var_73, no_default=var_82)
    var_84 = []
    var_85 = [var_22]
    var_86 = module_0.get_in(var_85, var_84)
    assert var_86 is None
    var_87 = 0
    var_88 = [var_87]
    var_89 = True
    var_90 = module_0.get_in(var_88, var_84, no_default=var_89)



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
    var_14 = True
    var_15 = False
    var_16 = [var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_15]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = 2
    var_25 = [var_1, var_4, var_24, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_15]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is True
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = 'missing_val'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing_val'
    var_38 = 5
    var_39 = [var_1, var_4, var_38]
    var_40 = 'out_of_bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out_of_bounds'
    var_42 = 'not_iterable'
    var_43 = [var_0, var_42]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'c'
    var_52 = 99
    var_53 = [var_50, var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'a'
    var_57 = 'not_iterable'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_17)
    var_63 = 'x'
    var_64 = [var_63]
    var_65 = {}
    var_66 = 'empty'
    var_67 = module_0.get_in(var_64, var_65, var_66)
    assert var_67 == 'empty'
    var_68 = [var_15]
    var_69 = []
    var_70 = module_0.get_in(var_68, var_69, var_66)
    assert var_70 == 'empty'



# Parsed testcases at query #10
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_14]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_15]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = [var_1, var_28]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 == 'default'
    var_33 = [var_1, var_28]
    var_34 = 'missing'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing'
    var_36 = 5
    var_37 = [var_2, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_2, var_36]
    var_40 = 'out_of_bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out_of_bounds'
    var_42 = [var_0, var_14]
    var_43 = module_0.get_in(var_42, var_17)
    assert var_43 is None
    var_44 = [var_0, var_14]
    var_45 = 'not_iterable'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'not_iterable'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'z'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'f'
    var_57 = 5
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'a'
    var_62 = 0
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_17, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_17)
    var_68 = {}
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = module_0.get_in(var_70, var_68)
    assert var_71 is None
    var_72 = 'any'
    var_73 = [var_72]
    var_74 = True
    var_75 = module_0.get_in(var_73, var_68, no_default=var_74)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'f'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 10
    var_7 = 20
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = 'e'
    var_11 = 'found'
    var_12 = {var_10: var_11}
    var_13 = {var_4: var_9, var_5: var_12}
    var_14 = None
    var_15 = False
    var_16 = [var_14, var_15, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_3]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 20
    var_22 = [var_1, var_5, var_10]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_15]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 is None
    var_26 = [var_2, var_3]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is False
    var_28 = 2
    var_29 = [var_2, var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 == 0
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = [var_1, var_31]
    var_35 = 'missing'
    var_36 = module_0.get_in(var_34, var_17, var_35)
    assert var_36 == 'missing'
    var_37 = 99
    var_38 = [var_1, var_4, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_1, var_4, var_37]
    var_41 = 'out of bounds'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'out of bounds'
    var_43 = 'not_a_subdict'
    var_44 = [var_0, var_43]
    var_45 = module_0.get_in(var_44, var_17)
    assert var_45 is None
    var_46 = [var_0, var_43]
    var_47 = 'error'
    var_48 = module_0.get_in(var_46, var_17, var_47)
    assert var_48 == 'error'
    var_49 = 'z'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'b'
    var_54 = 'z'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'b'
    var_59 = 'c'
    var_60 = 99
    var_61 = [var_58, var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = 'a'
    var_65 = 'not_a_subdict'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_17, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_17)



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
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = [var_2, var_6]
    var_36 = 'out_of_bounds'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'out_of_bounds'
    var_38 = 0
    var_39 = [var_0, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_0, var_38]
    var_42 = 'error'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'error'
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'z'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'f'
    var_54 = 10
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'a'
    var_59 = 0
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_17)
    var_65 = [var_59, var_5]
    var_66 = module_0.get_in(var_65, var_17)
    assert var_66 is None



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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'nonexistent'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 99
    var_38 = [var_1, var_4, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_31]
    var_41 = 'missing'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'missing'
    var_43 = [var_1, var_34]
    var_44 = 42
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 42
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'nonexistent'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'f'
    var_56 = 10
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'a'
    var_61 = 'not_a_container'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = {}
    var_68 = 'any'
    var_69 = [var_68]
    var_70 = module_0.get_in(var_69, var_67)
    assert var_70 is None
    var_71 = 'any'
    var_72 = [var_71]
    var_73 = True
    var_74 = module_0.get_in(var_72, var_67, no_default=var_73)



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
    var_32 = 'nonexistent'
    var_33 = [var_1, var_32]
    var_34 = 'missing'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing'
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_1, var_4, var_36]
    var_40 = 'out_of_bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out_of_bounds'
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
    var_53 = 'nonexistent'
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
    var_70 = 'any'
    var_71 = [var_70]
    var_72 = {}
    var_73 = 'empty'
    var_74 = module_0.get_in(var_71, var_72, var_73)
    assert var_74 == 'empty'
    var_75 = 'any'
    var_76 = [var_75]
    var_77 = {}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)



# Parsed testcases at query #15
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 1
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'not_here'
    var_32 = [var_1, var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = [var_1, var_4, var_6]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 'non_existent'
    var_37 = 'sub'
    var_38 = [var_1, var_36, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = 'extra'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_28]
    var_44 = 'missing'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'missing'
    var_46 = [var_1, var_31]
    var_47 = 42
    var_48 = module_0.get_in(var_46, var_17, var_47)
    assert var_48 == 42
    var_49 = 'z'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'b'
    var_54 = 'not_here'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'f'
    var_59 = 5
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = 'a'
    var_64 = 'sub_key'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_17, no_default=var_66)
    var_68 = 'sub_key'
    var_69 = [var_63, var_68]
    var_70 = module_0.get_in(var_69, var_17)
    assert var_70 is None
    var_71 = []
    var_72 = module_0.get_in(var_71, var_17)
    var_73 = 'any'
    var_74 = [var_73]
    var_75 = {}
    var_76 = 'empty'
    var_77 = module_0.get_in(var_74, var_75, var_76)
    assert var_77 == 'empty'
    var_78 = 'any'
    var_79 = [var_78]
    var_80 = {}
    var_81 = True
    var_82 = module_0.get_in(var_79, var_80, no_default=var_81)



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
    var_9 = 'hello'
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'hello'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'x'
    var_35 = [var_1, var_34]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing'
    var_38 = [var_2, var_6]
    var_39 = module_0.get_in(var_38, var_17, var_22)
    assert var_39 == 0
    var_40 = 5
    var_41 = [var_1, var_4, var_40]
    var_42 = 'not found'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'not found'
    var_44 = 'not_a_list'
    var_45 = [var_0, var_44]
    var_46 = 'error'
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 'error'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'x'
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
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'



# Parsed testcases at query #17
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
    var_25 = [var_2, var_3]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 2
    var_27 = 'z'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'not_here'
    var_31 = [var_1, var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = [var_1, var_4, var_6]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'non_existent_key'
    var_36 = 'nested'
    var_37 = [var_1, var_35, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_27]
    var_40 = 'missing'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'missing'
    var_42 = [var_1, var_30]
    var_43 = module_0.get_in(var_42, var_17, var_20)
    assert var_43 == 0
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'missing'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'f'
    var_54 = 5
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'a'
    var_59 = 'extra'
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_17)
    var_65 = [var_59, var_5]
    var_66 = module_0.get_in(var_65, var_17)
    assert var_66 is None



# Parsed testcases at query #18
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
    var_9 = 'hello'
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
    assert var_26 == 'hello'
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
    var_34 = 'nonexistent'
    var_35 = [var_1, var_34]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing'
    var_38 = [var_2, var_6]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_2, var_6]
    var_41 = 'out_of_bounds'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'out_of_bounds'
    var_43 = [var_0, var_22]
    var_44 = module_0.get_in(var_43, var_17)
    assert var_44 is None
    var_45 = [var_0, var_22]
    var_46 = 'type_error'
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 'type_error'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'nonexistent'
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
    var_69 = {}
    var_70 = 'any'
    var_71 = [var_70]
    var_72 = module_0.get_in(var_71, var_69)
    assert var_72 is None
    var_73 = 'any'
    var_74 = [var_73]
    var_75 = True
    var_76 = module_0.get_in(var_74, var_69, no_default=var_75)



# Parsed testcases at query #19
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = 'missing_val'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing_val'
    var_38 = [var_2, var_6]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_2, var_6]
    var_41 = 'out_of_bounds'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'out_of_bounds'
    var_43 = [var_0, var_22]
    var_44 = module_0.get_in(var_43, var_17)
    assert var_44 is None
    var_45 = [var_0, var_22]
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
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = 'fallback'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'fallback'
    var_74 = [var_22]
    var_75 = []
    var_76 = module_0.get_in(var_74, var_75, var_72)
    assert var_76 == 'fallback'



# Parsed testcases at query #20
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
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 is False
    var_27 = [var_1, var_5]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'nonexistent'
    var_33 = [var_1, var_32]
    var_34 = 'missing'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing'
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = 'out_of_bounds'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'out_of_bounds'
    var_40 = 'wrong_key'
    var_41 = [var_1, var_4, var_15, var_40]
    var_42 = 'not_here'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'not_here'
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'nonexistent'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'b'
    var_54 = 'c'
    var_55 = 5
    var_56 = [var_53, var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'a'
    var_60 = 0
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
    var_66 = [var_14, var_22]
    var_67 = 3
    var_68 = 4
    var_69 = [var_67, var_68]
    var_70 = [var_66, var_69]
    var_71 = [var_14, var_15]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 3
    var_73 = [var_36]
    var_74 = 'empty'
    var_75 = module_0.get_in(var_73, var_70, var_74)
    assert var_75 == 'empty'



# Parsed testcases at query #21
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_14]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_15]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 5
    var_35 = [var_1, var_4, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 'too_deep'
    var_38 = [var_0, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = 'non_existent_key'
    var_41 = [var_40]
    var_42 = 'fallback'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'fallback'
    var_44 = [var_1, var_31]
    var_45 = 'not_found'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'not_found'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'missing'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'b'
    var_57 = 'c'
    var_58 = 10
    var_59 = [var_56, var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = 'a'
    var_63 = 0
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_17, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_17)
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = 'empty'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'empty'



# Parsed testcases at query #22
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
    var_30 = 'nonexistent'
    var_31 = [var_1, var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = [var_1, var_4, var_6]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'not_a_dict'
    var_36 = [var_1, var_35, var_20]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_27]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'missing'
    var_41 = [var_1, var_30]
    var_42 = module_0.get_in(var_41, var_17, var_20)
    assert var_42 == 0
    var_43 = 'z'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_17, no_default=var_45)
    var_47 = 'b'
    var_48 = 'nonexistent'
    var_49 = [var_47, var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'f'
    var_53 = 10
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'a'
    var_58 = 0
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = []
    var_63 = module_0.get_in(var_62, var_17)
    var_64 = [var_58, var_5]
    var_65 = module_0.get_in(var_64, var_17)
    assert var_65 is None



# Parsed testcases at query #23
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
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'x'
    var_32 = [var_1, var_31]
    var_33 = 'missing'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = module_0.get_in(var_37, var_17, var_33)
    assert var_38 == 'missing'
    var_39 = 'sub'
    var_40 = [var_0, var_39]
    var_41 = module_0.get_in(var_40, var_17)
    assert var_41 is None
    var_42 = [var_0, var_39]
    var_43 = 'error'
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 'error'
    var_45 = 'z'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'b'
    var_50 = 'x'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'f'
    var_55 = 10
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'a'
    var_60 = 'sub'
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
    var_66 = [var_59]
    var_67 = {}
    var_68 = 'empty'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'empty'



# Parsed testcases at query #24
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
    var_33 = 'z'
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_18)
    assert var_35 is None
    var_36 = 'nonexistent'
    var_37 = [var_1, var_36]
    var_38 = module_0.get_in(var_37, var_18)
    assert var_38 is None
    var_39 = 5
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_18)
    assert var_41 is None
    var_42 = 'invalid_key'
    var_43 = [var_1, var_5, var_15, var_42]
    var_44 = module_0.get_in(var_43, var_18)
    assert var_44 is None
    var_45 = [var_33]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_45, var_18, var_46)
    assert var_47 == 'missing'
    var_48 = [var_1, var_36]
    var_49 = 42
    var_50 = module_0.get_in(var_48, var_18, var_49)
    assert var_50 == 42
    var_51 = 'z'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_18, no_default=var_53)
    var_55 = 'b'
    var_56 = 'nonexistent'
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
    var_67 = 0
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_18, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_18)
    var_73 = [var_66]
    var_74 = {}
    var_75 = 'empty'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'empty'



# Parsed testcases at query #25
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
    var_34 = module_0.get_in(var_33, var_17, var_32)
    assert var_34 == 'missing'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'out_of_bounds'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'out_of_bounds'
    var_40 = [var_0, var_20]
    var_41 = module_0.get_in(var_40, var_17)
    assert var_41 is None
    var_42 = [var_0, var_20]
    var_43 = 'error'
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 'error'
    var_45 = 'z'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'b'
    var_50 = 'missing'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'f'
    var_55 = 10
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'a'
    var_60 = 0
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
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



# Parsed testcases at query #26
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
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 'nonexistent'
    var_33 = [var_1, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 99
    var_36 = [var_1, var_4, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = 'sub'
    var_39 = [var_1, var_32, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_29]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'missing'
    var_44 = [var_1, var_32]
    var_45 = module_0.get_in(var_44, var_17, var_22)
    assert var_45 == 0
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'nonexistent'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'f'
    var_56 = 5
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
    var_67 = [var_61, var_5]
    var_68 = module_0.get_in(var_67, var_17)
    assert var_68 is None



# Parsed testcases at query #27
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
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'not_found'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'not_found'
    var_40 = 'sub_key'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_40]
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
    var_67 = 'any'
    var_68 = [var_67]
    var_69 = {}
    var_70 = 'empty'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'empty'



# Parsed testcases at query #28
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 1
    var_26 = [var_1, var_4, var_14]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 == 10
    var_28 = 'non_existent'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = [var_1, var_28]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = 99
    var_34 = [var_1, var_4, var_33]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 'too_deep'
    var_37 = [var_0, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = 'x'
    var_40 = [var_39]
    var_41 = 'missing'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'missing'
    var_43 = 'z'
    var_44 = [var_1, var_43]
    var_45 = module_0.get_in(var_44, var_17, var_14)
    assert var_45 == 0
    var_46 = 'non_existent'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'non_existent'
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
    var_65 = 'any'
    var_66 = [var_65]
    var_67 = {}
    var_68 = 'empty'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'empty'
    var_70 = 'any'
    var_71 = [var_70]
    var_72 = {}
    var_73 = True
    var_74 = module_0.get_in(var_71, var_72, no_default=var_73)
    var_75 = []
    var_76 = module_0.get_in(var_75, var_17)



# Parsed testcases at query #29
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4, var_14]
    var_21 = module_0.get_in(var_20, var_17)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_15, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_15]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = 'missing_val'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing_val'
    var_35 = [var_1, var_4, var_6]
    var_36 = 'out_of_bounds'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'out_of_bounds'
    var_38 = 'non_existent_index'
    var_39 = [var_0, var_38]
    var_40 = 'not_an_iterable'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'not_an_iterable'
    var_42 = 'z'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_17, no_default=var_44)
    var_46 = 'b'
    var_47 = 'non_existent'
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'c'
    var_53 = 10
    var_54 = [var_51, var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'a'
    var_58 = 0
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = []
    var_63 = module_0.get_in(var_62, var_17)
    var_64 = 'any'
    var_65 = [var_64]
    var_66 = {}
    var_67 = 'fallback'
    var_68 = module_0.get_in(var_65, var_66, var_67)
    assert var_68 == 'fallback'
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = True
    var_73 = module_0.get_in(var_70, var_71, no_default=var_72)



# Parsed testcases at query #30
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
    var_9 = 'hello'
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
    assert var_24 == 'hello'
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 is False
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
    var_42 = [var_0, var_15]
    var_43 = module_0.get_in(var_42, var_17)
    assert var_43 is None
    var_44 = [var_0, var_15]
    var_45 = 'error'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'error'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'missing'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'b'
    var_57 = 'c'
    var_58 = 5
    var_59 = [var_56, var_57, var_58]
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
    var_71 = 'fallback'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'fallback'
    var_73 = 'a'
    var_74 = [var_73]
    var_75 = {}
    var_76 = True
    var_77 = module_0.get_in(var_74, var_75, no_default=var_76)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = 'missing_val'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing_val'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'out_of_bounds'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'out_of_bounds'
    var_40 = 'not_a_subdict'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_40]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
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
    var_62 = 'any'
    var_63 = [var_62]
    var_64 = {}
    var_65 = 'empty'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'empty'



# Parsed testcases at query #2
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
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = 'missing_val'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing_val'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'not_found'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'not_found'
    var_40 = 'too_deep'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_40]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'missing'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'f'
    var_56 = 10
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'a'
    var_61 = 'too_deep'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = {}
    var_68 = 'any'
    var_69 = [var_68]
    var_70 = module_0.get_in(var_69, var_67)
    assert var_70 is None
    var_71 = 'any'
    var_72 = [var_71]
    var_73 = True
    var_74 = module_0.get_in(var_72, var_67, no_default=var_73)



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
    var_32 = 'x'
    var_33 = [var_1, var_32]
    var_34 = 'missing'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing'
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = 'out of bounds'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'out of bounds'
    var_40 = 'nonexistent'
    var_41 = [var_1, var_40, var_15]
    var_42 = 42
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 42
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'x'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'b'
    var_54 = 'c'
    var_55 = 10
    var_56 = [var_53, var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'too_deep'
    var_60 = [var_53, var_59]
    var_61 = 'error'
    var_62 = module_0.get_in(var_60, var_17, var_61)
    assert var_62 == 'error'
    var_63 = 'a'
    var_64 = 'too_deep'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_17, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_17)
    var_70 = [var_14, var_22]
    var_71 = 3
    var_72 = 4
    var_73 = [var_71, var_72]
    var_74 = [var_70, var_73]
    var_75 = [var_14, var_15]
    var_76 = module_0.get_in(var_75, var_74)
    assert var_76 == 3



# Parsed testcases at query #4
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = 'missing_val'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing_val'
    var_38 = [var_2, var_6]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = [var_2, var_6]
    var_41 = 'not_found'
    var_42 = module_0.get_in(var_40, var_17, var_41)
    assert var_42 == 'not_found'
    var_43 = [var_0, var_22]
    var_44 = module_0.get_in(var_43, var_17)
    assert var_44 is None
    var_45 = [var_0, var_22]
    var_46 = 'error'
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 'error'
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'f'
    var_53 = 10
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'a'
    var_58 = 0
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = []
    var_63 = module_0.get_in(var_62, var_17)
    var_64 = 'any'
    var_65 = [var_64]
    var_66 = {}
    var_67 = 'empty'
    var_68 = module_0.get_in(var_65, var_66, var_67)
    assert var_68 == 'empty'
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = True
    var_73 = module_0.get_in(var_70, var_71, no_default=var_72)



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
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = [var_2, var_6]
    var_36 = 0
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 0
    var_38 = 'not_a_container'
    var_39 = [var_0, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = 'invalid'
    var_42 = [var_0, var_41]
    var_43 = 'error'
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 'error'
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
    var_60 = 0
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
    var_66 = [var_60, var_5]
    var_67 = module_0.get_in(var_66, var_17)
    assert var_67 is None



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
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_14, var_8]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 2
    var_26 = 'nonexistent'
    var_27 = [var_26]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is None
    var_29 = 'not_here'
    var_30 = [var_1, var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 5
    var_33 = [var_1, var_4, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'z'
    var_36 = [var_1, var_35]
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_17, var_37)
    assert var_38 == 'missing'
    var_39 = 'nonexistent'
    var_40 = [var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_17, no_default=var_41)
    var_43 = 'b'
    var_44 = 'not_here'
    var_45 = [var_43, var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'f'
    var_49 = 10
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'invalid'
    var_54 = [var_48, var_53]
    var_55 = module_0.get_in(var_54, var_17)
    assert var_55 is None
    var_56 = 'a'
    var_57 = 'invalid'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'first'
    var_62 = 'second'
    var_63 = {var_62: var_14}
    var_64 = (var_61, var_63)
    var_65 = [var_64]
    var_66 = '0'
    var_67 = [var_66, var_62]
    var_68 = {var_62: var_14}
    var_69 = {var_66: var_68}
    var_70 = module_0.get_in(var_67, var_69)
    assert var_70 == 2
    var_71 = 'key'
    var_72 = (var_59, var_14)
    var_73 = {var_71: var_72}
    var_74 = [var_73]
    var_75 = 0
    var_76 = [var_75, var_71, var_59]
    var_77 = module_0.get_in(var_76, var_74)
    assert var_77 == 2
    var_78 = []
    var_79 = module_0.get_in(var_78, var_17)
    var_80 = [var_57, var_5]
    var_81 = module_0.get_in(var_80, var_17)
    assert var_81 is None
    var_82 = 'extra'
    var_83 = [var_57, var_5, var_82]
    var_84 = module_0.get_in(var_83, var_17)
    assert var_84 is None



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
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = [var_1, var_29]
    var_33 = 'missing'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'out_of_bounds'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'out_of_bounds'
    var_40 = 'not_an_index'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_40]
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
    var_61 = 'not_an_index'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = [var_61, var_5]
    var_68 = module_0.get_in(var_67, var_17)
    assert var_68 is None



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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_14]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = [var_1, var_4, var_15, var_8]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 'found'
    var_26 = [var_2, var_15]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 == 2
    var_28 = [var_1, var_5]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'z'
    var_31 = [var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = [var_1, var_30]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 5
    var_36 = [var_1, var_4, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = 'nonexistent'
    var_39 = [var_1, var_4, var_14, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_38]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'missing'
    var_44 = [var_1, var_30]
    var_45 = module_0.get_in(var_44, var_17, var_42)
    assert var_45 == 'missing'
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
    var_60 = 'not_an_index'
    var_61 = [var_55, var_60]
    var_62 = module_0.get_in(var_61, var_17)
    assert var_62 is None
    var_63 = 'a'
    var_64 = 'not_an_index'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_17, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_17)
    var_70 = [var_66, var_15]
    var_71 = 3
    var_72 = 4
    var_73 = [var_71, var_72]
    var_74 = [var_70, var_73]
    var_75 = [var_14, var_66]
    var_76 = module_0.get_in(var_75, var_74)
    assert var_76 == 2
    var_77 = [var_35]
    var_78 = module_0.get_in(var_77, var_74)
    assert var_78 is None



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
    var_9 = 'hello'
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
    assert var_24 == 'hello'
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
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_1, var_29]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'missing'
    var_41 = 'nonexistent'
    var_42 = [var_41]
    var_43 = 42
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 42
    var_45 = 5
    var_46 = [var_2, var_45]
    var_47 = module_0.get_in(var_46, var_17)
    assert var_47 is None
    var_48 = [var_1, var_4, var_6]
    var_49 = module_0.get_in(var_48, var_17)
    assert var_49 is None
    var_50 = [var_0, var_20]
    var_51 = module_0.get_in(var_50, var_17)
    assert var_51 is None
    var_52 = 'z'
    var_53 = [var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'b'
    var_57 = 'z'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'f'
    var_62 = 5
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_17, no_default=var_64)
    var_66 = 'a'
    var_67 = 0
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_17, no_default=var_69)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_17)
    var_73 = [var_66]
    var_74 = {}
    var_75 = 'fallback'
    var_76 = module_0.get_in(var_73, var_74, var_75)
    assert var_76 == 'fallback'



# Parsed testcases at query #10
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
    var_15 = [var_3, var_14]
    var_16 = {var_0: var_3, var_1: var_13, var_2: var_15}
    var_17 = [var_0]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 == 1
    var_19 = 0
    var_20 = [var_1, var_4, var_19]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 == 10
    var_22 = [var_1, var_4, var_14, var_8]
    var_23 = module_0.get_in(var_22, var_16)
    assert var_23 == 'found'
    var_24 = [var_2, var_3]
    var_25 = module_0.get_in(var_24, var_16)
    assert var_25 == 2
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_16)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_16)
    assert var_30 is None
    var_31 = 'nonexistent'
    var_32 = [var_1, var_31]
    var_33 = module_0.get_in(var_32, var_16)
    assert var_33 is None
    var_34 = 5
    var_35 = [var_2, var_34]
    var_36 = module_0.get_in(var_35, var_16)
    assert var_36 is None
    var_37 = [var_1, var_4, var_6]
    var_38 = module_0.get_in(var_37, var_16)
    assert var_38 is None
    var_39 = [var_28]
    var_40 = 'missing'
    var_41 = module_0.get_in(var_39, var_16, var_40)
    assert var_41 == 'missing'
    var_42 = 'x'
    var_43 = [var_1, var_42]
    var_44 = 42
    var_45 = module_0.get_in(var_43, var_16, var_44)
    assert var_45 == 42
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_16, no_default=var_48)
    var_50 = 'b'
    var_51 = 'z'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_16, no_default=var_53)
    var_55 = 'f'
    var_56 = 10
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_16, no_default=var_58)
    var_60 = 'sub_key'
    var_61 = [var_55, var_60]
    var_62 = module_0.get_in(var_61, var_16)
    assert var_62 is None
    var_63 = 'a'
    var_64 = 'sub_key'
    var_65 = [var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_16, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_16)
    var_70 = [var_64, var_67, var_19, var_31]
    var_71 = 'fallback'
    var_72 = module_0.get_in(var_70, var_16, var_71)
    assert var_72 == 'fallback'



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
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_21, var_37)
    assert var_38 == 'missing'
    var_39 = 5
    var_40 = [var_1, var_5, var_39]
    var_41 = 0
    var_42 = module_0.get_in(var_40, var_21, var_41)
    assert var_42 == 0
    var_43 = 'not_an_index'
    var_44 = [var_0, var_43]
    var_45 = 'error'
    var_46 = module_0.get_in(var_44, var_21, var_45)
    assert var_46 == 'error'
    var_47 = 'b'
    var_48 = 'nonexistent'
    var_49 = [var_47, var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_21, no_default=var_50)
    var_52 = 'b'
    var_53 = 'd'
    var_54 = 99
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_21, no_default=var_56)
    var_58 = 'a'
    var_59 = 0
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_21, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_21)
    var_65 = 'any'
    var_66 = [var_65]
    var_67 = {}
    var_68 = 'empty'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'empty'



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
    var_34 = module_0.get_in(var_33, var_17, var_32)
    assert var_34 == 'missing'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = 'not_found'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'not_found'
    var_40 = 'sub_key'
    var_41 = [var_0, var_40]
    var_42 = module_0.get_in(var_41, var_17)
    assert var_42 is None
    var_43 = [var_0, var_40]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'missing'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_17, no_default=var_53)
    var_55 = 'f'
    var_56 = 10
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'a'
    var_61 = 'sub_key'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = 'any'
    var_68 = [var_67]
    var_69 = {}
    var_70 = 'empty'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'empty'



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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_1, var_4, var_6]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_31]
    var_40 = module_0.get_in(var_39, var_17, var_34)
    assert var_40 == 'missing'
    var_41 = 'x'
    var_42 = [var_1, var_41]
    var_43 = 42
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 42
    var_45 = 'z'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'b'
    var_50 = 'nonexistent'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'f'
    var_55 = 10
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'a'
    var_60 = 'not_an_index'
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_17)
    var_66 = [var_62, var_14]
    var_67 = 4
    var_68 = [var_15, var_67]
    var_69 = [var_66, var_68]
    var_70 = [var_62, var_22]
    var_71 = module_0.get_in(var_70, var_69)
    assert var_71 == 3
    var_72 = [var_22, var_62]
    var_73 = module_0.get_in(var_72, var_69)
    assert var_73 == 2



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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_14]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = [var_1, var_4, var_15, var_8]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 'found'
    var_26 = [var_2, var_3]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 == 1
    var_28 = [var_1, var_5]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'z'
    var_31 = [var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = 'missing'
    var_34 = [var_1, var_33]
    var_35 = 'missing_val'
    var_36 = module_0.get_in(var_34, var_17, var_35)
    assert var_36 == 'missing_val'
    var_37 = [var_2, var_6]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_2, var_6]
    var_40 = 'out_of_bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out_of_bounds'
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
    var_57 = 'f'
    var_58 = 10
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = 'a'
    var_63 = 'not_an_index'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_17, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_17)
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = {}
    var_72 = 'fallback'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'fallback'



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
    var_32 = 'non'
    var_33 = 'existent'
    var_34 = [var_32, var_33]
    var_35 = module_0.get_in(var_34, var_21)
    assert var_35 is None
    var_36 = 'z'
    var_37 = [var_1, var_36]
    var_38 = module_0.get_in(var_37, var_21)
    assert var_38 is None
    var_39 = 99
    var_40 = [var_1, var_5, var_39]
    var_41 = module_0.get_in(var_40, var_21)
    assert var_41 is None
    var_42 = 'too'
    var_43 = 'deep'
    var_44 = [var_0, var_42, var_43]
    var_45 = module_0.get_in(var_44, var_21)
    assert var_45 is None
    var_46 = [var_32, var_33]
    var_47 = 'missing'
    var_48 = module_0.get_in(var_46, var_21, var_47)
    assert var_48 == 'missing'
    var_49 = [var_1, var_36]
    var_50 = 0
    var_51 = module_0.get_in(var_49, var_21, var_50)
    assert var_51 == 0
    var_52 = 'non'
    var_53 = 'existent'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_21, no_default=var_55)
    var_57 = 'b'
    var_58 = 'z'
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
    var_69 = 'not_indexable'
    var_70 = [var_68, var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_21, no_default=var_71)
    var_73 = []
    var_74 = module_0.get_in(var_73, var_21)
    var_75 = [var_70, var_50]
    var_76 = module_0.get_in(var_75, var_21)
    assert var_76 is None



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
    var_25 = [var_2, var_14]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 is False
    var_27 = 'nonexistent'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'missing'
    var_31 = [var_1, var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = 5
    var_34 = [var_1, var_4, var_33]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 'invalid_key'
    var_37 = [var_1, var_4, var_36]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = 'x'
    var_40 = [var_39]
    var_41 = module_0.get_in(var_40, var_17, var_30)
    assert var_41 == 'missing'
    var_42 = 'z'
    var_43 = [var_1, var_42]
    var_44 = module_0.get_in(var_43, var_17, var_15)
    assert var_44 == 0
    var_45 = 'nonexistent'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_17, no_default=var_47)
    var_49 = 'b'
    var_50 = 'missing'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'b'
    var_55 = 'c'
    var_56 = 10
    var_57 = [var_54, var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_17, no_default=var_58)
    var_60 = 'a'
    var_61 = 0
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_17, no_default=var_63)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_17)
    var_67 = [var_61, var_59]
    var_68 = module_0.get_in(var_67, var_17)
    assert var_68 is None



# Parsed testcases at query #17
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
    var_7 = 3
    var_8 = 4
    var_9 = 'e'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = [var_7, var_8, var_11]
    var_13 = {var_4: var_6, var_5: var_12}
    var_14 = None
    var_15 = 'hello'
    var_16 = [var_14, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 2
    var_24 = 0
    var_25 = [var_1, var_5, var_24]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 3
    var_27 = [var_1, var_5, var_6, var_9]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 5
    var_29 = [var_2, var_3]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 == 'hello'
    var_31 = [var_2, var_24]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = 'z'
    var_34 = [var_33]
    var_35 = module_0.get_in(var_34, var_17)
    assert var_35 is None
    var_36 = 'missing'
    var_37 = [var_1, var_36]
    var_38 = 'fallback'
    var_39 = module_0.get_in(var_37, var_17, var_38)
    assert var_39 == 'fallback'
    var_40 = 10
    var_41 = [var_1, var_5, var_40]
    var_42 = 99
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 99
    var_44 = 'extra'
    var_45 = [var_0, var_44]
    var_46 = module_0.get_in(var_45, var_17, var_36)
    assert var_46 == 'missing'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'nonexistent'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'b'
    var_57 = 'd'
    var_58 = 99
    var_59 = [var_56, var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = 'a'
    var_63 = 'not_an_index'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_17, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_17)
    var_69 = {}
    var_70 = 'any'
    var_71 = [var_70]
    var_72 = module_0.get_in(var_71, var_69)
    assert var_72 is None
    var_73 = 'any'
    var_74 = [var_73]
    var_75 = True
    var_76 = module_0.get_in(var_74, var_69, no_default=var_75)



# Parsed testcases at query #18
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
    var_25 = [var_2, var_3]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 2
    var_27 = 'z'
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = [var_1, var_27]
    var_31 = 'missing'
    var_32 = module_0.get_in(var_30, var_17, var_31)
    assert var_32 == 'missing'
    var_33 = [var_2, var_6]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = [var_2, var_6]
    var_36 = 'out_of_bounds'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'out_of_bounds'
    var_38 = 'not_an_index'
    var_39 = [var_0, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_0, var_38]
    var_42 = 'error'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'error'
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'f'
    var_49 = 10
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'a'
    var_54 = 'not_an_index'
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = [var_54, var_5]
    var_59 = module_0.get_in(var_58, var_17)
    assert var_59 is None
    var_60 = []
    var_61 = module_0.get_in(var_60, var_17)
    var_62 = 'any'
    var_63 = [var_62]
    var_64 = {}
    var_65 = 'empty'
    var_66 = module_0.get_in(var_63, var_64, var_65)
    assert var_66 == 'empty'



# Parsed testcases at query #19
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
    var_26 = [var_1, var_5]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 is None
    var_28 = 'z'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'missing'
    var_32 = [var_1, var_31]
    var_33 = 'fallback'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'fallback'
    var_35 = [var_2, var_6]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = [var_2, var_6]
    var_38 = module_0.get_in(var_37, var_17, var_33)
    assert var_38 == 'fallback'
    var_39 = 'not_an_index'
    var_40 = [var_0, var_39]
    var_41 = module_0.get_in(var_40, var_17)
    assert var_41 is None
    var_42 = [var_0, var_39]
    var_43 = module_0.get_in(var_42, var_17, var_33)
    assert var_43 == 'fallback'
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'missing'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'f'
    var_54 = 10
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'a'
    var_59 = 'not_an_index'
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_17)
    var_65 = 'any'
    var_66 = [var_65]
    var_67 = {}
    var_68 = module_0.get_in(var_66, var_67, var_31)
    assert var_68 == 'missing'



# Parsed testcases at query #20
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_15]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = 2
    var_25 = [var_1, var_4, var_24, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_15]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is True
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = 'missing_val'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing_val'
    var_38 = 99
    var_39 = [var_1, var_4, var_38]
    var_40 = module_0.get_in(var_39, var_17, var_15)
    assert var_40 == 0
    var_41 = 5
    var_42 = [var_2, var_41]
    var_43 = 'out_of_bounds'
    var_44 = module_0.get_in(var_42, var_17, var_43)
    assert var_44 == 'out_of_bounds'
    var_45 = 'not_a_container'
    var_46 = [var_0, var_45]
    var_47 = 'error'
    var_48 = module_0.get_in(var_46, var_17, var_47)
    assert var_48 == 'error'
    var_49 = 'z'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'f'
    var_54 = 5
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'a'
    var_59 = 'sub_key'
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_17)
    var_65 = 'x'
    var_66 = [var_65]
    var_67 = {}
    var_68 = 'none'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'none'
    var_70 = [var_15]
    var_71 = []
    var_72 = module_0.get_in(var_70, var_71, var_68)
    assert var_72 == 'none'



# Parsed testcases at query #21
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
    var_32 = 'x'
    var_33 = [var_1, var_32]
    var_34 = 'missing'
    var_35 = module_0.get_in(var_33, var_17, var_34)
    assert var_35 == 'missing'
    var_36 = [var_2, var_6]
    var_37 = module_0.get_in(var_36, var_17, var_20)
    assert var_37 == 0
    var_38 = 5
    var_39 = [var_1, var_4, var_38]
    var_40 = 'out of bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out of bounds'
    var_42 = 'not_a_key'
    var_43 = [var_0, var_42]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_17, var_44)
    assert var_45 == 'error'
    var_46 = 'z'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_17, no_default=var_48)
    var_50 = 'b'
    var_51 = 'x'
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
    var_67 = 'any'
    var_68 = [var_67]
    var_69 = {}
    var_70 = 'empty'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'empty'
    var_72 = 'any'
    var_73 = [var_72]
    var_74 = {}
    var_75 = True
    var_76 = module_0.get_in(var_73, var_74, no_default=var_75)



# Parsed testcases at query #22
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
    var_15 = [var_3, var_14]
    var_16 = {var_0: var_3, var_1: var_13, var_2: var_15}
    var_17 = [var_0]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 == 1
    var_19 = [var_1, var_4]
    var_20 = module_0.get_in(var_19, var_16)
    var_21 = [var_1, var_4, var_14, var_8]
    var_22 = module_0.get_in(var_21, var_16)
    assert var_22 == 'found'
    var_23 = [var_2, var_3]
    var_24 = module_0.get_in(var_23, var_16)
    assert var_24 == 2
    var_25 = 'z'
    var_26 = [var_25]
    var_27 = module_0.get_in(var_26, var_16)
    assert var_27 is None
    var_28 = 'missing'
    var_29 = [var_1, var_28]
    var_30 = module_0.get_in(var_29, var_16, var_28)
    assert var_30 == 'missing'
    var_31 = 5
    var_32 = [var_2, var_31]
    var_33 = 'out_of_bounds'
    var_34 = module_0.get_in(var_32, var_16, var_33)
    assert var_34 == 'out_of_bounds'
    var_35 = 'not_a_dict'
    var_36 = [var_0, var_35]
    var_37 = 'error'
    var_38 = module_0.get_in(var_36, var_16, var_37)
    assert var_38 == 'error'
    var_39 = 'z'
    var_40 = [var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_16, no_default=var_41)
    var_43 = 'f'
    var_44 = 5
    var_45 = [var_43, var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_16, no_default=var_46)
    var_48 = 'a'
    var_49 = 'not_a_dict'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_16, no_default=var_51)
    var_53 = []
    var_54 = module_0.get_in(var_53, var_16)
    var_55 = {}
    var_56 = 'any'
    var_57 = [var_56]
    var_58 = module_0.get_in(var_57, var_55)
    assert var_58 is None
    var_59 = 'any'
    var_60 = [var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_55, no_default=var_61)



# Parsed testcases at query #23
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 99
    var_38 = [var_1, var_4, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = 'x'
    var_41 = [var_40]
    var_42 = module_0.get_in(var_41, var_17, var_34)
    assert var_42 == 'missing'
    var_43 = 'nonexistent'
    var_44 = [var_1, var_43]
    var_45 = 42
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 42
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'missing'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'f'
    var_57 = 10
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'a'
    var_62 = 'too_deep'
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_17, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_17)
    var_68 = [var_61]
    var_69 = {}
    var_70 = 'fallback'
    var_71 = module_0.get_in(var_68, var_69, var_70)
    assert var_71 == 'fallback'
    var_72 = 'a'
    var_73 = [var_72]
    var_74 = {}
    var_75 = True
    var_76 = module_0.get_in(var_73, var_74, no_default=var_75)



# Parsed testcases at query #24
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = 0
    var_23 = [var_1, var_4, var_22]
    var_24 = module_0.get_in(var_23, var_17)
    assert var_24 == 10
    var_25 = [var_1, var_4, var_14, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_14]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 3
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'x'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'z'
    var_35 = [var_1, var_34]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing'
    var_38 = 5
    var_39 = [var_2, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_2, var_38]
    var_42 = 'out_of_bounds'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'out_of_bounds'
    var_44 = 'non_subscriptable'
    var_45 = [var_0, var_44]
    var_46 = module_0.get_in(var_45, var_17)
    assert var_46 is None
    var_47 = [var_0, var_44]
    var_48 = 'error'
    var_49 = module_0.get_in(var_47, var_17, var_48)
    assert var_49 == 'error'
    var_50 = 'x'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'b'
    var_55 = 'z'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'f'
    var_60 = 5
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = 'a'
    var_65 = 'not_an_index'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_17, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_17)
    var_71 = 'any'
    var_72 = [var_71]
    var_73 = {}
    var_74 = 'empty'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'empty'



# Parsed testcases at query #25
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
    var_20 = [var_1]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_15]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = 2
    var_25 = [var_1, var_4, var_24, var_8]
    var_26 = module_0.get_in(var_25, var_17)
    assert var_26 == 'found'
    var_27 = [var_2, var_15]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 is True
    var_29 = [var_1, var_5]
    var_30 = module_0.get_in(var_29, var_17)
    assert var_30 is None
    var_31 = 'z'
    var_32 = [var_31]
    var_33 = module_0.get_in(var_32, var_17)
    assert var_33 is None
    var_34 = 'non_existent'
    var_35 = [var_1, var_34]
    var_36 = module_0.get_in(var_35, var_17)
    assert var_36 is None
    var_37 = 99
    var_38 = [var_1, var_4, var_37]
    var_39 = module_0.get_in(var_38, var_17)
    assert var_39 is None
    var_40 = 'x'
    var_41 = [var_40]
    var_42 = 'missing'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'missing'
    var_44 = 'missing_key'
    var_45 = [var_1, var_44]
    var_46 = 42
    var_47 = module_0.get_in(var_45, var_17, var_46)
    assert var_47 == 42
    var_48 = 'z'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_17, no_default=var_50)
    var_52 = 'b'
    var_53 = 'missing_key'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_17, no_default=var_55)
    var_57 = 'f'
    var_58 = 5
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_17, no_default=var_60)
    var_62 = 'a'
    var_63 = 'sub_key'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_17, no_default=var_65)
    var_67 = []
    var_68 = module_0.get_in(var_67, var_17)
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



# Parsed testcases at query #26
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
    var_29 = 'nonexistent'
    var_30 = [var_1, var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = 5
    var_33 = [var_1, var_4, var_32]
    var_34 = module_0.get_in(var_33, var_17)
    assert var_34 is None
    var_35 = 'not_a_dict'
    var_36 = [var_1, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_26]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'missing'
    var_41 = [var_1, var_29]
    var_42 = 0
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 0
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_17, no_default=var_46)
    var_48 = 'b'
    var_49 = 'nonexistent'
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_17, no_default=var_51)
    var_53 = 'f'
    var_54 = 10
    var_55 = [var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_17, no_default=var_56)
    var_58 = 'a'
    var_59 = 0
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_17, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_17)
    var_65 = [var_60, var_42]
    var_66 = [var_61, var_14, var_15]
    var_67 = module_0.get_in(var_65, var_66)
    assert var_67 == 1



# Parsed testcases at query #27
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
    var_26 = 0
    var_27 = [var_1, var_4, var_26]
    var_28 = module_0.get_in(var_27, var_17)
    assert var_28 == 10
    var_29 = 'z'
    var_30 = [var_29]
    var_31 = module_0.get_in(var_30, var_17)
    assert var_31 is None
    var_32 = [var_1, var_29]
    var_33 = 'missing'
    var_34 = module_0.get_in(var_32, var_17, var_33)
    assert var_34 == 'missing'
    var_35 = 5
    var_36 = [var_2, var_35]
    var_37 = module_0.get_in(var_36, var_17)
    assert var_37 is None
    var_38 = [var_2, var_35]
    var_39 = 'out of bounds'
    var_40 = module_0.get_in(var_38, var_17, var_39)
    assert var_40 == 'out of bounds'
    var_41 = 'not_subscriptable'
    var_42 = [var_0, var_41]
    var_43 = module_0.get_in(var_42, var_17)
    assert var_43 is None
    var_44 = [var_0, var_41]
    var_45 = 'error'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'error'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'z'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'f'
    var_57 = 5
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'a'
    var_62 = 0
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_17, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_17)
    var_68 = [var_62, var_5]
    var_69 = module_0.get_in(var_68, var_17)
    assert var_69 is None



# Parsed testcases at query #28
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
    var_34 = 'missing'
    var_35 = [var_1, var_34]
    var_36 = 'missing_val'
    var_37 = module_0.get_in(var_35, var_17, var_36)
    assert var_37 == 'missing_val'
    var_38 = 5
    var_39 = [var_2, var_38]
    var_40 = module_0.get_in(var_39, var_17)
    assert var_40 is None
    var_41 = [var_2, var_38]
    var_42 = 'out_of_bounds'
    var_43 = module_0.get_in(var_41, var_17, var_42)
    assert var_43 == 'out_of_bounds'
    var_44 = 'not_subscriptable'
    var_45 = [var_0, var_44]
    var_46 = module_0.get_in(var_45, var_17)
    assert var_46 is None
    var_47 = [var_0, var_44]
    var_48 = 'error'
    var_49 = module_0.get_in(var_47, var_17, var_48)
    assert var_49 == 'error'
    var_50 = 'z'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_17, no_default=var_52)
    var_54 = 'b'
    var_55 = 'missing'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_17, no_default=var_57)
    var_59 = 'f'
    var_60 = 5
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_17, no_default=var_62)
    var_64 = 'a'
    var_65 = 'not_subscriptable'
    var_66 = [var_64, var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_17, no_default=var_67)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_17)
    var_71 = {}
    var_72 = 'any'
    var_73 = [var_72]
    var_74 = module_0.get_in(var_73, var_71)
    assert var_74 is None
    var_75 = 'any'
    var_76 = [var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_71, no_default=var_77)



# Parsed testcases at query #29
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
    var_15 = [var_3, var_14]
    var_16 = {var_0: var_3, var_1: var_13, var_2: var_15}
    var_17 = [var_0]
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 == 1
    var_19 = [var_1, var_4]
    var_20 = module_0.get_in(var_19, var_16)
    var_21 = 0
    var_22 = [var_1, var_4, var_21]
    var_23 = module_0.get_in(var_22, var_16)
    assert var_23 == 10
    var_24 = [var_1, var_4, var_14, var_8]
    var_25 = module_0.get_in(var_24, var_16)
    assert var_25 == 'found'
    var_26 = [var_2, var_3]
    var_27 = module_0.get_in(var_26, var_16)
    assert var_27 == 2
    var_28 = [var_1, var_5]
    var_29 = module_0.get_in(var_28, var_16)
    assert var_29 is None
    var_30 = 'z'
    var_31 = [var_30]
    var_32 = module_0.get_in(var_31, var_16)
    assert var_32 is None
    var_33 = 'x'
    var_34 = [var_1, var_33]
    var_35 = module_0.get_in(var_34, var_16)
    assert var_35 is None
    var_36 = 5
    var_37 = [var_1, var_4, var_36]
    var_38 = module_0.get_in(var_37, var_16)
    assert var_38 is None
    var_39 = 'not_an_int'
    var_40 = [var_1, var_4, var_39]
    var_41 = module_0.get_in(var_40, var_16)
    assert var_41 is None
    var_42 = 'too_deep'
    var_43 = [var_0, var_42]
    var_44 = module_0.get_in(var_43, var_16)
    assert var_44 is None
    var_45 = [var_30]
    var_46 = 'missing'
    var_47 = module_0.get_in(var_45, var_16, var_46)
    assert var_47 == 'missing'
    var_48 = [var_1, var_33]
    var_49 = module_0.get_in(var_48, var_16, var_21)
    assert var_49 == 0
    var_50 = 'z'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_16, no_default=var_52)
    var_54 = 'b'
    var_55 = 'x'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_16, no_default=var_57)
    var_59 = 'b'
    var_60 = 'c'
    var_61 = 99
    var_62 = [var_59, var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_16, no_default=var_63)
    var_65 = 'a'
    var_66 = 0
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_16, no_default=var_68)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_16)
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



# Parsed testcases at query #30
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
    var_14 = 0
    var_15 = 2
    var_16 = [var_14, var_3, var_15]
    var_17 = {var_0: var_3, var_1: var_13, var_2: var_16}
    var_18 = [var_0]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 1
    var_20 = [var_1, var_4]
    var_21 = module_0.get_in(var_20, var_17)
    var_22 = [var_1, var_4, var_14]
    var_23 = module_0.get_in(var_22, var_17)
    assert var_23 == 10
    var_24 = [var_1, var_4, var_15, var_8]
    var_25 = module_0.get_in(var_24, var_17)
    assert var_25 == 'found'
    var_26 = [var_2, var_15]
    var_27 = module_0.get_in(var_26, var_17)
    assert var_27 == 2
    var_28 = [var_1, var_5]
    var_29 = module_0.get_in(var_28, var_17)
    assert var_29 is None
    var_30 = 'z'
    var_31 = [var_30]
    var_32 = module_0.get_in(var_31, var_17)
    assert var_32 is None
    var_33 = 'missing'
    var_34 = [var_1, var_33]
    var_35 = 'missing_val'
    var_36 = module_0.get_in(var_34, var_17, var_35)
    assert var_36 == 'missing_val'
    var_37 = [var_2, var_6]
    var_38 = module_0.get_in(var_37, var_17)
    assert var_38 is None
    var_39 = [var_2, var_6]
    var_40 = 'out_of_bounds'
    var_41 = module_0.get_in(var_39, var_17, var_40)
    assert var_41 == 'out_of_bounds'
    var_42 = [var_0, var_14]
    var_43 = module_0.get_in(var_42, var_17)
    assert var_43 is None
    var_44 = [var_0, var_14]
    var_45 = 'error'
    var_46 = module_0.get_in(var_44, var_17, var_45)
    assert var_46 == 'error'
    var_47 = 'z'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_17, no_default=var_49)
    var_51 = 'b'
    var_52 = 'nonexistent'
    var_53 = [var_51, var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_17, no_default=var_54)
    var_56 = 'f'
    var_57 = 10
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_17, no_default=var_59)
    var_61 = 'a'
    var_62 = 0
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_17, no_default=var_64)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_17)
    var_68 = {}
    var_69 = 'any'
    var_70 = [var_69]
    var_71 = module_0.get_in(var_70, var_68)
    assert var_71 is None
    var_72 = [var_69]
    var_73 = 'default'
    var_74 = module_0.get_in(var_72, var_68, var_73)
    assert var_74 == 'default'
    var_75 = 'any'
    var_76 = [var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_68, no_default=var_77)



