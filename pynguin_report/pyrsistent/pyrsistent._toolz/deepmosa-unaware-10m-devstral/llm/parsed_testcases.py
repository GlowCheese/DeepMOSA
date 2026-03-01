####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_18 = None
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 is None
    var_20 = [var_0, var_2]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_16, var_21)
    assert var_22 == 0
    var_23 = {var_0: var_3}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_27, var_9, var_10]
    var_29 = {var_24: var_28}
    var_30 = 'a'
    var_31 = 5
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_29, no_default=var_33)
    var_35 = {var_30: var_33}
    var_36 = 'a'
    var_37 = 'b'
    var_38 = [var_36, var_37]
    var_39 = True
    var_40 = module_0.get_in(var_38, var_35, no_default=var_39)
    var_41 = {var_36: var_39}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = [var_39, var_9, var_10]
    var_45 = {var_37: var_44}
    var_46 = 4
    var_47 = {var_38: var_46}
    var_48 = [var_45, var_47]
    var_49 = {var_36: var_48}
    var_50 = [var_36, var_21, var_37, var_39]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 2
    var_52 = [var_36, var_39, var_38]
    var_53 = module_0.get_in(var_52, var_49)
    assert var_53 == 4
    var_54 = {var_37: var_39}
    var_55 = {var_36: var_54}
    var_56 = 'd'
    var_57 = [var_36, var_37, var_38, var_56]
    var_58 = 'missing'
    var_59 = module_0.get_in(var_57, var_55, var_58)
    assert var_59 == 'missing'



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
    var_18 = 3
    var_19 = 4
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = [var_3, var_21]
    var_23 = [var_3, var_3, var_15]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 3
    var_25 = [var_3, var_3]
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
    var_50 = None
    var_51 = {var_43: var_50}
    var_52 = [var_44]
    var_53 = 'default'
    var_54 = module_0.get_in(var_52, var_51, var_53)
    assert var_54 == 'default'
    var_55 = [var_43]
    var_56 = module_0.get_in(var_55, var_51, var_53)
    assert var_56 is None



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
    var_30 = 5
    var_31 = 6
    var_32 = [var_30, var_31]
    var_33 = 7
    var_34 = 8
    var_35 = [var_33, var_34]
    var_36 = [var_32, var_35]
    var_37 = [var_29, var_36]
    var_38 = [var_22, var_3, var_3]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 4
    var_40 = [var_3, var_22, var_22]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 5
    var_42 = [var_24]
    var_43 = module_0.get_in(var_42, var_37)
    assert var_43 is None
    var_44 = [var_24]
    var_45 = 'out_of_bounds'
    var_46 = module_0.get_in(var_44, var_37, var_45)
    assert var_46 == 'out_of_bounds'
    var_47 = [var_3, var_24, var_26]
    var_48 = {var_1: var_47}
    var_49 = [var_27, var_30, var_31]
    var_50 = {var_2: var_49}
    var_51 = [var_48, var_50]
    var_52 = {var_0: var_51}
    var_53 = [var_0, var_22, var_1, var_3]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 == 2
    var_55 = [var_0, var_3, var_2, var_22]
    var_56 = module_0.get_in(var_55, var_52)
    assert var_56 == 4
    var_57 = [var_0, var_24]
    var_58 = module_0.get_in(var_57, var_52)
    assert var_58 is None
    var_59 = [var_0, var_24]
    var_60 = 'missing'
    var_61 = module_0.get_in(var_59, var_52, var_60)
    assert var_61 == 'missing'
    var_62 = {var_0: var_3}
    var_63 = 'b'
    var_64 = [var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_62, no_default=var_65)
    var_67 = {var_63: var_66}
    var_68 = []
    var_69 = module_0.get_in(var_68, var_67)
    var_70 = None
    var_71 = {var_64: var_70}
    var_72 = {var_63: var_71}
    var_73 = [var_63, var_64]
    var_74 = module_0.get_in(var_73, var_72)
    assert var_74 is None
    var_75 = [var_63, var_64]
    var_76 = module_0.get_in(var_75, var_72, var_17)
    assert var_76 is None
    var_77 = [var_63, var_65]
    var_78 = module_0.get_in(var_77, var_72, var_17)
    assert var_78 == 'default'



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
    var_25 = 3
    var_26 = 4
    var_27 = 5
    var_28 = 6
    var_29 = [var_27, var_28]
    var_30 = [var_25, var_26, var_29]
    var_31 = [var_3, var_24, var_30]
    var_32 = [var_22]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 1
    var_34 = [var_24, var_3]
    var_35 = module_0.get_in(var_34, var_31)
    assert var_35 == 4
    var_36 = [var_24, var_24, var_3]
    var_37 = module_0.get_in(var_36, var_31)
    assert var_37 == 6
    var_38 = [var_25]
    var_39 = module_0.get_in(var_38, var_31)
    assert var_39 is None
    var_40 = [var_24, var_25]
    var_41 = module_0.get_in(var_40, var_31)
    assert var_41 is None
    var_42 = [var_24, var_25]
    var_43 = -1
    var_44 = module_0.get_in(var_42, var_31, var_43)
    assert var_44 == -1
    var_45 = {var_1: var_25}
    var_46 = [var_3, var_24, var_45]
    var_47 = {var_0: var_46}
    var_48 = [var_0, var_24, var_1]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 3
    var_50 = [var_0, var_3]
    var_51 = module_0.get_in(var_50, var_47)
    assert var_51 == 2
    var_52 = [var_0, var_25]
    var_53 = module_0.get_in(var_52, var_47)
    assert var_53 is None
    var_54 = [var_0, var_25]
    var_55 = module_0.get_in(var_54, var_47, var_22)
    assert var_55 == 0
    var_56 = {var_0: var_3}
    var_57 = 'b'
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_56, no_default=var_59)
    var_61 = 0
    var_62 = [var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_56, no_default=var_63)
    var_65 = {var_61: var_64}
    var_66 = []
    var_67 = module_0.get_in(var_66, var_65)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_65, var_17)
    var_70 = {}
    var_71 = [var_61]
    var_72 = module_0.get_in(var_71, var_70, var_17)
    assert var_72 == 'default'
    var_73 = [var_61, var_62]
    var_74 = module_0.get_in(var_73, var_70, var_22)
    assert var_74 == 0



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
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_3, var_19, var_22]
    var_24 = 0
    var_25 = [var_19, var_24]
    var_26 = module_0.get_in(var_25, var_23)
    assert var_26 == 3
    var_27 = [var_19, var_3]
    var_28 = module_0.get_in(var_27, var_23)
    assert var_28 == 4
    var_29 = [var_19, var_19]
    var_30 = module_0.get_in(var_29, var_23)
    assert var_30 is None
    var_31 = [var_19, var_19]
    var_32 = module_0.get_in(var_31, var_23, var_17)
    assert var_32 == 'default'
    var_33 = {var_1: var_20}
    var_34 = [var_3, var_19, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_19, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 3
    var_38 = [var_0, var_19, var_2]
    var_39 = module_0.get_in(var_38, var_35)
    assert var_39 is None
    var_40 = [var_0, var_19, var_2]
    var_41 = module_0.get_in(var_40, var_35, var_17)
    assert var_41 == 'default'
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
    var_54 = module_0.get_in(var_53, var_50, var_17)
    assert var_54 == 'default'



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
    var_21 = [var_3, var_20]
    var_22 = 3
    var_23 = 4
    var_24 = [var_22, var_23]
    var_25 = [var_21, var_24]
    var_26 = 0
    var_27 = [var_26, var_3]
    var_28 = module_0.get_in(var_27, var_25)
    assert var_28 == 2
    var_29 = [var_3, var_26]
    var_30 = module_0.get_in(var_29, var_25)
    assert var_30 == 3
    var_31 = [var_20]
    var_32 = module_0.get_in(var_31, var_25)
    assert var_32 is None
    var_33 = [var_26, var_20]
    var_34 = module_0.get_in(var_33, var_25)
    assert var_34 is None
    var_35 = {var_1: var_20}
    var_36 = [var_3, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_3, var_1]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_0, var_3]
    var_41 = module_0.get_in(var_40, var_37)
    var_42 = [var_0, var_26]
    var_43 = module_0.get_in(var_42, var_37)
    assert var_43 == 1
    var_44 = [var_0, var_20]
    var_45 = module_0.get_in(var_44, var_37)
    assert var_45 is None
    var_46 = [var_13]
    var_47 = 'default'
    var_48 = module_0.get_in(var_46, var_37, var_47)
    assert var_48 == 'default'
    var_49 = [var_0, var_13]
    var_50 = module_0.get_in(var_49, var_37, var_47)
    assert var_50 == 'default'
    var_51 = [var_0, var_3, var_13]
    var_52 = module_0.get_in(var_51, var_37, var_47)
    assert var_52 == 'default'
    var_53 = 'x'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_37, no_default=var_55)
    var_57 = 'a'
    var_58 = 'x'
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.get_in(var_59, var_37, no_default=var_60)
    var_62 = 'a'
    var_63 = 1
    var_64 = 'x'
    var_65 = [var_62, var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_37, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_37)
    var_70 = [var_13]
    var_71 = {}
    var_72 = module_0.get_in(var_70, var_71)
    assert var_72 is None
    var_73 = [var_13]
    var_74 = {}
    var_75 = module_0.get_in(var_73, var_74, var_47)
    assert var_75 == 'default'
    var_76 = 'x'
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
    var_34 = [var_22]
    var_35 = module_0.get_in(var_34, var_29)
    var_36 = [var_24]
    var_37 = module_0.get_in(var_36, var_29)
    assert var_37 is None
    var_38 = [var_24]
    var_39 = 'out of bounds'
    var_40 = module_0.get_in(var_38, var_29, var_39)
    assert var_40 == 'out of bounds'
    var_41 = {var_1: var_24}
    var_42 = [var_3, var_41]
    var_43 = {var_0: var_42}
    var_44 = [var_0, var_3, var_1]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 2
    var_46 = [var_0, var_3]
    var_47 = module_0.get_in(var_46, var_43)
    var_48 = [var_0, var_22]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 == 1
    var_50 = [var_0, var_24]
    var_51 = module_0.get_in(var_50, var_43)
    assert var_51 is None
    var_52 = 'y'
    var_53 = [var_13, var_52]
    var_54 = module_0.get_in(var_53, var_43)
    assert var_54 is None
    var_55 = {var_0: var_3}
    var_56 = 'x'
    var_57 = [var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_55, no_default=var_58)
    var_60 = 'a'
    var_61 = 'b'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_55, no_default=var_63)
    var_65 = {var_60: var_63}
    var_66 = []
    var_67 = module_0.get_in(var_66, var_65)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_65, var_17)
    var_70 = {var_60: var_63}
    var_71 = [var_60, var_61]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 is None
    var_73 = [var_60, var_61]
    var_74 = module_0.get_in(var_73, var_70, var_17)
    assert var_74 == 'default'



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
    var_40 = 'x'
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_33, no_default=var_42)
    var_44 = 2
    var_45 = [var_44]
    var_46 = 1
    var_47 = [var_46, var_44]
    var_48 = True
    var_49 = module_0.get_in(var_45, var_47, no_default=var_48)
    var_50 = [var_11]
    var_51 = 'default'
    var_52 = module_0.get_in(var_50, var_33, var_51)
    assert var_52 == 'default'
    var_53 = [var_17]
    var_54 = [var_47, var_17]
    var_55 = module_0.get_in(var_53, var_54, var_51)
    assert var_55 == 'default'
    var_56 = []
    var_57 = module_0.get_in(var_56, var_33)



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
    var_18 = 3
    var_19 = [var_17, var_18]
    var_20 = 4
    var_21 = [var_3, var_19, var_20]
    var_22 = [var_3, var_15]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 2
    var_24 = [var_3]
    var_25 = module_0.get_in(var_24, var_21)
    var_26 = 5
    var_27 = [var_26]
    var_28 = module_0.get_in(var_27, var_21)
    assert var_28 is None
    var_29 = [var_26]
    var_30 = module_0.get_in(var_29, var_21, var_15)
    assert var_30 == 0
    var_31 = {var_1: var_17}
    var_32 = [var_3, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_3, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 2
    var_36 = [var_0, var_3]
    var_37 = module_0.get_in(var_36, var_33)
    var_38 = [var_0, var_26]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_0, var_26]
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
    var_51 = [var_11]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_11]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0
    var_55 = 'x'
    var_56 = [var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_50, no_default=var_57)



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
    var_19 = [var_13]
    var_20 = True
    var_21 = module_0.get_in(var_19, var_6, no_default=var_20)
    assert var_21 is None
    var_22 = 2
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_20, var_22, var_25]
    var_27 = [var_22, var_20]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 4
    var_29 = [var_22]
    var_30 = module_0.get_in(var_29, var_26)
    var_31 = [var_23]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_23]
    var_34 = module_0.get_in(var_33, var_26, var_17)
    assert var_34 == 0
    var_35 = {var_1: var_22}
    var_36 = [var_20, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_20, var_1]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_0, var_20]
    var_41 = module_0.get_in(var_40, var_37)
    var_42 = [var_0, var_20, var_2]
    var_43 = module_0.get_in(var_42, var_37)
    assert var_43 is None
    var_44 = {var_0: var_20}
    var_45 = 'b'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_44, no_default=var_47)
    var_49 = {var_45: var_20}
    var_50 = []
    var_51 = module_0.get_in(var_50, var_49)
    var_52 = None
    var_53 = {var_46: var_52}
    var_54 = {var_45: var_53}
    var_55 = [var_45, var_46]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 is None
    var_57 = [var_45, var_46]
    var_58 = module_0.get_in(var_57, var_54, var_17)
    assert var_58 is None
    var_59 = [var_45, var_47]
    var_60 = module_0.get_in(var_59, var_54, var_17)
    assert var_60 == 0
    var_61 = 123
    var_62 = [var_45]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_45]
    var_65 = module_0.get_in(var_64, var_61, var_17)
    assert var_65 == 0



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
    var_18 = 'default'
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 'default'
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 5
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 'default'
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
    var_36 = {var_31: var_35}
    var_37 = [var_31, var_33]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 is None
    var_39 = {var_32: var_34}
    var_40 = {var_33: var_9}
    var_41 = [var_39, var_40]
    var_42 = {var_31: var_41}
    var_43 = [var_31, var_34, var_33]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 2
    var_45 = {var_31: var_34}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = {var_31: var_34}
    var_49 = [var_31, var_32]
    var_50 = module_0.get_in(var_49, var_48, var_18)
    assert var_50 == 'default'



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
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_3, var_26]
    var_28 = [var_3, var_3, var_20]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 3
    var_30 = [var_20]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 1
    var_32 = [var_3, var_3]
    var_33 = module_0.get_in(var_32, var_27)
    var_34 = 5
    var_35 = [var_34]
    var_36 = module_0.get_in(var_35, var_27)
    assert var_36 is None
    var_37 = [var_3, var_3, var_34]
    var_38 = -1
    var_39 = module_0.get_in(var_37, var_27, var_38)
    assert var_39 == -1
    var_40 = {var_1: var_22}
    var_41 = [var_3, var_40]
    var_42 = {var_0: var_41}
    var_43 = [var_0, var_3, var_1]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 2
    var_45 = [var_0, var_20]
    var_46 = module_0.get_in(var_45, var_42)
    assert var_46 == 1
    var_47 = [var_0, var_3]
    var_48 = module_0.get_in(var_47, var_42)
    var_49 = [var_13]
    var_50 = module_0.get_in(var_49, var_42)
    assert var_50 is None
    var_51 = {var_0: var_3}
    var_52 = 'x'
    var_53 = [var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_51, no_default=var_54)
    var_56 = 0
    var_57 = [var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_51, no_default=var_58)
    var_60 = {var_56: var_59}
    var_61 = []
    var_62 = module_0.get_in(var_61, var_60)
    var_63 = 'y'
    var_64 = 'z'
    var_65 = [var_13, var_63, var_64]
    var_66 = {}
    var_67 = 'not found'
    var_68 = module_0.get_in(var_65, var_66, var_67)
    assert var_68 == 'not found'
    var_69 = [var_20, var_59, var_22]
    var_70 = []
    var_71 = module_0.get_in(var_69, var_70, var_20)
    assert var_71 == 0



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
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = 2
    var_18 = 3
    var_19 = 4
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = [var_3, var_21]
    var_23 = [var_3, var_3, var_15]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 3
    var_25 = [var_3, var_3]
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
    var_40 = {var_0: var_3}
    var_41 = [var_1]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_40, no_default=var_42)
    assert var_43 is None
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_40, no_default=var_46)
    var_48 = {var_44: var_42}
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
    assert var_59 == 0



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
    var_39 = [var_0, var_3, var_11]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = {var_42: var_45}
    var_47 = []
    var_48 = module_0.get_in(var_47, var_46)
    var_49 = {var_42: var_45}
    var_50 = [var_42, var_43]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 is None
    var_52 = [var_42, var_43]
    var_53 = module_0.get_in(var_52, var_49, var_15)
    assert var_53 == 'default'
    var_54 = None
    var_55 = {var_43: var_54}
    var_56 = {var_42: var_55}
    var_57 = [var_42, var_43]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 is None
    var_59 = [var_42, var_43]
    var_60 = module_0.get_in(var_59, var_56, var_15)
    assert var_60 is None
    var_61 = [var_42, var_11]
    var_62 = module_0.get_in(var_61, var_56, var_15)
    assert var_62 == 'default'



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
    var_25 = 3
    var_26 = [var_3, var_24, var_25]
    var_27 = 4
    var_28 = 5
    var_29 = 6
    var_30 = [var_27, var_28, var_29]
    var_31 = [var_26, var_30]
    var_32 = [var_22, var_3]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 2
    var_34 = [var_3, var_24]
    var_35 = module_0.get_in(var_34, var_31)
    assert var_35 == 6
    var_36 = [var_24]
    var_37 = module_0.get_in(var_36, var_31)
    assert var_37 is None
    var_38 = [var_24]
    var_39 = 'out of range'
    var_40 = module_0.get_in(var_38, var_31, var_39)
    assert var_40 == 'out of range'
    var_41 = [var_22, var_25]
    var_42 = module_0.get_in(var_41, var_31)
    assert var_42 is None
    var_43 = [var_22, var_25]
    var_44 = module_0.get_in(var_43, var_31, var_22)
    assert var_44 == 0
    var_45 = {var_1: var_25}
    var_46 = [var_3, var_24, var_45]
    var_47 = {var_0: var_46}
    var_48 = [var_0, var_24, var_1]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 3
    var_50 = [var_0, var_22]
    var_51 = module_0.get_in(var_50, var_47)
    assert var_51 == 1
    var_52 = [var_0, var_24]
    var_53 = module_0.get_in(var_52, var_47)
    var_54 = [var_0, var_25]
    var_55 = module_0.get_in(var_54, var_47)
    assert var_55 is None
    var_56 = [var_0, var_25]
    var_57 = 'not found'
    var_58 = module_0.get_in(var_56, var_47, var_57)
    assert var_58 == 'not found'
    var_59 = {var_0: var_3}
    var_60 = 'b'
    var_61 = [var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_59, no_default=var_62)
    var_64 = 0
    var_65 = [var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_59, no_default=var_66)
    var_68 = {var_64: var_67}
    var_69 = []
    var_70 = module_0.get_in(var_69, var_68)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_68, var_17)
    var_73 = 'string'
    var_74 = []
    var_75 = module_0.get_in(var_74, var_73)
    var_76 = [var_22]
    var_77 = module_0.get_in(var_76, var_73)
    assert var_77 == 's'
    var_78 = [var_67]
    var_79 = module_0.get_in(var_78, var_73)
    assert var_79 == 't'
    var_80 = 10
    var_81 = [var_80]
    var_82 = module_0.get_in(var_81, var_73)
    assert var_82 is None
    var_83 = [var_80]
    var_84 = module_0.get_in(var_83, var_73, var_39)
    assert var_84 == 'out of range'



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
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 'default'
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = [var_29, var_3]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 == 2
    var_32 = [var_3, var_29]
    var_33 = module_0.get_in(var_32, var_28)
    assert var_33 == 3
    var_34 = [var_29]
    var_35 = module_0.get_in(var_34, var_28)
    var_36 = [var_23]
    var_37 = module_0.get_in(var_36, var_28)
    assert var_37 is None
    var_38 = [var_23]
    var_39 = module_0.get_in(var_38, var_28, var_17)
    assert var_39 == 'default'
    var_40 = [var_29, var_23]
    var_41 = module_0.get_in(var_40, var_28)
    assert var_41 is None
    var_42 = [var_29, var_23]
    var_43 = module_0.get_in(var_42, var_28, var_17)
    assert var_43 == 'default'
    var_44 = {var_1: var_23}
    var_45 = [var_3, var_44]
    var_46 = {var_0: var_45}
    var_47 = [var_0, var_3, var_1]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2
    var_49 = [var_0, var_3]
    var_50 = module_0.get_in(var_49, var_46)
    var_51 = [var_0, var_29]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 == 1
    var_53 = [var_0, var_23]
    var_54 = module_0.get_in(var_53, var_46)
    assert var_54 is None
    var_55 = [var_0, var_23]
    var_56 = module_0.get_in(var_55, var_46, var_17)
    assert var_56 == 'default'
    var_57 = [var_0, var_3, var_13]
    var_58 = module_0.get_in(var_57, var_46)
    assert var_58 is None
    var_59 = [var_0, var_3, var_13]
    var_60 = module_0.get_in(var_59, var_46, var_17)
    assert var_60 == 'default'
    var_61 = {var_0: var_3}
    var_62 = 'x'
    var_63 = [var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_61, no_default=var_64)
    var_66 = 'a'
    var_67 = 'x'
    var_68 = [var_66, var_67]
    var_69 = True
    var_70 = module_0.get_in(var_68, var_61, no_default=var_69)
    var_71 = {var_66: var_69}
    var_72 = []
    var_73 = module_0.get_in(var_72, var_71)
    var_74 = []
    var_75 = module_0.get_in(var_74, var_71, var_17)
    var_76 = {var_66: var_69}
    var_77 = [var_66]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 == 1
    var_79 = [var_67]
    var_80 = module_0.get_in(var_79, var_76)
    assert var_80 is None
    var_81 = [var_67]
    var_82 = module_0.get_in(var_81, var_76, var_17)
    assert var_82 == 'default'
    var_83 = {var_66: var_69}
    var_84 = [var_66, var_67]
    var_85 = module_0.get_in(var_84, var_83)
    assert var_85 is None
    var_86 = [var_66, var_67]
    var_87 = module_0.get_in(var_86, var_83, var_17)
    assert var_87 == 'default'
    var_88 = {var_67: var_69}
    var_89 = [var_66, var_67]
    var_90 = module_0.get_in(var_89, var_83)
    assert var_90 == 1
    var_91 = [var_13]
    var_92 = module_0.get_in(var_91, var_83)
    assert var_92 is None



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
    var_45 = [var_44]
    var_46 = {var_35: var_45}
    var_47 = 0
    var_48 = [var_35, var_47, var_36, var_38]
    var_49 = module_0.get_in(var_48, var_46)
    assert var_49 == 2



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
    var_29 = {var_1: var_17}
    var_30 = [var_3, var_29]
    var_31 = {var_0: var_30}
    var_32 = [var_0, var_3, var_1]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 2
    var_34 = [var_0, var_15]
    var_35 = module_0.get_in(var_34, var_31)
    assert var_35 == 1
    var_36 = [var_0, var_3, var_2]
    var_37 = module_0.get_in(var_36, var_31)
    assert var_37 is None
    var_38 = {var_0: var_3}
    var_39 = [var_0]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 1
    var_41 = 'b'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_38, no_default=var_43)
    var_45 = {var_41: var_44}
    var_46 = [var_41, var_42]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 is None
    var_48 = [var_41, var_42]
    var_49 = module_0.get_in(var_48, var_45, var_15)
    assert var_49 == 0
    var_50 = {var_41: var_44}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = None
    var_54 = {var_42: var_53}
    var_55 = {var_41: var_54}
    var_56 = [var_41, var_42]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_41, var_42]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 is None
    var_60 = [var_41, var_43]
    var_61 = module_0.get_in(var_60, var_55, var_15)
    assert var_61 == 0



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
    var_17 = 'default'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'default'
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = [var_3, var_23]
    var_25 = 0
    var_26 = [var_3, var_3, var_25]
    var_27 = module_0.get_in(var_26, var_24)
    assert var_27 == 3
    var_28 = [var_3, var_3]
    var_29 = module_0.get_in(var_28, var_24)
    var_30 = [var_3]
    var_31 = module_0.get_in(var_30, var_24)
    var_32 = [var_19]
    var_33 = module_0.get_in(var_32, var_24)
    assert var_33 is None
    var_34 = [var_19]
    var_35 = module_0.get_in(var_34, var_24, var_17)
    assert var_35 == 'default'
    var_36 = {var_1: var_19}
    var_37 = [var_3, var_36]
    var_38 = {var_0: var_37}
    var_39 = [var_0, var_3, var_1]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 2
    var_41 = [var_0, var_3]
    var_42 = module_0.get_in(var_41, var_38)
    var_43 = [var_0, var_25]
    var_44 = module_0.get_in(var_43, var_38)
    assert var_44 == 1
    var_45 = [var_0, var_19]
    var_46 = module_0.get_in(var_45, var_38)
    assert var_46 is None
    var_47 = {var_0: var_3}
    var_48 = 'b'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = [var_49]
    var_53 = module_0.get_in(var_52, var_47, var_25)
    assert var_53 == 0
    var_54 = [var_49]
    var_55 = None
    var_56 = module_0.get_in(var_54, var_47, var_55)
    assert var_56 is None
    var_57 = []
    var_58 = module_0.get_in(var_57, var_47)
    var_59 = {var_48: var_51}
    var_60 = [var_48, var_49]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 is None
    var_62 = [var_48, var_49]
    var_63 = module_0.get_in(var_62, var_59, var_17)
    assert var_63 == 'default'



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
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 'default'
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = 5
    var_30 = 6
    var_31 = [var_29, var_30]
    var_32 = 7
    var_33 = 8
    var_34 = [var_32, var_33]
    var_35 = [var_31, var_34]
    var_36 = [var_28, var_35]
    var_37 = 0
    var_38 = [var_37, var_3, var_3]
    var_39 = module_0.get_in(var_38, var_36)
    assert var_39 == 4
    var_40 = [var_3, var_37, var_37]
    var_41 = module_0.get_in(var_40, var_36)
    assert var_41 == 5
    var_42 = [var_37]
    var_43 = module_0.get_in(var_42, var_36)
    var_44 = [var_23]
    var_45 = module_0.get_in(var_44, var_36)
    assert var_45 is None
    var_46 = [var_23]
    var_47 = module_0.get_in(var_46, var_36, var_17)
    assert var_47 == 'default'
    var_48 = [var_37, var_3, var_23]
    var_49 = module_0.get_in(var_48, var_36)
    assert var_49 is None
    var_50 = [var_37, var_3, var_23]
    var_51 = module_0.get_in(var_50, var_36, var_17)
    assert var_51 == 'default'
    var_52 = [var_3, var_23, var_25]
    var_53 = {var_1: var_52}
    var_54 = [var_26, var_29, var_30]
    var_55 = {var_2: var_54}
    var_56 = [var_53, var_55]
    var_57 = {var_0: var_56}
    var_58 = [var_0, var_37, var_1, var_3]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 == 2
    var_60 = [var_0, var_3, var_2, var_37]
    var_61 = module_0.get_in(var_60, var_57)
    assert var_61 == 4
    var_62 = [var_0, var_37]
    var_63 = module_0.get_in(var_62, var_57)
    var_64 = [var_13]
    var_65 = module_0.get_in(var_64, var_57)
    assert var_65 is None
    var_66 = [var_13]
    var_67 = module_0.get_in(var_66, var_57, var_17)
    assert var_67 == 'default'
    var_68 = [var_0, var_23]
    var_69 = module_0.get_in(var_68, var_57)
    assert var_69 is None
    var_70 = [var_0, var_23]
    var_71 = module_0.get_in(var_70, var_57, var_17)
    assert var_71 == 'default'
    var_72 = {var_2: var_3}
    var_73 = {var_1: var_72}
    var_74 = {var_0: var_73}
    var_75 = 'x'
    var_76 = [var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_74, no_default=var_77)
    var_79 = 'a'
    var_80 = 'b'
    var_81 = 'x'
    var_82 = [var_79, var_80, var_81]
    var_83 = True
    var_84 = module_0.get_in(var_82, var_74, no_default=var_83)
    var_85 = {var_79: var_82}
    var_86 = []
    var_87 = module_0.get_in(var_86, var_85)
    var_88 = None
    var_89 = {var_81: var_88}
    var_90 = {var_80: var_89}
    var_91 = {var_79: var_90}
    var_92 = [var_79, var_80, var_81]
    var_93 = module_0.get_in(var_92, var_91)
    assert var_93 is None
    var_94 = [var_79, var_80, var_81]
    var_95 = module_0.get_in(var_94, var_91, var_17)
    assert var_95 is None
    var_96 = [var_79, var_80, var_13]
    var_97 = module_0.get_in(var_96, var_91, var_17)
    assert var_97 == 'default'



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
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 'default'
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
    var_42 = [var_26, var_9, var_10]
    var_43 = {var_24: var_42}
    var_44 = 4
    var_45 = {var_25: var_44}
    var_46 = [var_43, var_45]
    var_47 = {var_23: var_46}
    var_48 = 0
    var_49 = [var_23, var_48, var_24, var_26]
    var_50 = module_0.get_in(var_49, var_47)
    assert var_50 == 2
    var_51 = [var_23, var_26, var_25]
    var_52 = module_0.get_in(var_51, var_47)
    assert var_52 == 4



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
    var_57 = 0
    var_58 = [var_43, var_46, var_57]
    var_59 = [var_42]
    var_60 = 'total'
    var_61 = [var_43, var_60]
    var_62 = [var_43, var_60]



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
    var_15 = 0
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 0
    var_17 = [var_0, var_1, var_11]
    var_18 = module_0.get_in(var_17, var_6)
    assert var_18 is None
    var_19 = [var_0, var_1, var_11]
    var_20 = module_0.get_in(var_19, var_6, var_15)
    assert var_20 == 0
    var_21 = 2
    var_22 = [var_3, var_21]
    var_23 = 3
    var_24 = 4
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = [var_15, var_3]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 == 2
    var_29 = [var_3, var_15]
    var_30 = module_0.get_in(var_29, var_26)
    assert var_30 == 3
    var_31 = [var_21]
    var_32 = module_0.get_in(var_31, var_26)
    assert var_32 is None
    var_33 = [var_21]
    var_34 = module_0.get_in(var_33, var_26, var_15)
    assert var_34 == 0
    var_35 = [var_15, var_21]
    var_36 = module_0.get_in(var_35, var_26)
    assert var_36 is None
    var_37 = [var_15, var_21]
    var_38 = module_0.get_in(var_37, var_26, var_15)
    assert var_38 == 0
    var_39 = {var_1: var_21}
    var_40 = [var_3, var_39]
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_3, var_1]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 2
    var_44 = [var_0, var_3]
    var_45 = module_0.get_in(var_44, var_41)
    var_46 = [var_0, var_15]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 == 1
    var_48 = [var_11]
    var_49 = module_0.get_in(var_48, var_41)
    assert var_49 is None
    var_50 = [var_11]
    var_51 = module_0.get_in(var_50, var_41, var_15)
    assert var_51 == 0
    var_52 = [var_0, var_21]
    var_53 = module_0.get_in(var_52, var_41)
    assert var_53 is None
    var_54 = [var_0, var_21]
    var_55 = module_0.get_in(var_54, var_41, var_15)
    assert var_55 == 0
    var_56 = {var_0: var_3}
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
    var_69 = {var_61: var_64}
    var_70 = [var_61]
    var_71 = module_0.get_in(var_70, var_69)
    assert var_71 == 1
    var_72 = [var_62]
    var_73 = module_0.get_in(var_72, var_69)
    assert var_73 is None
    var_74 = [var_62]
    var_75 = module_0.get_in(var_74, var_69, var_15)
    assert var_75 == 0
    var_76 = 1
    var_77 = [var_61]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 is None
    var_79 = [var_61]
    var_80 = module_0.get_in(var_79, var_76, var_15)
    assert var_80 == 0
    var_81 = 'a'
    var_82 = [var_81]
    var_83 = True
    var_84 = module_0.get_in(var_82, var_76, no_default=var_83)



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
    var_47 = 0
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_42, no_default=var_49)
    var_51 = {var_47: var_50}
    var_52 = []
    var_53 = module_0.get_in(var_52, var_51)
    var_54 = {var_47: var_50}
    var_55 = [var_47]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 == 1
    var_57 = [var_48]
    var_58 = module_0.get_in(var_57, var_54)
    assert var_58 is None
    var_59 = [var_48]
    var_60 = module_0.get_in(var_59, var_54, var_15)
    assert var_60 == 0



# Parsed testcases at query #26
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
    var_41 = [var_0, var_3, var_11]
    var_42 = module_0.get_in(var_41, var_36)
    assert var_42 is None
    var_43 = {var_0: var_3}
    var_44 = 'x'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = 0
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_43, no_default=var_50)
    var_52 = {var_48: var_51}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = {var_48: var_51}
    var_56 = [var_48]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 == 1
    var_58 = [var_49]
    var_59 = module_0.get_in(var_58, var_55)
    assert var_59 is None



# Parsed testcases at query #27
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
    var_43 = {var_36: var_38}
    var_44 = {var_37: var_9}
    var_45 = [var_43, var_44]
    var_46 = {var_35: var_45}
    var_47 = [var_35, var_38, var_37]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2



# Parsed testcases at query #28
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
    var_14 = [var_0, var_11]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_1, var_11]
    var_17 = module_0.get_in(var_16, var_6)
    assert var_17 is None
    var_18 = [var_11]
    var_19 = 'default'
    var_20 = module_0.get_in(var_18, var_6, var_19)
    assert var_20 == 'default'
    var_21 = [var_0, var_11]
    var_22 = module_0.get_in(var_21, var_6, var_19)
    assert var_22 == 'default'
    var_23 = [var_0, var_1, var_11]
    var_24 = module_0.get_in(var_23, var_6, var_19)
    assert var_24 == 'default'
    var_25 = 'x'
    var_26 = [var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_6, no_default=var_27)
    var_29 = 'a'
    var_30 = 'x'
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_6, no_default=var_32)
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 'x'
    var_37 = [var_34, var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_6, no_default=var_38)
    var_40 = 2
    var_41 = 3
    var_42 = 4
    var_43 = 5
    var_44 = 6
    var_45 = [var_43, var_44]
    var_46 = [var_41, var_42, var_45]
    var_47 = [var_37, var_40, var_46]
    var_48 = [var_40, var_40, var_37]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 6
    var_50 = 0
    var_51 = [var_50]
    var_52 = module_0.get_in(var_51, var_47)
    assert var_52 == 1
    var_53 = [var_43]
    var_54 = module_0.get_in(var_53, var_47)
    assert var_54 is None
    var_55 = [var_40, var_43]
    var_56 = module_0.get_in(var_55, var_47)
    assert var_56 is None
    var_57 = [var_40, var_40, var_43]
    var_58 = module_0.get_in(var_57, var_47)
    assert var_58 is None
    var_59 = {var_35: var_40}
    var_60 = [var_37, var_59]
    var_61 = {var_34: var_60}
    var_62 = [var_34, var_37, var_35]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 == 2
    var_64 = [var_34, var_50]
    var_65 = module_0.get_in(var_64, var_61)
    assert var_65 == 1
    var_66 = [var_11]
    var_67 = module_0.get_in(var_66, var_61)
    assert var_67 is None
    var_68 = [var_34, var_43]
    var_69 = module_0.get_in(var_68, var_61)
    assert var_69 is None
    var_70 = [var_34, var_37, var_11]
    var_71 = module_0.get_in(var_70, var_61)
    assert var_71 is None
    var_72 = []
    var_73 = module_0.get_in(var_72, var_6)
    var_74 = []
    var_75 = module_0.get_in(var_74, var_47)
    var_76 = []
    var_77 = module_0.get_in(var_76, var_61)
    var_78 = 42
    var_79 = [var_11]
    var_80 = module_0.get_in(var_79, var_78)
    assert var_80 is None
    var_81 = [var_11]
    var_82 = module_0.get_in(var_81, var_78, var_19)
    assert var_82 == 'default'
    var_83 = 'x'
    var_84 = [var_83]
    var_85 = True
    var_86 = module_0.get_in(var_84, var_78, no_default=var_85)



# Parsed testcases at query #29
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
    var_33 = []
    var_34 = module_0.get_in(var_32, var_25, var_33)
    var_35 = {var_1: var_20}
    var_36 = [var_3, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_3, var_1]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_0, var_3]
    var_41 = module_0.get_in(var_40, var_37)
    var_42 = [var_0, var_3, var_11]
    var_43 = module_0.get_in(var_42, var_37)
    assert var_43 is None
    var_44 = {var_0: var_3}
    var_45 = 'x'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_44, no_default=var_47)
    var_49 = 0
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_44, no_default=var_51)
    var_53 = {var_49: var_52}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {var_49: var_52}
    var_57 = [var_49]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 == 1
    var_59 = [var_50]
    var_60 = module_0.get_in(var_59, var_56)
    assert var_60 is None



# Parsed testcases at query #30
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
    var_43 = (var_38, var_9, var_10)
    var_44 = {var_36: var_43}
    var_45 = [var_44]
    var_46 = {var_35: var_45}
    var_47 = 0
    var_48 = [var_35, var_47, var_36, var_38]
    var_49 = module_0.get_in(var_48, var_46)
    assert var_49 == 2
    var_50 = {var_35: var_38}
    var_51 = [var_36]
    var_52 = None
    var_53 = module_0.get_in(var_51, var_50, var_52)
    assert var_53 is None



# Parsed testcases at query #31
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
    var_29 = 10
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
    var_57 = 0
    var_58 = [var_43, var_46, var_57]
    var_59 = [var_42]
    var_60 = 'total'
    var_61 = [var_43, var_60]
    var_62 = [var_43, var_60]



# Parsed testcases at query #32
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
    var_15 = 'x'
    var_16 = [var_15]
    var_17 = 'default'
    var_18 = module_0.get_in(var_16, var_12, var_17)
    assert var_18 == 'default'
    var_19 = 5
    assert var_19 == 'Apple'
    var_20 = [var_0, var_19]
    var_21 = module_0.get_in(var_20, var_12, var_17)
    assert var_21 == 'default'
    assert var_21 == 'Alice'
    var_22 = 'x'
    var_23 = [var_22]
    var_24 = True
    var_25 = module_0.get_in(var_23, var_12, no_default=var_24)
    var_26 = 'a'
    var_27 = 5
    var_28 = [var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_12, no_default=var_29)
    var_31 = 'name'
    var_32 = 'purchase'
    var_33 = 'credit card'
    var_34 = 'Alice'
    var_35 = 'items'
    var_36 = 'costs'
    var_37 = 'Apple'
    var_38 = 'Orange'
    var_39 = [var_37, var_38]
    var_40 = 0.5
    var_41 = 1.25
    var_42 = [var_40, var_41]
    var_43 = {var_35: var_39, var_36: var_42}
    var_44 = '5555-1234-1234-1234'
    var_45 = {var_31: var_34, var_32: var_43, var_33: var_44}
    var_46 = 0
    var_47 = [var_32, var_35, var_46]
    var_48 = [var_31]
    var_49 = 'total'
    var_50 = [var_32, var_49]
    var_51 = [var_32, var_49]
    var_52 = {var_32: var_34}
    var_53 = {var_32: var_39}
    var_54 = [var_52, var_53]
    var_55 = {var_31: var_54}
    var_56 = 0
    var_57 = [var_31, var_56, var_32]
    var_58 = module_0.get_in(var_57, var_55)
    assert var_58 == 1
    var_59 = [var_31, var_34, var_32]
    var_60 = module_0.get_in(var_59, var_55)
    assert var_60 == 2
    var_61 = [var_31, var_39, var_32]
    var_62 = 'not_found'
    var_63 = module_0.get_in(var_61, var_55, var_62)
    assert var_63 == 'not_found'
    var_64 = []
    var_65 = module_0.get_in(var_64, var_12)



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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
    var_18 = 3
    var_19 = 4
    var_20 = 5
    var_21 = 6
    var_22 = [var_20, var_21]
    var_23 = [var_18, var_19, var_22]
    var_24 = [var_3, var_17, var_23]
    var_25 = [var_17, var_17, var_3]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 6
    var_27 = 0
    var_28 = [var_27]
    var_29 = module_0.get_in(var_28, var_24)
    assert var_29 == 1
    var_30 = [var_20]
    var_31 = module_0.get_in(var_30, var_24)
    assert var_31 is None
    var_32 = [var_20]
    var_33 = module_0.get_in(var_32, var_24, var_15)
    assert var_33 == 'default'
    var_34 = {var_1: var_17}
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
    var_43 = {var_0: var_3}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_47, var_17, var_18]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_52}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = None
    var_57 = {var_50: var_56}
    var_58 = {var_49: var_57}
    var_59 = [var_49, var_50]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_49, var_50]
    var_62 = module_0.get_in(var_61, var_58, var_15)
    assert var_62 is None
    var_63 = [var_49, var_51]
    var_64 = module_0.get_in(var_63, var_58, var_15)
    assert var_64 == 'default'



# Parsed testcases at query #35
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
    var_23 = 5
    var_24 = 6
    var_25 = [var_23, var_24]
    var_26 = 7
    var_27 = 8
    var_28 = [var_26, var_27]
    var_29 = [var_25, var_28]
    var_30 = [var_22, var_29]
    var_31 = [var_15, var_3, var_3]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 4
    var_33 = [var_3, var_15]
    var_34 = module_0.get_in(var_33, var_30)
    var_35 = [var_17]
    var_36 = module_0.get_in(var_35, var_30)
    assert var_36 is None
    var_37 = [var_17]
    var_38 = module_0.get_in(var_37, var_30, var_15)
    assert var_38 == 0
    var_39 = {var_1: var_17}
    var_40 = [var_3, var_39]
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_3, var_1]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 2
    var_44 = [var_0, var_15]
    var_45 = module_0.get_in(var_44, var_41)
    assert var_45 == 1
    var_46 = [var_11]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 is None
    var_48 = {var_1: var_3}
    var_49 = {var_0: var_48}
    var_50 = 'x'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_0.get_in(var_51, var_49, no_default=var_52)
    var_54 = 'a'
    var_55 = 'x'
    var_56 = [var_54, var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_49, no_default=var_57)
    var_59 = 0
    var_60 = [var_59]
    var_61 = 1
    var_62 = 2
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_60, var_63, no_default=var_64)
    var_66 = {var_59: var_62}
    var_67 = []
    var_68 = module_0.get_in(var_67, var_66)
    var_69 = {var_60: var_62}
    var_70 = {var_59: var_69}
    var_71 = 'y'
    var_72 = [var_11, var_71]
    var_73 = 'default'
    var_74 = module_0.get_in(var_72, var_70, var_73)
    assert var_74 == 'default'
    var_75 = [var_59, var_11]
    var_76 = module_0.get_in(var_75, var_70, var_73)
    assert var_76 == 'default'



# Parsed testcases at query #36
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
    var_14 = [var_0, var_11]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_1, var_11]
    var_17 = module_0.get_in(var_16, var_6)
    assert var_17 is None
    var_18 = [var_11]
    var_19 = 'default'
    var_20 = module_0.get_in(var_18, var_6, var_19)
    assert var_20 == 'default'
    var_21 = [var_0, var_11]
    var_22 = module_0.get_in(var_21, var_6, var_19)
    assert var_22 == 'default'
    var_23 = [var_0, var_1, var_11]
    var_24 = module_0.get_in(var_23, var_6, var_19)
    assert var_24 == 'default'
    var_25 = 'x'
    var_26 = [var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_6, no_default=var_27)
    var_29 = 'a'
    var_30 = 'x'
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_6, no_default=var_32)
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 'x'
    var_37 = [var_34, var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_6, no_default=var_38)
    var_40 = 2
    var_41 = 3
    var_42 = [var_37, var_40, var_41]
    var_43 = 4
    var_44 = 5
    var_45 = 6
    var_46 = [var_43, var_44, var_45]
    var_47 = [var_42, var_46]
    var_48 = 7
    var_49 = 8
    var_50 = 9
    var_51 = [var_48, var_49, var_50]
    var_52 = 10
    var_53 = 11
    var_54 = 12
    var_55 = [var_52, var_53, var_54]
    var_56 = [var_51, var_55]
    var_57 = [var_47, var_56]
    var_58 = 0
    var_59 = [var_58, var_58, var_37]
    var_60 = module_0.get_in(var_59, var_57)
    assert var_60 == 2
    var_61 = [var_37, var_37, var_40]
    var_62 = module_0.get_in(var_61, var_57)
    assert var_62 == 12
    var_63 = [var_58, var_58, var_44]
    var_64 = module_0.get_in(var_63, var_57)
    assert var_64 is None
    var_65 = [var_40, var_58, var_58]
    var_66 = module_0.get_in(var_65, var_57)
    assert var_66 is None
    var_67 = {var_35: var_41}
    var_68 = [var_37, var_40, var_67]
    var_69 = {var_34: var_68}
    var_70 = [var_34, var_40, var_35]
    var_71 = module_0.get_in(var_70, var_69)
    assert var_71 == 3
    var_72 = [var_34, var_40, var_11]
    var_73 = module_0.get_in(var_72, var_69)
    assert var_73 is None
    var_74 = [var_34, var_44]
    var_75 = module_0.get_in(var_74, var_69)
    assert var_75 is None
    var_76 = []
    var_77 = module_0.get_in(var_76, var_6)
    var_78 = []
    var_79 = module_0.get_in(var_78, var_57)
    var_80 = []
    var_81 = module_0.get_in(var_80, var_69)
    var_82 = [var_11]
    var_83 = 'string'
    var_84 = module_0.get_in(var_82, var_83)
    assert var_84 is None
    var_85 = [var_11]
    var_86 = 123
    var_87 = module_0.get_in(var_85, var_86)
    assert var_87 is None



# Parsed testcases at query #37
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
    var_20 = [var_13]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 'default'
    var_23 = [var_0, var_13]
    var_24 = module_0.get_in(var_23, var_6, var_21)
    assert var_24 == 'default'
    var_25 = [var_0, var_1, var_13]
    var_26 = module_0.get_in(var_25, var_6, var_21)
    assert var_26 == 'default'
    var_27 = 'x'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'a'
    var_32 = 'x'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_6, no_default=var_34)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = 'x'
    var_39 = [var_36, var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_6, no_default=var_40)
    var_42 = 2
    var_43 = 3
    var_44 = [var_39, var_42, var_43]
    var_45 = [var_44]
    var_46 = [var_45]
    var_47 = 0
    var_48 = [var_47, var_47, var_39]
    var_49 = module_0.get_in(var_48, var_46)
    assert var_49 == 2
    var_50 = [var_47]
    var_51 = module_0.get_in(var_50, var_46)
    var_52 = [var_47, var_47]
    var_53 = module_0.get_in(var_52, var_46)
    var_54 = [var_39]
    var_55 = module_0.get_in(var_54, var_46)
    assert var_55 is None
    var_56 = [var_47, var_39]
    var_57 = module_0.get_in(var_56, var_46)
    assert var_57 is None
    var_58 = [var_47, var_47, var_43]
    var_59 = module_0.get_in(var_58, var_46)
    assert var_59 is None
    var_60 = [var_39]
    var_61 = module_0.get_in(var_60, var_46, var_21)
    assert var_61 == 'default'
    var_62 = [var_47, var_39]
    var_63 = module_0.get_in(var_62, var_46, var_21)
    assert var_63 == 'default'
    var_64 = [var_47, var_47, var_43]
    var_65 = module_0.get_in(var_64, var_46, var_21)
    assert var_65 == 'default'
    var_66 = 1
    var_67 = [var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_46, no_default=var_68)
    var_70 = 0
    var_71 = 1
    var_72 = [var_70, var_71]
    var_73 = True
    var_74 = module_0.get_in(var_72, var_46, no_default=var_73)
    var_75 = 0
    var_76 = 3
    var_77 = [var_75, var_75, var_76]
    var_78 = True
    var_79 = module_0.get_in(var_77, var_46, no_default=var_78)
    var_80 = {var_76: var_43}
    var_81 = [var_78, var_42, var_80]
    var_82 = {var_75: var_81}
    var_83 = [var_75, var_42, var_76]
    var_84 = module_0.get_in(var_83, var_82)
    assert var_84 == 3
    var_85 = [var_75, var_47]
    var_86 = module_0.get_in(var_85, var_82)
    assert var_86 == 1
    var_87 = [var_75]
    var_88 = module_0.get_in(var_87, var_82)
    var_89 = [var_13]
    var_90 = module_0.get_in(var_89, var_82)
    assert var_90 is None
    var_91 = [var_75, var_43]
    var_92 = module_0.get_in(var_91, var_82)
    assert var_92 is None
    var_93 = [var_75, var_42, var_13]
    var_94 = module_0.get_in(var_93, var_82)
    assert var_94 is None
    var_95 = [var_13]
    var_96 = module_0.get_in(var_95, var_82, var_21)
    assert var_96 == 'default'
    var_97 = [var_75, var_43]
    var_98 = module_0.get_in(var_97, var_82, var_21)
    assert var_98 == 'default'
    var_99 = [var_75, var_42, var_13]
    var_100 = module_0.get_in(var_99, var_82, var_21)
    assert var_100 == 'default'
    var_101 = 'x'
    var_102 = [var_101]
    var_103 = True
    var_104 = module_0.get_in(var_102, var_82, no_default=var_103)
    var_105 = 'a'
    var_106 = 3
    var_107 = [var_105, var_106]
    var_108 = True
    var_109 = module_0.get_in(var_107, var_82, no_default=var_108)
    var_110 = 'a'
    var_111 = 2
    var_112 = 'x'
    var_113 = [var_110, var_111, var_112]
    var_114 = True
    var_115 = module_0.get_in(var_113, var_82, no_default=var_114)



# Parsed testcases at query #38
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



# Parsed testcases at query #39
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
    var_13 = 'z'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_11, var_12, var_13]
    var_17 = 42
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 42
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
    var_33 = 0
    var_34 = [var_33, var_3, var_3]
    var_35 = module_0.get_in(var_34, var_32)
    assert var_35 == 4
    var_36 = [var_3, var_33, var_33]
    var_37 = module_0.get_in(var_36, var_32)
    assert var_37 == 5
    var_38 = [var_19, var_33, var_33]
    var_39 = module_0.get_in(var_38, var_32)
    assert var_39 is None
    var_40 = [var_19, var_33, var_33]
    var_41 = module_0.get_in(var_40, var_32, var_33)
    assert var_41 == 0
    var_42 = {var_1: var_19}
    var_43 = [var_3, var_42]
    var_44 = {var_0: var_43}
    var_45 = [var_0, var_3, var_1]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 2
    var_47 = [var_0, var_33]
    var_48 = module_0.get_in(var_47, var_44)
    assert var_48 == 1
    var_49 = [var_0, var_3, var_2]
    var_50 = module_0.get_in(var_49, var_44)
    assert var_50 is None
    var_51 = [var_11]
    var_52 = {}
    var_53 = 'default'
    var_54 = module_0.get_in(var_51, var_52, var_53)
    assert var_54 == 'default'
    var_55 = [var_11, var_12]
    var_56 = {}
    var_57 = module_0.get_in(var_55, var_56, var_33)
    assert var_57 == 0
    var_58 = 'x'
    var_59 = [var_58]
    var_60 = {}
    var_61 = True
    var_62 = module_0.get_in(var_59, var_60, no_default=var_61)
    var_63 = 0
    var_64 = [var_63]
    var_65 = []
    var_66 = True
    var_67 = module_0.get_in(var_64, var_65, no_default=var_66)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_6)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_32)
    var_72 = 'd'
    var_73 = [var_63, var_64, var_72]
    var_74 = module_0.get_in(var_73, var_6)
    assert var_74 is None
    var_75 = [var_33, var_33, var_33, var_33]
    var_76 = module_0.get_in(var_75, var_32)
    assert var_76 is None
    var_77 = [var_63, var_64, var_65, var_72]
    var_78 = 'string'
    var_79 = {var_65: var_78}
    var_80 = {var_64: var_79}
    var_81 = {var_63: var_80}
    var_82 = module_0.get_in(var_77, var_81)
    assert var_82 is None



# Parsed testcases at query #40
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
    var_50 = {}
    var_51 = [var_43, var_44, var_45]
    var_52 = 'default'
    var_53 = module_0.get_in(var_51, var_50, var_52)
    assert var_53 == 'default'



# Parsed testcases at query #41
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
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 0
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = {var_23: var_27}
    var_29 = 5
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 0
    var_34 = {var_23: var_26}
    var_35 = [var_23, var_24]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_23, var_24]
    var_38 = module_0.get_in(var_37, var_34, var_20)
    assert var_38 == 0
    var_39 = {var_23: var_26}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
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
    var_57 = [var_43, var_46, var_20]
    var_58 = [var_42]
    var_59 = 'total'
    var_60 = [var_43, var_59]
    var_61 = [var_43, var_59]



# Parsed testcases at query #42
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
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = (var_38, var_9, var_10)
    var_44 = {var_36: var_43}
    var_45 = [var_44]
    var_46 = {var_35: var_45}
    var_47 = 0
    var_48 = [var_35, var_47, var_36, var_38]
    var_49 = module_0.get_in(var_48, var_46)
    assert var_49 == 2
    var_50 = 'a'
    var_51 = 'b'
    var_52 = 1
    var_53 = {var_51: var_52}
    var_54 = {var_50: var_53}
    var_55 = [var_50, var_51]
    var_56 = module_0.get_in(var_55, var_46)
    assert var_56 == 1



# Parsed testcases at query #43
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
    var_20 = [var_13]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 0
    var_23 = [var_0, var_13]
    var_24 = module_0.get_in(var_23, var_6, var_21)
    assert var_24 == 0
    var_25 = [var_0, var_1, var_13]
    var_26 = module_0.get_in(var_25, var_6, var_21)
    assert var_26 == 0
    var_27 = 'x'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'a'
    var_32 = 'x'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_6, no_default=var_34)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = 'x'
    var_39 = [var_36, var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_6, no_default=var_40)
    var_42 = 2
    var_43 = 3
    var_44 = 4
    var_45 = 5
    var_46 = 6
    var_47 = [var_45, var_46]
    var_48 = [var_43, var_44, var_47]
    var_49 = [var_39, var_42, var_48]
    var_50 = [var_21]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 1
    var_52 = [var_42]
    var_53 = module_0.get_in(var_52, var_49)
    var_54 = [var_42, var_42]
    var_55 = module_0.get_in(var_54, var_49)
    var_56 = [var_42, var_42, var_21]
    var_57 = module_0.get_in(var_56, var_49)
    assert var_57 == 5
    var_58 = [var_42, var_42, var_43]
    var_59 = module_0.get_in(var_58, var_49)
    assert var_59 is None
    var_60 = [var_42, var_42, var_43]
    var_61 = module_0.get_in(var_60, var_49, var_21)
    assert var_61 == 0
    var_62 = {var_37: var_42}
    var_63 = [var_39, var_62]
    var_64 = {var_36: var_63}
    var_65 = [var_36, var_39, var_37]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 == 2
    var_67 = [var_36, var_39, var_38]
    var_68 = module_0.get_in(var_67, var_64)
    assert var_68 is None
    var_69 = [var_36, var_39, var_38]
    var_70 = module_0.get_in(var_69, var_64, var_21)
    assert var_70 == 0
    var_71 = []
    var_72 = module_0.get_in(var_71, var_6)
    var_73 = []
    var_74 = module_0.get_in(var_73, var_49)
    var_75 = []
    var_76 = module_0.get_in(var_75, var_64)
    var_77 = 42
    var_78 = [var_36]
    var_79 = module_0.get_in(var_78, var_77)
    assert var_79 is None
    var_80 = [var_36]
    var_81 = module_0.get_in(var_80, var_77, var_21)
    assert var_81 == 0
    var_82 = 'a'
    var_83 = [var_82]
    var_84 = True
    var_85 = module_0.get_in(var_83, var_77, no_default=var_84)



# Parsed testcases at query #44
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
    var_49 = [var_45]
    var_50 = module_0.get_in(var_49, var_48, var_23)
    assert var_50 == 0
    var_51 = [var_45]
    var_52 = None
    var_53 = module_0.get_in(var_51, var_48, var_52)
    assert var_53 is None
    var_54 = {var_44: var_47}
    var_55 = []
    var_56 = module_0.get_in(var_55, var_54)



# Parsed testcases at query #45
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
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 'default'
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = [var_29, var_3]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 == 2
    var_32 = [var_3, var_29]
    var_33 = module_0.get_in(var_32, var_28)
    assert var_33 == 3
    var_34 = [var_23]
    var_35 = module_0.get_in(var_34, var_28)
    assert var_35 is None
    var_36 = [var_23]
    var_37 = module_0.get_in(var_36, var_28, var_17)
    assert var_37 == 'default'
    var_38 = [var_29, var_23]
    var_39 = module_0.get_in(var_38, var_28)
    assert var_39 is None
    var_40 = [var_29, var_23]
    var_41 = module_0.get_in(var_40, var_28, var_17)
    assert var_41 == 'default'
    var_42 = {var_1: var_23}
    var_43 = [var_3, var_42]
    var_44 = {var_0: var_43}
    var_45 = [var_0, var_3, var_1]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 2
    var_47 = [var_0, var_3]
    var_48 = module_0.get_in(var_47, var_44)
    var_49 = [var_0, var_29]
    var_50 = module_0.get_in(var_49, var_44)
    assert var_50 == 1
    var_51 = [var_0, var_23]
    var_52 = module_0.get_in(var_51, var_44)
    assert var_52 is None
    var_53 = [var_0, var_23]
    var_54 = module_0.get_in(var_53, var_44, var_17)
    assert var_54 == 'default'
    var_55 = [var_13]
    var_56 = module_0.get_in(var_55, var_44)
    assert var_56 is None
    var_57 = [var_13]
    var_58 = module_0.get_in(var_57, var_44, var_17)
    assert var_58 == 'default'
    var_59 = {var_1: var_3}
    var_60 = {var_0: var_59}
    var_61 = 'x'
    var_62 = [var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_60, no_default=var_63)
    var_65 = 'a'
    var_66 = 'x'
    var_67 = [var_65, var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_60, no_default=var_68)
    var_70 = 0
    var_71 = [var_70]
    var_72 = True
    var_73 = module_0.get_in(var_71, var_60, no_default=var_72)
    var_74 = {var_70: var_73}
    var_75 = []
    var_76 = module_0.get_in(var_75, var_74)
    var_77 = []
    var_78 = module_0.get_in(var_77, var_74, var_17)
    var_79 = {var_70: var_73}
    var_80 = [var_70]
    var_81 = module_0.get_in(var_80, var_79)
    assert var_81 == 1
    var_82 = [var_71]
    var_83 = module_0.get_in(var_82, var_79)
    assert var_83 is None
    var_84 = [var_71]
    var_85 = module_0.get_in(var_84, var_79, var_17)
    assert var_85 == 'default'



# Parsed testcases at query #46
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
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 10
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 42
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
    var_35 = 10
    var_36 = [var_34, var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_33, no_default=var_37)
    var_39 = {var_34: var_37}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = [var_37, var_9, var_10]
    var_43 = {var_35: var_42}
    var_44 = 4
    var_45 = {var_36: var_44}
    var_46 = [var_43, var_45]
    var_47 = {var_34: var_46}
    var_48 = 0
    var_49 = [var_34, var_48, var_35, var_37]
    var_50 = module_0.get_in(var_49, var_47)
    assert var_50 == 2
    var_51 = {var_34: var_37}
    var_52 = [var_34, var_35]
    var_53 = module_0.get_in(var_52, var_51, var_18)
    assert var_53 == 42



# Parsed testcases at query #47
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
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = {var_37: var_9}
    var_44 = [var_38, var_43]
    var_45 = {var_36: var_44}
    var_46 = {var_35: var_45}
    var_47 = [var_35, var_36, var_38, var_37]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2



# Parsed testcases at query #48
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
    var_11 = [var_9, var_10]
    var_12 = 4
    var_13 = [var_3, var_11, var_12]
    var_14 = 0
    var_15 = [var_3, var_14]
    var_16 = module_0.get_in(var_15, var_13)
    assert var_16 == 2
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = 'default'
    var_21 = module_0.get_in(var_19, var_13, var_20)
    assert var_21 == 'default'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_13, no_default=var_25)
    var_27 = {var_23: var_9}
    var_28 = [var_25, var_27]
    var_29 = {var_22: var_28}
    var_30 = [var_22, var_25, var_23]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 2
    var_32 = []
    var_33 = module_0.get_in(var_32, var_29)
    var_34 = {var_22: var_25}
    var_35 = [var_22, var_23]
    var_36 = module_0.get_in(var_35, var_34, var_20)
    assert var_36 == 'default'
    var_37 = [var_25, var_9, var_10]
    var_38 = 5
    var_39 = [var_38]
    var_40 = module_0.get_in(var_39, var_37, var_20)
    assert var_40 == 'default'
    var_41 = {var_22: var_25}
    var_42 = [var_23]
    var_43 = module_0.get_in(var_42, var_41, var_20)
    assert var_43 == 'default'



# Parsed testcases at query #49
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
    var_51 = [var_11]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_11]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0



# Parsed testcases at query #50
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
    var_48 = (var_43, var_9, var_10)
    var_49 = {var_41: var_48}
    var_50 = [var_49]
    var_51 = {var_40: var_50}
    var_52 = 0
    var_53 = [var_40, var_52, var_41, var_43]
    var_54 = module_0.get_in(var_53, var_51)
    assert var_54 == 2



# Parsed testcases at query #51
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
    var_39 = [var_0, var_23]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 == 1
    var_41 = [var_11]
    var_42 = module_0.get_in(var_41, var_34)
    assert var_42 is None
    var_43 = {var_0: var_3}
    var_44 = 'x'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = {var_44: var_47}
    var_49 = []
    var_50 = module_0.get_in(var_49, var_48)
    var_51 = {var_44: var_47}
    var_52 = [var_44, var_45]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_44, var_45]
    var_55 = module_0.get_in(var_54, var_51, var_15)
    assert var_55 == 'default'
    var_56 = [var_47, var_17, var_19]
    var_57 = 5
    var_58 = [var_57]
    var_59 = module_0.get_in(var_58, var_56)
    assert var_59 is None
    var_60 = [var_57]
    var_61 = module_0.get_in(var_60, var_56, var_15)
    assert var_61 == 'default'



# Parsed testcases at query #52
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
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_3, var_24, var_27]
    var_29 = [var_22]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 1
    var_31 = [var_24, var_22]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 3
    var_33 = [var_24, var_3]
    var_34 = module_0.get_in(var_33, var_28)
    assert var_34 == 4
    var_35 = [var_25]
    var_36 = module_0.get_in(var_35, var_28)
    assert var_36 is None
    var_37 = [var_25]
    var_38 = module_0.get_in(var_37, var_28, var_17)
    assert var_38 == 'default'
    var_39 = [var_24, var_24]
    var_40 = module_0.get_in(var_39, var_28)
    assert var_40 is None
    var_41 = [var_24, var_24]
    var_42 = module_0.get_in(var_41, var_28, var_22)
    assert var_42 == 0
    var_43 = {var_1: var_24}
    var_44 = [var_3, var_43]
    var_45 = {var_0: var_44}
    var_46 = [var_0, var_3, var_1]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 2
    var_48 = [var_0, var_3]
    var_49 = module_0.get_in(var_48, var_45)
    var_50 = [var_0, var_22]
    var_51 = module_0.get_in(var_50, var_45)
    assert var_51 == 1
    var_52 = [var_0, var_24]
    var_53 = module_0.get_in(var_52, var_45)
    assert var_53 is None
    var_54 = [var_0, var_24]
    var_55 = module_0.get_in(var_54, var_45, var_17)
    assert var_55 == 'default'
    var_56 = {var_0: var_3}
    var_57 = 'b'
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_56, no_default=var_59)
    var_61 = 0
    var_62 = [var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_56, no_default=var_63)
    var_65 = [var_64, var_24, var_25]
    var_66 = 5
    var_67 = [var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_65, no_default=var_68)
    var_70 = [var_13]
    var_71 = {}
    var_72 = module_0.get_in(var_70, var_71, var_17)
    assert var_72 == 'default'
    var_73 = [var_22]
    var_74 = []
    var_75 = module_0.get_in(var_73, var_74, var_17)
    assert var_75 == 'default'
    var_76 = 'y'
    var_77 = [var_13, var_76]
    var_78 = {}
    var_79 = module_0.get_in(var_77, var_78, var_17)
    assert var_79 == 'default'



# Parsed testcases at query #53
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
    var_39 = [var_0, var_18]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 == 1
    var_41 = [var_0, var_3]
    var_42 = module_0.get_in(var_41, var_36)
    var_43 = [var_0, var_20]
    var_44 = module_0.get_in(var_43, var_36)
    assert var_44 is None
    var_45 = {var_0: var_3}
    var_46 = 'b'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_45, no_default=var_48)
    var_50 = {var_46: var_49}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = {var_47: var_49}
    var_54 = {var_46: var_53}
    var_55 = 'a'
    var_56 = 'c'
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_54, no_default=var_58)
    var_60 = {var_55: var_58}
    var_61 = [var_55, var_56]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 is None
    var_63 = [var_55, var_56]
    var_64 = module_0.get_in(var_63, var_60, var_15)
    assert var_64 == 'default'



# Parsed testcases at query #54
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
    var_17 = [var_0, var_11]
    var_18 = module_0.get_in(var_17, var_6)
    assert var_18 is None
    var_19 = 2
    var_20 = [var_3, var_19]
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = 0
    var_26 = [var_25, var_3]
    var_27 = module_0.get_in(var_26, var_24)
    assert var_27 == 2
    var_28 = [var_3, var_25]
    var_29 = module_0.get_in(var_28, var_24)
    assert var_29 == 3
    var_30 = [var_19]
    var_31 = module_0.get_in(var_30, var_24)
    assert var_31 is None
    var_32 = {var_1: var_19}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3, var_11]
    var_38 = module_0.get_in(var_37, var_34)
    assert var_38 is None
    var_39 = 'x'
    var_40 = [var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_34, no_default=var_41)
    var_43 = 0
    var_44 = [var_43]
    var_45 = []
    var_46 = True
    var_47 = module_0.get_in(var_44, var_45, no_default=var_46)
    var_48 = [var_11]
    var_49 = {}
    var_50 = 42
    var_51 = module_0.get_in(var_48, var_49, var_50)
    assert var_51 == 42
    var_52 = [var_25]
    var_53 = []
    var_54 = module_0.get_in(var_52, var_53, var_50)
    assert var_54 == 42
    var_55 = []
    var_56 = module_0.get_in(var_55, var_34)
    var_57 = {var_43: var_46}
    var_58 = [var_43, var_44]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_43, var_44]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 == 'default'



# Parsed testcases at query #55
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
    assert var_7 == 2
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_3, var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_0, var_3]
    var_14 = module_0.get_in(var_13, var_12)
    assert var_14 == 2
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = 'default'
    var_19 = module_0.get_in(var_17, var_12, var_18)
    assert var_19 == 'default'
    var_20 = [var_15, var_16]
    var_21 = module_0.get_in(var_20, var_12)
    assert var_21 is None
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_12, no_default=var_25)
    var_27 = [var_25, var_9]
    var_28 = 4
    var_29 = [var_10, var_28]
    var_30 = [var_27, var_29]
    var_31 = {var_22: var_30}
    var_32 = 0
    var_33 = [var_22, var_25, var_32]
    var_34 = module_0.get_in(var_33, var_31)
    assert var_34 == 3
    var_35 = 'string'
    var_36 = {var_22: var_35}
    var_37 = [var_22, var_23]
    var_38 = 'error'
    var_39 = module_0.get_in(var_37, var_36, var_38)
    assert var_39 == 'error'
    var_40 = 'a'
    var_41 = 'b'
    var_42 = [var_40, var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_36, no_default=var_43)
    var_45 = []
    var_46 = module_0.get_in(var_45, var_36)
    var_47 = 'a'
    var_48 = 'b'
    var_49 = 2
    var_50 = {var_48: var_49}
    var_51 = {var_47: var_50}
    var_52 = [var_47, var_48]



# Parsed testcases at query #56
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
    var_20 = {var_1: var_3}
    assert var_20 == 'Apple'
    var_21 = {var_0: var_20}
    var_22 = [var_0, var_2]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 is None
    assert var_23 == 'Alice'
    var_24 = {var_1: var_3}
    var_25 = {var_0: var_24}
    var_26 = 'a'
    var_27 = 'c'
    var_28 = [var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_25, no_default=var_29)
    var_31 = [var_29, var_9]
    var_32 = 4
    assert var_32 is None
    var_33 = [var_10, var_32]
    var_34 = [var_31, var_33]
    assert var_34 == 0
    var_35 = {var_26: var_34}
    var_36 = 0
    var_37 = [var_26, var_29, var_36]
    var_38 = module_0.get_in(var_37, var_35)
    assert var_38 == 3
    var_39 = [var_29, var_9, var_10]
    var_40 = {var_26: var_39}
    var_41 = 'a'
    var_42 = 10
    var_43 = [var_41, var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_40, no_default=var_44)
    var_46 = {var_41: var_44}
    var_47 = 'a'
    var_48 = 'b'
    var_49 = [var_47, var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_46, no_default=var_50)
    var_52 = {var_47: var_50}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = 'name'
    var_56 = 'purchase'
    var_57 = 'credit card'
    var_58 = 'Alice'
    var_59 = 'items'
    var_60 = 'costs'
    var_61 = 'Apple'
    var_62 = 'Orange'
    var_63 = [var_61, var_62]
    var_64 = 0.5
    var_65 = 1.25
    var_66 = [var_64, var_65]
    var_67 = {var_59: var_63, var_60: var_66}
    var_68 = '5555-1234-1234-1234'
    var_69 = {var_55: var_58, var_56: var_67, var_57: var_68}
    var_70 = 0
    var_71 = [var_56, var_59, var_70]
    var_72 = [var_55]
    var_73 = 'total'
    var_74 = [var_56, var_73]
    var_75 = [var_56, var_73]



# Parsed testcases at query #57
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
    var_15 = 'not found'
    var_16 = module_0.get_in(var_14, var_6, var_15)
    assert var_16 == 'not found'
    var_17 = [var_0, var_1, var_11]
    var_18 = 0
    var_19 = module_0.get_in(var_17, var_6, var_18)
    assert var_19 == 0
    var_20 = 2
    var_21 = 3
    var_22 = 4
    var_23 = [var_21, var_22]
    var_24 = [var_20, var_23]
    var_25 = [var_3, var_24]
    var_26 = [var_3, var_3, var_18]
    var_27 = module_0.get_in(var_26, var_25)
    assert var_27 == 3
    var_28 = [var_18]
    var_29 = module_0.get_in(var_28, var_25)
    assert var_29 == 1
    var_30 = [var_3, var_3]
    var_31 = module_0.get_in(var_30, var_25)
    var_32 = 5
    var_33 = [var_32]
    var_34 = module_0.get_in(var_33, var_25)
    assert var_34 is None
    var_35 = [var_3, var_3, var_32]
    var_36 = -1
    var_37 = module_0.get_in(var_35, var_25, var_36)
    assert var_37 == -1
    var_38 = {var_1: var_20}
    var_39 = [var_3, var_38]
    var_40 = {var_0: var_39}
    var_41 = [var_0, var_3, var_1]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 == 2
    var_43 = [var_0, var_18]
    var_44 = module_0.get_in(var_43, var_40)
    assert var_44 == 1
    var_45 = [var_0, var_3]
    var_46 = module_0.get_in(var_45, var_40)
    var_47 = [var_0, var_32]
    var_48 = module_0.get_in(var_47, var_40)
    assert var_48 is None
    var_49 = [var_0, var_3, var_11]
    var_50 = 'missing'
    var_51 = module_0.get_in(var_49, var_40, var_50)
    assert var_51 == 'missing'
    var_52 = {var_0: var_3}
    var_53 = 'x'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_52, no_default=var_55)
    var_57 = 0
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_52, no_default=var_59)
    var_61 = []
    var_62 = module_0.get_in(var_61, var_52)
    var_63 = []
    var_64 = None
    var_65 = module_0.get_in(var_63, var_64)
    assert var_65 is None
    var_66 = []
    var_67 = 'default'
    var_68 = module_0.get_in(var_66, var_64, var_67)
    assert var_68 is None
    var_69 = {var_57: var_60}
    var_70 = [var_57, var_58]
    var_71 = module_0.get_in(var_70, var_69)
    assert var_71 is None
    var_72 = [var_57, var_58]
    var_73 = module_0.get_in(var_72, var_69, var_18)
    assert var_73 == 0



# Parsed testcases at query #58
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
    var_22 = 5
    var_23 = [var_0, var_22]
    var_24 = 'out of range'
    var_25 = module_0.get_in(var_23, var_21, var_24)
    assert var_25 == 'out of range'
    var_26 = {var_0: var_3}
    var_27 = [var_0, var_1]
    var_28 = 'type error'
    var_29 = module_0.get_in(var_27, var_26, var_28)
    assert var_29 == 'type error'
    var_30 = {var_0: var_3}
    var_31 = 'b'
    var_32 = [var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_30, no_default=var_33)
    var_35 = [var_34, var_9, var_10]
    var_36 = 5
    var_37 = [var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_35, no_default=var_38)
    var_40 = 1
    var_41 = 'a'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_40, no_default=var_43)
    var_45 = {var_41: var_44}
    var_46 = [var_42]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 is None
    var_48 = {var_41: var_44}
    var_49 = []
    var_50 = module_0.get_in(var_49, var_48)
    var_51 = {var_42: var_44}
    var_52 = {var_43: var_9}
    var_53 = [var_51, var_52]
    var_54 = {var_41: var_53}
    var_55 = [var_41, var_44, var_43]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 == 2



# Parsed testcases at query #59
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
    var_43 = {var_36: var_38}
    var_44 = {var_37: var_9}
    var_45 = [var_43, var_44]
    var_46 = {var_35: var_45}
    var_47 = [var_35, var_38, var_37]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2
    var_49 = 'name'
    var_50 = 'purchase'
    var_51 = 'credit card'
    var_52 = 'Alice'
    var_53 = 'items'
    var_54 = 'costs'
    var_55 = 'Apple'
    var_56 = 'Orange'
    var_57 = [var_55, var_56]
    var_58 = 0.5
    var_59 = 1.25
    var_60 = [var_58, var_59]
    var_61 = {var_53: var_57, var_54: var_60}
    var_62 = '5555-1234-1234-1234'
    var_63 = {var_49: var_52, var_50: var_61, var_51: var_62}
    var_64 = 0
    var_65 = [var_50, var_53, var_64]
    var_66 = [var_49]
    var_67 = 'total'
    var_68 = [var_50, var_67]
    var_69 = [var_50, var_67]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_45 = [var_44]
    var_46 = {var_35: var_45}
    var_47 = 0
    var_48 = [var_35, var_47, var_36, var_38]
    var_49 = module_0.get_in(var_48, var_46)
    assert var_49 == 2
    var_50 = [var_35, var_36, var_38]
    var_51 = module_0.get_in(var_50, var_46)
    assert var_51 == 2



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
    var_42 = {var_38: var_41}
    var_43 = [var_39]
    var_44 = module_0.get_in(var_43, var_42, var_23)
    assert var_44 == 0
    var_45 = {var_38: var_41}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = 123
    var_49 = [var_38]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 is None
    var_51 = [var_38]
    var_52 = 'error'
    var_53 = module_0.get_in(var_51, var_48, var_52)
    assert var_53 == 'error'



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
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'x'
    var_14 = 'y'
    var_15 = 'z'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'not found'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'not found'
    var_19 = [var_0, var_13, var_14]
    var_20 = None
    var_21 = module_0.get_in(var_19, var_6, var_20)
    assert var_21 is None
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_6, no_default=var_25)
    var_27 = 2
    var_28 = 3
    var_29 = [var_25, var_27, var_28]
    var_30 = 4
    var_31 = 5
    var_32 = 6
    var_33 = [var_30, var_31, var_32]
    var_34 = [var_29, var_33]
    var_35 = 0
    var_36 = [var_35, var_25]
    var_37 = module_0.get_in(var_36, var_34)
    assert var_37 == 2
    var_38 = [var_25, var_27]
    var_39 = module_0.get_in(var_38, var_34)
    assert var_39 == 6
    var_40 = {var_23: var_28}
    var_41 = [var_25, var_27, var_40]
    var_42 = {var_22: var_41}
    var_43 = [var_22, var_27, var_23]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 3
    var_45 = []
    var_46 = module_0.get_in(var_45, var_6)
    var_47 = 'd'
    var_48 = [var_22, var_23, var_24, var_47]
    var_49 = 'error'
    var_50 = module_0.get_in(var_48, var_6, var_49)
    assert var_50 == 'error'



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
    var_23 = 5
    var_24 = 6
    var_25 = [var_23, var_24]
    var_26 = 7
    var_27 = 8
    var_28 = [var_26, var_27]
    var_29 = [var_25, var_28]
    var_30 = [var_22, var_29]
    var_31 = [var_15, var_3, var_3]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 4
    var_33 = [var_3]
    var_34 = module_0.get_in(var_33, var_30)
    var_35 = [var_17]
    var_36 = module_0.get_in(var_35, var_30)
    assert var_36 is None
    var_37 = [var_17]
    var_38 = module_0.get_in(var_37, var_30, var_15)
    assert var_38 == 0
    var_39 = {var_1: var_17}
    var_40 = [var_3, var_39]
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_3, var_1]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 2
    var_44 = [var_0, var_15]
    var_45 = module_0.get_in(var_44, var_41)
    assert var_45 == 1
    var_46 = [var_0, var_17]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 is None
    var_48 = {var_0: var_3}
    var_49 = 'b'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_52}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = None
    var_57 = {var_50: var_56}
    var_58 = {var_49: var_57}
    var_59 = [var_49, var_50]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_49, var_51]
    var_62 = module_0.get_in(var_61, var_58, var_15)
    assert var_62 == 0



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
    var_38 = [var_11]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_11]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
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
    var_18 = 3
    var_19 = [var_3, var_17, var_18]
    var_20 = 4
    var_21 = 5
    var_22 = 6
    var_23 = [var_20, var_21, var_22]
    var_24 = [var_19, var_23]
    var_25 = 7
    var_26 = 8
    var_27 = 9
    var_28 = [var_25, var_26, var_27]
    var_29 = 10
    var_30 = 11
    var_31 = 12
    var_32 = [var_29, var_30, var_31]
    var_33 = [var_28, var_32]
    var_34 = [var_24, var_33]
    var_35 = 0
    var_36 = [var_35, var_3, var_17]
    var_37 = module_0.get_in(var_36, var_34)
    assert var_37 == 6
    var_38 = [var_3, var_35]
    var_39 = module_0.get_in(var_38, var_34)
    var_40 = [var_17]
    var_41 = module_0.get_in(var_40, var_34)
    assert var_41 is None
    var_42 = [var_17]
    var_43 = module_0.get_in(var_42, var_34, var_15)
    assert var_43 == 'default'
    var_44 = {var_1: var_18}
    var_45 = [var_3, var_17, var_44]
    var_46 = {var_0: var_45}
    var_47 = [var_0, var_17, var_1]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 3
    var_49 = [var_0, var_3]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 2
    var_51 = [var_0, var_18]
    var_52 = module_0.get_in(var_51, var_46)
    assert var_52 is None
    var_53 = [var_0, var_18]
    var_54 = module_0.get_in(var_53, var_46, var_15)
    assert var_54 == 'default'
    var_55 = 'x'
    var_56 = [var_55]
    var_57 = True
    var_58 = module_0.get_in(var_56, var_6, no_default=var_57)
    var_59 = 0
    var_60 = 3
    var_61 = [var_59, var_60]
    var_62 = True
    var_63 = module_0.get_in(var_61, var_34, no_default=var_62)
    var_64 = []
    var_65 = module_0.get_in(var_64, var_6)
    var_66 = []
    var_67 = module_0.get_in(var_66, var_34)
    var_68 = 'y'
    var_69 = 'z'
    var_70 = [var_11, var_68, var_69]
    var_71 = module_0.get_in(var_70, var_6, var_15)
    assert var_71 == 'default'
    var_72 = [var_35, var_62, var_17, var_18]
    var_73 = module_0.get_in(var_72, var_34, var_15)
    assert var_73 == 'default'



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
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 'default'
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
    var_46 = [var_23, var_26, var_25]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 2
    var_48 = 'd'
    var_49 = [var_23, var_26, var_48]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 is None



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
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 'default'
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = [var_29, var_3]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 == 2
    var_32 = [var_3, var_29]
    var_33 = module_0.get_in(var_32, var_28)
    assert var_33 == 3
    var_34 = [var_29]
    var_35 = module_0.get_in(var_34, var_28)
    var_36 = [var_23]
    var_37 = module_0.get_in(var_36, var_28)
    assert var_37 is None
    var_38 = [var_23]
    var_39 = module_0.get_in(var_38, var_28, var_17)
    assert var_39 == 'default'
    var_40 = [var_29, var_23]
    var_41 = module_0.get_in(var_40, var_28)
    assert var_41 is None
    var_42 = [var_29, var_23]
    var_43 = module_0.get_in(var_42, var_28, var_17)
    assert var_43 == 'default'
    var_44 = {var_1: var_23}
    var_45 = [var_3, var_44]
    var_46 = {var_0: var_45}
    var_47 = [var_0, var_3, var_1]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2
    var_49 = [var_0, var_29]
    var_50 = module_0.get_in(var_49, var_46)
    assert var_50 == 1
    var_51 = [var_0, var_3]
    var_52 = module_0.get_in(var_51, var_46)
    var_53 = [var_0, var_23]
    var_54 = module_0.get_in(var_53, var_46)
    assert var_54 is None
    var_55 = [var_0, var_23]
    var_56 = module_0.get_in(var_55, var_46, var_17)
    assert var_56 == 'default'
    var_57 = [var_0, var_3, var_2]
    var_58 = module_0.get_in(var_57, var_46)
    assert var_58 is None
    var_59 = [var_0, var_3, var_2]
    var_60 = module_0.get_in(var_59, var_46, var_17)
    assert var_60 == 'default'
    var_61 = {var_1: var_3}
    var_62 = {var_0: var_61}
    var_63 = 'x'
    var_64 = [var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_62, no_default=var_65)
    var_67 = 'a'
    var_68 = 'x'
    var_69 = [var_67, var_68]
    var_70 = True
    var_71 = module_0.get_in(var_69, var_62, no_default=var_70)
    var_72 = 0
    var_73 = [var_72]
    var_74 = True
    var_75 = module_0.get_in(var_73, var_62, no_default=var_74)
    var_76 = {var_72: var_75}
    var_77 = []
    var_78 = module_0.get_in(var_77, var_76)
    var_79 = []
    var_80 = module_0.get_in(var_79, var_76, var_17)
    var_81 = {var_72: var_75}
    var_82 = [var_13]
    var_83 = None
    var_84 = module_0.get_in(var_82, var_81, var_83)
    assert var_84 is None



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
    var_18 = module_0.get_in(var_17, var_16)
    assert var_18 is None
    var_19 = [var_0, var_2]
    var_20 = 0
    var_21 = module_0.get_in(var_19, var_16, var_20)
    assert var_21 == 0
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
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 0
    var_34 = {var_23: var_26}
    var_35 = [var_23, var_24]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_23, var_24]
    var_38 = module_0.get_in(var_37, var_34, var_20)
    assert var_38 == 0
    var_39 = {var_23: var_26}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_24: var_26}
    var_43 = {var_25: var_9}
    var_44 = [var_42, var_43]
    var_45 = {var_23: var_44}
    var_46 = [var_23, var_26, var_25]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 2
    var_48 = [var_23, var_20, var_24]
    var_49 = module_0.get_in(var_48, var_45)
    assert var_49 == 1



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
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = module_0.get_in(var_17, var_12)
    assert var_18 is None
    var_19 = [var_15, var_16]
    var_20 = 'default'
    var_21 = module_0.get_in(var_19, var_12, var_20)
    assert var_21 == 'default'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_12, no_default=var_25)
    var_27 = 'a'
    var_28 = 10
    var_29 = [var_27, var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_12, no_default=var_30)
    var_32 = {var_27: var_30}
    var_33 = 'a'
    var_34 = 'b'
    var_35 = [var_33, var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_32, no_default=var_36)
    var_38 = []
    var_39 = module_0.get_in(var_38, var_32)
    var_40 = [var_36, var_9, var_10]
    var_41 = {var_34: var_40}
    var_42 = [var_41]
    var_43 = {var_33: var_42}
    var_44 = 0
    var_45 = [var_33, var_44, var_34, var_36]
    var_46 = module_0.get_in(var_45, var_43)
    assert var_46 == 2



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
    var_42 = [var_1]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_41, no_default=var_43)
    assert var_44 is None
    var_45 = 'b'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_41, no_default=var_47)
    var_49 = {var_45: var_43}
    var_50 = []
    var_51 = module_0.get_in(var_50, var_49)
    var_52 = {var_45: var_43}
    var_53 = [var_46]
    var_54 = module_0.get_in(var_53, var_52, var_15)
    assert var_54 == 'default'



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
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 10
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 42
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
    var_35 = 10
    var_36 = [var_34, var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_33, no_default=var_37)
    var_39 = {var_34: var_37}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = [var_37, var_9, var_10]
    var_43 = {var_35: var_42}
    var_44 = 4
    var_45 = {var_36: var_44}
    var_46 = [var_43, var_45]
    var_47 = {var_34: var_46}
    var_48 = 0
    var_49 = [var_34, var_48, var_35, var_37]
    var_50 = module_0.get_in(var_49, var_47)
    assert var_50 == 2
    var_51 = [var_34, var_37, var_36]
    var_52 = module_0.get_in(var_51, var_47)
    assert var_52 == 4
    var_53 = {var_34: var_37}
    var_54 = [var_34, var_35]
    var_55 = module_0.get_in(var_54, var_53, var_18)
    assert var_55 == 42



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
    var_23 = 5
    var_24 = 6
    var_25 = [var_23, var_24]
    var_26 = 7
    var_27 = 8
    var_28 = [var_26, var_27]
    var_29 = [var_25, var_28]
    var_30 = [var_22, var_29]
    var_31 = 0
    var_32 = [var_31, var_3, var_31]
    var_33 = module_0.get_in(var_32, var_30)
    assert var_33 == 3
    var_34 = [var_3]
    var_35 = module_0.get_in(var_34, var_30)
    var_36 = [var_17]
    var_37 = module_0.get_in(var_36, var_30)
    assert var_37 is None
    var_38 = [var_17]
    var_39 = module_0.get_in(var_38, var_30, var_15)
    assert var_39 == 'default'
    var_40 = {var_1: var_17}
    var_41 = [var_3, var_40]
    var_42 = {var_0: var_41}
    var_43 = [var_0, var_3, var_1]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 2
    var_45 = [var_0, var_31]
    var_46 = module_0.get_in(var_45, var_42)
    assert var_46 == 1
    var_47 = [var_0, var_3]
    var_48 = module_0.get_in(var_47, var_42)
    var_49 = [var_11]
    var_50 = module_0.get_in(var_49, var_42)
    assert var_50 is None
    var_51 = {var_0: var_3}
    var_52 = 'x'
    var_53 = [var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_51, no_default=var_54)
    var_56 = 0
    var_57 = [var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_51, no_default=var_58)
    var_60 = {var_56: var_59}
    var_61 = []
    var_62 = module_0.get_in(var_61, var_60)
    var_63 = {var_56: var_59}
    var_64 = [var_56, var_57]
    var_65 = module_0.get_in(var_64, var_63)
    assert var_65 is None
    var_66 = [var_56, var_57]
    var_67 = module_0.get_in(var_66, var_63, var_15)
    assert var_67 == 'default'



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
    var_29 = 5
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 'default'
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
    var_57 = 0
    var_58 = [var_43, var_46, var_57]
    var_59 = [var_42]
    var_60 = 'total'
    var_61 = [var_43, var_60]
    var_62 = [var_43, var_60]



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
    var_31 = {var_1: var_19}
    var_32 = [var_3, var_17, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_17, var_1]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 3
    var_36 = [var_0, var_17, var_11]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 is None
    var_38 = [var_0, var_17, var_11]
    var_39 = module_0.get_in(var_38, var_33, var_15)
    assert var_39 == 0
    var_40 = {var_1: var_3}
    var_41 = {var_0: var_40}
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = 'a'
    var_47 = 'x'
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_41, no_default=var_49)
    var_51 = {var_46: var_49}
    var_52 = []
    var_53 = module_0.get_in(var_52, var_51)
    var_54 = {var_46: var_49}
    var_55 = [var_46, var_47]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 is None
    var_57 = [var_46, var_47]
    var_58 = module_0.get_in(var_57, var_54, var_15)
    assert var_58 == 0



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
    var_26 = [var_19]
    var_27 = module_0.get_in(var_26, var_23)
    var_28 = 5
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_23)
    assert var_30 is None
    var_31 = [var_28]
    var_32 = module_0.get_in(var_31, var_23, var_15)
    assert var_32 == 'default'
    var_33 = {var_1: var_19}
    var_34 = [var_3, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_3, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 2
    var_38 = [var_0, var_3]
    var_39 = module_0.get_in(var_38, var_35)
    var_40 = [var_0, var_28]
    var_41 = module_0.get_in(var_40, var_35)
    assert var_41 is None
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = 5
    var_48 = [var_47]
    var_49 = 1
    var_50 = 2
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.get_in(var_48, var_51, no_default=var_52)
    var_54 = {var_47: var_50}
    var_55 = []
    var_56 = module_0.get_in(var_55, var_54)
    var_57 = 5
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 == 5
    var_60 = [var_11]
    var_61 = module_0.get_in(var_60, var_57)
    assert var_61 is None
    var_62 = 'x'
    var_63 = [var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_57, no_default=var_64)



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
    var_19 = [var_0, var_1, var_13]
    var_20 = module_0.get_in(var_19, var_6)
    assert var_20 is None
    var_21 = [var_0, var_1, var_13]
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 0
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = [var_17, var_3]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = [var_3, var_17]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 3
    var_33 = [var_17]
    var_34 = module_0.get_in(var_33, var_28)
    var_35 = [var_23]
    var_36 = module_0.get_in(var_35, var_28)
    assert var_36 is None
    var_37 = [var_23]
    var_38 = module_0.get_in(var_37, var_28, var_17)
    assert var_38 == 0
    var_39 = [var_17, var_23]
    var_40 = module_0.get_in(var_39, var_28)
    assert var_40 is None
    var_41 = [var_17, var_23]
    var_42 = module_0.get_in(var_41, var_28, var_17)
    assert var_42 == 0
    var_43 = {var_1: var_23}
    var_44 = [var_3, var_43]
    var_45 = {var_0: var_44}
    var_46 = [var_0, var_3, var_1]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 2
    var_48 = [var_0, var_3]
    var_49 = module_0.get_in(var_48, var_45)
    var_50 = [var_0, var_17]
    var_51 = module_0.get_in(var_50, var_45)
    assert var_51 == 1
    var_52 = [var_0, var_23]
    var_53 = module_0.get_in(var_52, var_45)
    assert var_53 is None
    var_54 = [var_0, var_23]
    var_55 = module_0.get_in(var_54, var_45, var_17)
    assert var_55 == 0
    var_56 = {var_0: var_3}
    var_57 = 'b'
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_56, no_default=var_59)
    var_61 = 0
    var_62 = [var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_56, no_default=var_63)
    var_65 = {var_61: var_64}
    var_66 = []
    var_67 = module_0.get_in(var_66, var_65)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_65, var_17)
    var_70 = {var_61: var_64}
    var_71 = [var_62]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 is None
    var_73 = [var_61, var_62]
    var_74 = module_0.get_in(var_73, var_70)
    assert var_74 is None
    var_75 = [var_17]
    var_76 = module_0.get_in(var_75, var_70)
    assert var_76 is None



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
    var_20 = [var_13]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 0
    var_23 = [var_0, var_13]
    var_24 = module_0.get_in(var_23, var_6, var_21)
    assert var_24 == 0
    var_25 = [var_0, var_1, var_13]
    var_26 = module_0.get_in(var_25, var_6, var_21)
    assert var_26 == 0
    var_27 = 'x'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'a'
    var_32 = 'x'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_6, no_default=var_34)
    var_36 = 2
    var_37 = 3
    var_38 = 4
    var_39 = [var_38]
    var_40 = [var_37, var_39]
    var_41 = [var_36, var_40]
    var_42 = [var_34, var_41]
    var_43 = [var_34, var_34, var_34]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 4
    var_45 = [var_34, var_34]
    var_46 = module_0.get_in(var_45, var_42)
    var_47 = [var_34]
    var_48 = module_0.get_in(var_47, var_42)
    var_49 = [var_34, var_34, var_34, var_34]
    var_50 = module_0.get_in(var_49, var_42)
    assert var_50 is None
    var_51 = [var_34, var_34, var_34, var_34]
    var_52 = module_0.get_in(var_51, var_42, var_21)
    assert var_52 == 0
    var_53 = {var_32: var_36}
    var_54 = [var_34, var_53]
    var_55 = {var_31: var_54}
    var_56 = [var_31, var_34, var_32]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 == 2
    var_58 = [var_31, var_34]
    var_59 = module_0.get_in(var_58, var_55)
    var_60 = [var_31, var_21]
    var_61 = module_0.get_in(var_60, var_55)
    assert var_61 == 1
    var_62 = [var_31, var_36]
    var_63 = module_0.get_in(var_62, var_55)
    assert var_63 is None
    var_64 = [var_31, var_36]
    var_65 = module_0.get_in(var_64, var_55, var_21)
    assert var_65 == 0
    var_66 = []
    var_67 = module_0.get_in(var_66, var_6)
    var_68 = []
    var_69 = module_0.get_in(var_68, var_42)
    var_70 = []
    var_71 = module_0.get_in(var_70, var_55)
    var_72 = 123
    var_73 = [var_31]
    var_74 = module_0.get_in(var_73, var_72)
    assert var_74 is None
    var_75 = [var_31]
    var_76 = module_0.get_in(var_75, var_72, var_21)
    assert var_76 == 0



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
    var_38 = [var_11]
    var_39 = module_0.get_in(var_38, var_33)
    assert var_39 is None
    var_40 = [var_11]
    var_41 = module_0.get_in(var_40, var_33, var_15)
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
    var_54 = []
    var_55 = module_0.get_in(var_54, var_51, var_15)
    var_56 = None
    var_57 = {var_47: var_56}
    var_58 = [var_47]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_47]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 is None
    var_62 = [var_11]
    var_63 = module_0.get_in(var_62, var_57, var_15)
    assert var_63 == 0



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
    var_20 = [var_13]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 0
    var_23 = [var_0, var_13]
    var_24 = module_0.get_in(var_23, var_6, var_21)
    assert var_24 == 0
    var_25 = [var_0, var_1, var_13]
    var_26 = module_0.get_in(var_25, var_6, var_21)
    assert var_26 == 0
    var_27 = 'x'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'a'
    var_32 = 'x'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_6, no_default=var_34)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = 'x'
    var_39 = [var_36, var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_6, no_default=var_40)
    var_42 = 2
    var_43 = 3
    var_44 = 4
    var_45 = 5
    var_46 = 6
    var_47 = [var_45, var_46]
    var_48 = [var_43, var_44, var_47]
    var_49 = [var_39, var_42, var_48]
    var_50 = [var_42, var_42, var_39]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 6
    var_52 = [var_42, var_42]
    var_53 = module_0.get_in(var_52, var_49)
    var_54 = [var_42]
    var_55 = module_0.get_in(var_54, var_49)
    var_56 = 10
    var_57 = [var_56]
    var_58 = module_0.get_in(var_57, var_49)
    assert var_58 is None
    var_59 = [var_42, var_56]
    var_60 = module_0.get_in(var_59, var_49)
    assert var_60 is None
    var_61 = [var_42, var_42, var_56]
    var_62 = module_0.get_in(var_61, var_49)
    assert var_62 is None
    var_63 = [var_56]
    var_64 = module_0.get_in(var_63, var_49, var_21)
    assert var_64 == 0
    var_65 = [var_42, var_56]
    var_66 = module_0.get_in(var_65, var_49, var_21)
    assert var_66 == 0
    var_67 = [var_42, var_42, var_56]
    var_68 = module_0.get_in(var_67, var_49, var_21)
    assert var_68 == 0
    var_69 = 10
    var_70 = [var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_49, no_default=var_71)
    var_73 = 2
    var_74 = 10
    var_75 = [var_73, var_74]
    var_76 = True
    var_77 = module_0.get_in(var_75, var_49, no_default=var_76)
    var_78 = 2
    var_79 = 10
    var_80 = [var_78, var_78, var_79]
    var_81 = True
    var_82 = module_0.get_in(var_80, var_49, no_default=var_81)
    var_83 = [var_43, var_44]
    var_84 = {var_79: var_83}
    var_85 = [var_81, var_42, var_84]
    var_86 = {var_78: var_85}
    var_87 = [var_78, var_42, var_79, var_81]
    var_88 = module_0.get_in(var_87, var_86)
    assert var_88 == 4
    var_89 = [var_78, var_42, var_79]
    var_90 = module_0.get_in(var_89, var_86)
    var_91 = [var_78, var_42]
    var_92 = module_0.get_in(var_91, var_86)
    var_93 = [var_78]
    var_94 = module_0.get_in(var_93, var_86)
    var_95 = [var_13]
    var_96 = module_0.get_in(var_95, var_86)
    assert var_96 is None
    var_97 = [var_78, var_56]
    var_98 = module_0.get_in(var_97, var_86)
    assert var_98 is None
    var_99 = [var_78, var_42, var_13]
    var_100 = module_0.get_in(var_99, var_86)
    assert var_100 is None
    var_101 = [var_78, var_42, var_79, var_56]
    var_102 = module_0.get_in(var_101, var_86)
    assert var_102 is None
    var_103 = [var_13]
    var_104 = module_0.get_in(var_103, var_86, var_21)
    assert var_104 == 0
    var_105 = [var_78, var_56]
    var_106 = module_0.get_in(var_105, var_86, var_21)
    assert var_106 == 0
    var_107 = [var_78, var_42, var_13]
    var_108 = module_0.get_in(var_107, var_86, var_21)
    assert var_108 == 0
    var_109 = [var_78, var_42, var_79, var_56]
    var_110 = module_0.get_in(var_109, var_86, var_21)
    assert var_110 == 0
    var_111 = 'x'
    var_112 = [var_111]
    var_113 = True
    var_114 = module_0.get_in(var_112, var_86, no_default=var_113)
    var_115 = 'a'
    var_116 = 10
    var_117 = [var_115, var_116]
    var_118 = True
    var_119 = module_0.get_in(var_117, var_86, no_default=var_118)
    var_120 = 'a'
    var_121 = 2
    var_122 = 'x'
    var_123 = [var_120, var_121, var_122]
    var_124 = True
    var_125 = module_0.get_in(var_123, var_86, no_default=var_124)
    var_126 = 'a'
    var_127 = 2
    var_128 = 'b'
    var_129 = 10
    var_130 = [var_126, var_127, var_128, var_129]
    var_131 = True
    var_132 = module_0.get_in(var_130, var_86, no_default=var_131)
    var_133 = []
    var_134 = module_0.get_in(var_133, var_6)
    var_135 = []
    var_136 = module_0.get_in(var_135, var_49)
    var_137 = []
    var_138 = module_0.get_in(var_137, var_86)
    var_139 = 123
    var_140 = [var_13]
    var_141 = module_0.get_in(var_140, var_139)
    assert var_141 is None
    var_142 = [var_13]
    var_143 = module_0.get_in(var_142, var_139, var_21)
    assert var_143 == 0
    var_144 = 'x'
    var_145 = [var_144]
    var_146 = True
    var_147 = module_0.get_in(var_145, var_139, no_default=var_146)



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
    var_18 = 42
    var_19 = module_0.get_in(var_17, var_16, var_18)
    assert var_19 == 42
    var_20 = [var_3, var_9, var_10]
    var_21 = {var_0: var_20}
    var_22 = 5
    var_23 = [var_0, var_22]
    var_24 = module_0.get_in(var_23, var_21, var_18)
    assert var_24 == 42
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
    var_39 = {var_34: var_37}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_35: var_9}
    var_43 = {var_36: var_10}
    var_44 = [var_42, var_43]
    var_45 = {var_34: var_44}
    var_46 = [var_34, var_37, var_36]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 3
    var_48 = {var_34: var_37}
    var_49 = [var_34, var_35]
    var_50 = module_0.get_in(var_49, var_48, var_18)
    assert var_50 == 42



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
    var_60 = [var_46, var_48]
    var_61 = module_0.get_in(var_60, var_55, var_15)
    assert var_61 == 'default'



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
    var_17 = 2
    var_18 = 3
    var_19 = 4
    var_20 = [var_19]
    var_21 = [var_18, var_20]
    var_22 = [var_17, var_21]
    var_23 = [var_3, var_22]
    var_24 = [var_3, var_3, var_3]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 4
    var_26 = [var_3, var_3]
    var_27 = module_0.get_in(var_26, var_23)
    var_28 = [var_3, var_3, var_3, var_3]
    var_29 = module_0.get_in(var_28, var_23)
    assert var_29 is None
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_3]
    var_36 = module_0.get_in(var_35, var_32)
    var_37 = [var_0, var_3, var_11]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = {var_0: var_3}
    var_40 = [var_0]
    var_41 = module_0.get_in(var_40, var_39)
    assert var_41 == 1
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_39, no_default=var_44)
    var_46 = {var_42: var_45}
    var_47 = [var_11]
    var_48 = module_0.get_in(var_47, var_46, var_15)
    assert var_48 == 'default'
    var_49 = 'y'
    var_50 = [var_11, var_49]
    var_51 = module_0.get_in(var_50, var_46, var_15)
    assert var_51 == 'default'
    var_52 = {var_42: var_45}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = 5
    var_56 = [var_11]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_11]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 == 'default'



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



# Parsed testcases at query #26
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
    var_34 = {var_29: var_32}
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_35, var_36]
    var_38 = True
    var_39 = module_0.get_in(var_37, var_34, no_default=var_38)
    var_40 = {var_35: var_38}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = {var_36: var_38}
    var_44 = {var_37: var_9}
    var_45 = [var_43, var_44]
    var_46 = {var_35: var_45}
    var_47 = [var_35, var_38, var_37]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 2



# Parsed testcases at query #27
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
    var_49 = None
    var_50 = {var_43: var_49}
    var_51 = {var_42: var_50}
    var_52 = [var_42, var_43]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_42, var_43]
    var_55 = module_0.get_in(var_54, var_51, var_15)
    assert var_55 is None
    var_56 = [var_42, var_44]
    var_57 = module_0.get_in(var_56, var_51, var_15)
    assert var_57 == 'default'



# Parsed testcases at query #28
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
    var_43 = [var_0]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 1
    var_45 = 'b'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_42, no_default=var_47)
    var_49 = {var_45: var_48}
    var_50 = []
    var_51 = module_0.get_in(var_50, var_49)
    var_52 = None
    var_53 = {var_46: var_52}
    var_54 = {var_45: var_53}
    var_55 = [var_45, var_46]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 is None
    var_57 = [var_45, var_46]
    var_58 = module_0.get_in(var_57, var_54, var_15)
    assert var_58 is None
    var_59 = [var_45, var_47]
    var_60 = module_0.get_in(var_59, var_54, var_15)
    assert var_60 == 0



# Parsed testcases at query #29
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
    var_42 = {var_38: var_41}
    var_43 = [var_38, var_39]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 is None
    var_45 = {var_38: var_41}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)



# Parsed testcases at query #30
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
    var_51 = [var_43]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 == 1
    var_53 = [var_44]
    var_54 = module_0.get_in(var_53, var_50)
    assert var_54 is None
    var_55 = [var_44]
    var_56 = module_0.get_in(var_55, var_50, var_15)
    assert var_56 == 0
    var_57 = 123
    var_58 = [var_43]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_43]
    var_61 = module_0.get_in(var_60, var_57, var_15)
    assert var_61 == 0



# Parsed testcases at query #31
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
    var_20 = {var_1: var_3}
    var_21 = {var_0: var_20}
    var_22 = [var_0, var_2]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 is None
    var_24 = {var_1: var_3}
    var_25 = {var_0: var_24}
    var_26 = 'a'
    var_27 = 'c'
    var_28 = [var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_25, no_default=var_29)
    var_31 = [var_29, var_9, var_10]
    var_32 = {var_26: var_31}
    var_33 = 'a'
    var_34 = 10
    var_35 = [var_33, var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_32, no_default=var_36)
    var_38 = {var_33: var_36}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_34: var_9}
    var_42 = {var_35: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_33: var_43}
    var_45 = [var_33, var_36, var_35]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 == 3
    var_47 = {var_33: var_36}
    var_48 = [var_33, var_34]
    var_49 = module_0.get_in(var_48, var_47, var_18)
    assert var_49 == 42



# Parsed testcases at query #32
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
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_12, var_18)
    assert var_19 == 'not found'
    var_20 = [var_15, var_16]
    var_21 = module_0.get_in(var_20, var_12)
    assert var_21 is None
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_12, no_default=var_25)
    var_27 = [var_25, var_9]
    var_28 = 4
    var_29 = [var_10, var_28]
    var_30 = [var_27, var_29]
    var_31 = {var_22: var_30}
    var_32 = 0
    var_33 = [var_22, var_25, var_32]
    var_34 = module_0.get_in(var_33, var_31)
    assert var_34 == 3
    var_35 = [var_25, var_9, var_10]
    var_36 = {var_23: var_35}
    var_37 = {var_22: var_36}
    var_38 = [var_22, var_23, var_9]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 3
    var_40 = []
    var_41 = module_0.get_in(var_40, var_37)
    var_42 = {var_22: var_25}
    var_43 = [var_22, var_23]
    var_44 = 'error'
    var_45 = module_0.get_in(var_43, var_42, var_44)
    assert var_45 == 'error'
    var_46 = [var_25, var_9, var_10]
    var_47 = {var_22: var_46}
    var_48 = 10
    var_49 = [var_22, var_48]
    var_50 = 'out of range'
    var_51 = module_0.get_in(var_49, var_47, var_50)
    assert var_51 == 'out of range'



# Parsed testcases at query #33
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
    var_11 = 4
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_3, var_13]
    var_15 = 0
    var_16 = [var_3, var_3, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 3
    var_18 = {var_1: var_3}
    var_19 = {var_0: var_18}
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_19, var_15)
    assert var_21 == 0
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = 5
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_27, no_default=var_30)
    var_32 = 1
    var_33 = 0
    var_34 = [var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_32, no_default=var_35)
    var_37 = {var_33: var_36}
    var_38 = []
    var_39 = module_0.get_in(var_38, var_37)
    var_40 = {var_34: var_36}
    var_41 = {var_33: var_40}
    var_42 = 'x'
    var_43 = 'y'
    var_44 = 'z'
    var_45 = [var_42, var_43, var_44]
    var_46 = 'default'
    var_47 = module_0.get_in(var_45, var_41, var_46)
    assert var_47 == 'default'
    var_48 = {var_34: var_9}
    var_49 = [var_36, var_48]
    var_50 = {var_33: var_49}
    var_51 = [var_33, var_36, var_34]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 == 2



# Parsed testcases at query #34
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
    var_18 = 3
    var_19 = 4
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = [var_3, var_21]
    var_23 = [var_3, var_3, var_15]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 3
    var_25 = [var_15]
    var_26 = module_0.get_in(var_25, var_22)
    assert var_26 == 1
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



# Parsed testcases at query #35
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
    var_25 = 5
    var_26 = 6
    var_27 = [var_25, var_26]
    var_28 = 7
    var_29 = 8
    var_30 = [var_28, var_29]
    var_31 = [var_27, var_30]
    var_32 = [var_24, var_31]
    var_33 = [var_15, var_3, var_3]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 4
    var_35 = [var_3, var_15]
    var_36 = module_0.get_in(var_35, var_32)
    var_37 = [var_19]
    var_38 = module_0.get_in(var_37, var_32)
    assert var_38 is None
    var_39 = [var_15, var_3, var_19]
    var_40 = module_0.get_in(var_39, var_32, var_15)
    assert var_40 == 0
    var_41 = {var_1: var_19}
    var_42 = [var_3, var_41]
    var_43 = {var_0: var_42}
    var_44 = [var_0, var_3, var_1]
    var_45 = module_0.get_in(var_44, var_43)
    assert var_45 == 2
    var_46 = [var_0, var_3]
    var_47 = module_0.get_in(var_46, var_43)
    var_48 = [var_0, var_19]
    var_49 = module_0.get_in(var_48, var_43)
    assert var_49 is None
    var_50 = [var_0, var_19]
    var_51 = module_0.get_in(var_50, var_43, var_15)
    assert var_51 == 0
    var_52 = {var_0: var_3}
    var_53 = 'b'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_52, no_default=var_55)
    var_57 = 0
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_52, no_default=var_59)
    var_61 = {var_57: var_60}
    var_62 = []
    var_63 = module_0.get_in(var_62, var_61)
    var_64 = {var_57: var_60}
    var_65 = [var_58]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 is None
    var_67 = [var_57, var_58]
    var_68 = module_0.get_in(var_67, var_64)
    assert var_68 is None



# Parsed testcases at query #36
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
    var_23 = 5
    var_24 = 6
    var_25 = [var_23, var_24]
    var_26 = 7
    var_27 = 8
    var_28 = [var_26, var_27]
    var_29 = [var_25, var_28]
    var_30 = [var_22, var_29]
    var_31 = [var_15, var_3, var_3]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 4
    var_33 = [var_3, var_15]
    var_34 = module_0.get_in(var_33, var_30)
    var_35 = [var_17]
    var_36 = module_0.get_in(var_35, var_30)
    assert var_36 is None
    var_37 = [var_17]
    var_38 = module_0.get_in(var_37, var_30, var_15)
    assert var_38 == 0
    var_39 = {var_1: var_17}
    var_40 = [var_3, var_39]
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_3, var_1]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 2
    var_44 = [var_0, var_15]
    var_45 = module_0.get_in(var_44, var_41)
    assert var_45 == 1
    var_46 = [var_11]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 is None
    var_48 = [var_11]
    var_49 = module_0.get_in(var_48, var_41, var_15)
    assert var_49 == 0
    var_50 = {var_0: var_3}
    var_51 = 'x'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_50, no_default=var_53)
    var_55 = {var_51: var_54}
    var_56 = []
    var_57 = module_0.get_in(var_56, var_55)
    var_58 = {var_52: var_54}
    var_59 = {var_51: var_58}
    var_60 = [var_51, var_53]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 is None
    var_62 = [var_51, var_53]
    var_63 = module_0.get_in(var_62, var_59, var_15)
    assert var_63 == 0
    var_64 = {var_51: var_54}
    var_65 = [var_51, var_52]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 is None
    var_67 = [var_51, var_52]
    var_68 = module_0.get_in(var_67, var_64, var_15)
    assert var_68 == 0



# Parsed testcases at query #37
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
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = module_0.get_in(var_17, var_12)
    assert var_18 is None
    var_19 = [var_15, var_16]
    var_20 = 'default'
    var_21 = module_0.get_in(var_19, var_12, var_20)
    assert var_21 == 'default'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_12, no_default=var_25)
    var_27 = 'a'
    var_28 = 10
    var_29 = [var_27, var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_12, no_default=var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = [var_32, var_33]
    var_35 = 123
    var_36 = True
    var_37 = module_0.get_in(var_34, var_35, no_default=var_36)
    var_38 = []
    var_39 = module_0.get_in(var_38, var_12)
    var_40 = [var_35, var_9, var_10]
    var_41 = {var_33: var_40}
    var_42 = 'value'
    var_43 = {var_34: var_42}
    var_44 = [var_41, var_43]
    var_45 = {var_32: var_44}
    var_46 = 0
    var_47 = [var_32, var_46, var_33, var_35]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 2
    var_49 = [var_32, var_35, var_34]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 == 'value'
    var_51 = [var_32, var_9]
    var_52 = module_0.get_in(var_51, var_45)
    assert var_52 is None



# Parsed testcases at query #38
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
    var_50 = None
    var_51 = {var_44: var_50}
    var_52 = {var_43: var_51}
    var_53 = [var_43, var_44]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = [var_43, var_44]
    var_56 = module_0.get_in(var_55, var_52, var_15)
    assert var_56 is None
    var_57 = [var_43, var_45]
    var_58 = module_0.get_in(var_57, var_52, var_15)
    assert var_58 == 0



# Parsed testcases at query #39
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
    var_25 = [var_17, var_3]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 2
    var_27 = [var_3, var_17]
    var_28 = module_0.get_in(var_27, var_24)
    assert var_28 == 3
    var_29 = [var_19]
    var_30 = module_0.get_in(var_29, var_24)
    assert var_30 is None
    var_31 = [var_19]
    var_32 = module_0.get_in(var_31, var_24, var_17)
    assert var_32 == 0
    var_33 = {var_1: var_19}
    var_34 = [var_3, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_3, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 2
    var_38 = [var_0, var_3]
    var_39 = module_0.get_in(var_38, var_35)
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_35)
    assert var_41 == 1
    var_42 = [var_0, var_19]
    var_43 = module_0.get_in(var_42, var_35)
    assert var_43 is None
    var_44 = [var_0, var_19]
    var_45 = module_0.get_in(var_44, var_35, var_17)
    assert var_45 == 0
    var_46 = {var_1: var_3}
    var_47 = {var_0: var_46}
    var_48 = 'x'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = 'a'
    var_53 = 'x'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_47, no_default=var_55)
    var_57 = 0
    var_58 = [var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_47, no_default=var_59)
    var_61 = {var_57: var_60}
    var_62 = []
    var_63 = module_0.get_in(var_62, var_61)
    var_64 = {var_59: var_60}
    var_65 = {var_58: var_64}
    var_66 = {var_57: var_65}
    var_67 = [var_57, var_58, var_13]
    var_68 = module_0.get_in(var_67, var_66)
    assert var_68 is None
    var_69 = [var_57, var_58, var_13]
    var_70 = module_0.get_in(var_69, var_66, var_17)
    assert var_70 == 0
    var_71 = {var_57: var_60}
    var_72 = [var_57, var_58]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = [var_57, var_58]
    var_75 = module_0.get_in(var_74, var_71, var_17)
    assert var_75 == 0



# Parsed testcases at query #40
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
    var_15 = [var_3, var_13, var_14]
    var_16 = {var_0: var_15}
    var_17 = 0
    var_18 = [var_0, var_17]
    var_19 = module_0.get_in(var_18, var_16)
    assert var_19 == 1
    var_20 = [var_0, var_3]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 == 2
    var_22 = [var_0, var_13]
    var_23 = module_0.get_in(var_22, var_16)
    assert var_23 == 3
    var_24 = 'x'
    var_25 = [var_24]
    var_26 = module_0.get_in(var_25, var_16)
    assert var_26 is None
    var_27 = [var_24]
    var_28 = 'default'
    var_29 = module_0.get_in(var_27, var_16, var_28)
    assert var_29 == 'default'
    var_30 = 5
    var_31 = [var_0, var_30]
    var_32 = 'out_of_bounds'
    var_33 = module_0.get_in(var_31, var_16, var_32)
    assert var_33 == 'out_of_bounds'
    var_34 = 'x'
    var_35 = [var_34]
    var_36 = True
    var_37 = module_0.get_in(var_35, var_16, no_default=var_36)
    var_38 = 'a'
    var_39 = 5
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_16, no_default=var_41)
    var_43 = 123
    var_44 = {var_38: var_43}
    var_45 = [var_38, var_39]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 is None
    var_47 = [var_38, var_39]
    var_48 = 'not_a_dict'
    var_49 = module_0.get_in(var_47, var_44, var_48)
    assert var_49 == 'not_a_dict'
    var_50 = 'a'
    var_51 = 'b'
    var_52 = [var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_44, no_default=var_53)
    var_55 = []
    var_56 = module_0.get_in(var_55, var_44)
    var_57 = [var_53, var_13, var_14]
    var_58 = {var_51: var_57}
    var_59 = 4
    var_60 = {var_52: var_59}
    var_61 = [var_58, var_60]
    var_62 = {var_50: var_61}
    var_63 = [var_50, var_17, var_51, var_53]
    var_64 = module_0.get_in(var_63, var_62)
    assert var_64 == 2
    var_65 = [var_50, var_53, var_52]
    var_66 = module_0.get_in(var_65, var_62)
    assert var_66 == 4
    var_67 = [var_50, var_17, var_51, var_30]
    var_68 = module_0.get_in(var_67, var_62)
    assert var_68 is None
    var_69 = [var_50, var_17, var_51, var_30]
    var_70 = 99
    var_71 = module_0.get_in(var_69, var_62, var_70)
    assert var_71 == 99



# Parsed testcases at query #41
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
    var_20 = {var_1: var_3}
    var_21 = {var_0: var_20}
    var_22 = [var_0, var_2]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 is None
    var_24 = {var_1: var_3}
    var_25 = {var_0: var_24}
    var_26 = 'a'
    var_27 = 'c'
    var_28 = [var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_25, no_default=var_29)
    var_31 = [var_29, var_9, var_10]
    var_32 = {var_26: var_31}
    var_33 = 10
    var_34 = [var_26, var_33]
    var_35 = module_0.get_in(var_34, var_32, var_18)
    assert var_35 == 42
    var_36 = [var_29, var_9, var_10]
    var_37 = {var_26: var_36}
    var_38 = 'a'
    var_39 = 10
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = module_0.get_in(var_40, var_37, no_default=var_41)
    var_43 = {var_38: var_41}
    var_44 = [var_38, var_39]
    var_45 = module_0.get_in(var_44, var_43, var_18)
    assert var_45 == 42
    var_46 = {var_38: var_41}
    var_47 = 'a'
    var_48 = 'b'
    var_49 = [var_47, var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_46, no_default=var_50)
    var_52 = {var_47: var_50}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = 'name'
    var_56 = 'purchase'
    var_57 = 'credit card'
    var_58 = 'Alice'
    var_59 = 'items'
    var_60 = 'costs'
    var_61 = 'Apple'
    var_62 = 'Orange'
    var_63 = [var_61, var_62]
    var_64 = 0.5
    var_65 = 1.25
    var_66 = [var_64, var_65]
    var_67 = {var_59: var_63, var_60: var_66}
    var_68 = '5555-1234-1234-1234'
    var_69 = {var_55: var_58, var_56: var_67, var_57: var_68}
    var_70 = 0
    var_71 = [var_56, var_59, var_70]
    var_72 = [var_55]
    var_73 = 'total'
    var_74 = [var_56, var_73]
    var_75 = [var_56, var_73]



# Parsed testcases at query #42
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
    var_19 = [var_0, var_13]
    var_20 = module_0.get_in(var_19, var_6)
    assert var_20 is None
    var_21 = [var_0, var_13]
    var_22 = module_0.get_in(var_21, var_6, var_17)
    assert var_22 == 0
    var_23 = 2
    var_24 = [var_3, var_23]
    var_25 = 3
    var_26 = 4
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = 5
    var_30 = 6
    var_31 = [var_29, var_30]
    var_32 = 7
    var_33 = 8
    var_34 = [var_32, var_33]
    var_35 = [var_31, var_34]
    var_36 = [var_28, var_35]
    var_37 = [var_17, var_17, var_17]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 1
    var_39 = [var_3, var_3, var_3]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 == 8
    var_41 = [var_17, var_3]
    var_42 = module_0.get_in(var_41, var_36)
    var_43 = [var_23]
    var_44 = module_0.get_in(var_43, var_36)
    assert var_44 is None
    var_45 = [var_23]
    var_46 = module_0.get_in(var_45, var_36, var_17)
    assert var_46 == 0
    var_47 = [var_17, var_17, var_23]
    var_48 = module_0.get_in(var_47, var_36)
    assert var_48 is None
    var_49 = [var_17, var_17, var_23]
    var_50 = module_0.get_in(var_49, var_36, var_17)
    assert var_50 == 0
    var_51 = [var_3, var_23, var_25]
    var_52 = {var_1: var_51}
    var_53 = [var_26, var_29, var_30]
    var_54 = {var_2: var_53}
    var_55 = [var_52, var_54]
    var_56 = {var_0: var_55}
    var_57 = [var_0, var_17, var_1, var_3]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 == 2
    var_59 = [var_0, var_3, var_2, var_17]
    var_60 = module_0.get_in(var_59, var_56)
    assert var_60 == 4
    var_61 = [var_0, var_17, var_13]
    var_62 = module_0.get_in(var_61, var_56)
    assert var_62 is None
    var_63 = [var_0, var_17, var_13]
    var_64 = module_0.get_in(var_63, var_56, var_17)
    assert var_64 == 0
    var_65 = [var_0, var_23]
    var_66 = module_0.get_in(var_65, var_56)
    assert var_66 is None
    var_67 = [var_0, var_23]
    var_68 = module_0.get_in(var_67, var_56, var_17)
    assert var_68 == 0
    var_69 = {var_1: var_3}
    var_70 = {var_0: var_69}
    var_71 = 'x'
    var_72 = [var_71]
    var_73 = True
    var_74 = module_0.get_in(var_72, var_70, no_default=var_73)
    var_75 = 'a'
    var_76 = 'x'
    var_77 = [var_75, var_76]
    var_78 = True
    var_79 = module_0.get_in(var_77, var_70, no_default=var_78)
    var_80 = [var_78, var_23, var_25]
    var_81 = [var_80]
    var_82 = 0
    var_83 = 5
    var_84 = [var_82, var_83]
    var_85 = True
    var_86 = module_0.get_in(var_84, var_81, no_default=var_85)
    var_87 = {var_82: var_85}
    var_88 = []
    var_89 = module_0.get_in(var_88, var_87)
    var_90 = 'string'
    var_91 = []
    var_92 = module_0.get_in(var_91, var_90)
    var_93 = [var_17]
    var_94 = module_0.get_in(var_93, var_90)
    assert var_94 == 's'
    var_95 = [var_85]
    var_96 = module_0.get_in(var_95, var_90)
    assert var_96 == 't'
    var_97 = [var_29]
    var_98 = module_0.get_in(var_97, var_90)
    assert var_98 == 'g'
    var_99 = [var_30]
    var_100 = module_0.get_in(var_99, var_90)
    assert var_100 is None
    var_101 = [var_30]
    var_102 = module_0.get_in(var_101, var_90, var_17)
    assert var_102 == 0
    var_103 = 6
    var_104 = [var_103]
    var_105 = True
    var_106 = module_0.get_in(var_104, var_90, no_default=var_105)



# Parsed testcases at query #43
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
    var_25 = [var_17, var_3]
    var_26 = module_0.get_in(var_25, var_24)
    assert var_26 == 2
    var_27 = [var_3, var_17]
    var_28 = module_0.get_in(var_27, var_24)
    assert var_28 == 3
    var_29 = [var_19]
    var_30 = module_0.get_in(var_29, var_24)
    assert var_30 is None
    var_31 = [var_19]
    var_32 = module_0.get_in(var_31, var_24, var_17)
    assert var_32 == 0
    var_33 = {var_1: var_19}
    var_34 = [var_3, var_33]
    var_35 = {var_0: var_34}
    var_36 = [var_0, var_3, var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 == 2
    var_38 = [var_0, var_3]
    var_39 = module_0.get_in(var_38, var_35)
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_35)
    assert var_41 == 1
    var_42 = [var_0, var_19]
    var_43 = module_0.get_in(var_42, var_35)
    assert var_43 is None
    var_44 = [var_0, var_19]
    var_45 = module_0.get_in(var_44, var_35, var_17)
    assert var_45 == 0
    var_46 = {var_0: var_3}
    var_47 = 'b'
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_46, no_default=var_49)
    var_51 = {var_47: var_50}
    var_52 = []
    var_53 = module_0.get_in(var_52, var_51)
    var_54 = {var_48: var_50}
    var_55 = {var_47: var_54}
    var_56 = [var_47, var_49]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_47, var_49]
    var_59 = module_0.get_in(var_58, var_55, var_17)
    assert var_59 == 0
    var_60 = {var_47: var_50}
    var_61 = [var_47, var_48]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 is None
    var_63 = [var_47, var_48]
    var_64 = module_0.get_in(var_63, var_60, var_17)
    assert var_64 == 0



# Parsed testcases at query #44
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
    var_18 = 3
    var_19 = [var_18]
    var_20 = [var_17, var_19]
    var_21 = [var_3, var_20]
    var_22 = [var_3, var_3, var_15]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 3
    var_24 = [var_15]
    var_25 = module_0.get_in(var_24, var_21)
    assert var_25 == 1
    var_26 = [var_17]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 is None
    var_28 = [var_17]
    var_29 = module_0.get_in(var_28, var_21, var_15)
    assert var_29 == 0
    var_30 = {var_1: var_17}
    var_31 = [var_3, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_3, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 2
    var_35 = [var_0, var_15]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 1
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_32)
    var_39 = [var_11]
    var_40 = module_0.get_in(var_39, var_32)
    assert var_40 is None
    var_41 = {var_0: var_3}
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = [var_45, var_17, var_18]
    var_47 = 5
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_46, no_default=var_49)
    var_51 = 5
    var_52 = 0
    var_53 = [var_52]
    var_54 = True
    var_55 = module_0.get_in(var_53, var_51, no_default=var_54)
    var_56 = {var_52: var_55}
    var_57 = []
    var_58 = module_0.get_in(var_57, var_56)
    var_59 = None
    var_60 = {var_53: var_59}
    var_61 = {var_52: var_60}
    var_62 = [var_52, var_53]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_52, var_53]
    var_65 = module_0.get_in(var_64, var_61, var_15)
    assert var_65 is None
    var_66 = [var_52, var_54]
    var_67 = module_0.get_in(var_66, var_61, var_15)
    assert var_67 == 0



# Parsed testcases at query #45
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
    var_20 = [var_13]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 0
    var_23 = [var_0, var_13]
    var_24 = module_0.get_in(var_23, var_6, var_21)
    assert var_24 == 0
    var_25 = [var_0, var_1, var_13]
    var_26 = module_0.get_in(var_25, var_6, var_21)
    assert var_26 == 0
    var_27 = 'x'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'a'
    var_32 = 'x'
    var_33 = [var_31, var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_6, no_default=var_34)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = 'x'
    var_39 = [var_36, var_37, var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_6, no_default=var_40)
    var_42 = 2
    var_43 = 3
    var_44 = 4
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = [var_39, var_46]
    var_48 = [var_39]
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = [var_39, var_39]
    var_51 = module_0.get_in(var_50, var_47)
    var_52 = [var_39, var_39, var_21]
    var_53 = module_0.get_in(var_52, var_47)
    assert var_53 == 3
    var_54 = [var_39, var_39, var_42]
    var_55 = module_0.get_in(var_54, var_47)
    assert var_55 is None
    var_56 = [var_39, var_39, var_42]
    var_57 = module_0.get_in(var_56, var_47, var_21)
    assert var_57 == 0
    var_58 = {var_37: var_42}
    var_59 = [var_39, var_58]
    var_60 = {var_36: var_59}
    var_61 = [var_36, var_39, var_37]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 == 2
    var_63 = [var_36, var_39, var_13]
    var_64 = module_0.get_in(var_63, var_60)
    assert var_64 is None
    var_65 = [var_36, var_39, var_13]
    var_66 = module_0.get_in(var_65, var_60, var_21)
    assert var_66 == 0
    var_67 = []
    var_68 = module_0.get_in(var_67, var_6)
    var_69 = []
    var_70 = module_0.get_in(var_69, var_47)
    var_71 = []
    var_72 = module_0.get_in(var_71, var_60)
    var_73 = {var_36: var_39}
    var_74 = [var_36]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 == 1
    var_76 = [var_13]
    var_77 = module_0.get_in(var_76, var_73)
    assert var_77 is None
    var_78 = [var_13]
    var_79 = module_0.get_in(var_78, var_73, var_21)
    assert var_79 == 0



# Parsed testcases at query #46
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
    var_20 = {var_0: var_3}
    var_21 = 'b'
    var_22 = [var_21]
    var_23 = True
    var_24 = module_0.get_in(var_22, var_20, no_default=var_23)
    var_25 = [var_24, var_9, var_10]
    var_26 = 5
    var_27 = [var_26]
    var_28 = True
    var_29 = module_0.get_in(var_27, var_25, no_default=var_28)
    var_30 = {var_27: var_9}
    var_31 = {var_28: var_10}
    var_32 = [var_30, var_31]
    var_33 = {var_26: var_32}
    var_34 = [var_26, var_29, var_28]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 3
    var_36 = {var_26: var_29}
    var_37 = []
    var_38 = module_0.get_in(var_37, var_36)
    var_39 = 'string'
    var_40 = {var_26: var_39}
    var_41 = [var_26, var_27]
    var_42 = 'default'
    var_43 = module_0.get_in(var_41, var_40, var_42)
    assert var_43 == 'default'
    var_44 = {var_26: var_29}
    var_45 = [var_27]
    var_46 = module_0.get_in(var_45, var_44)
    assert var_46 is None



# Parsed testcases at query #47
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
    var_18 = 3
    var_19 = 4
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = [var_3, var_21]
    var_23 = 0
    var_24 = [var_3, var_3, var_23]
    var_25 = module_0.get_in(var_24, var_22)
    assert var_25 == 3
    var_26 = [var_23]
    var_27 = module_0.get_in(var_26, var_22)
    assert var_27 == 1
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
    var_37 = [var_0, var_23]
    var_38 = module_0.get_in(var_37, var_34)
    assert var_38 == 1
    var_39 = [var_11]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 is None
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {}
    var_44 = True
    var_45 = module_0.get_in(var_42, var_43, no_default=var_44)
    var_46 = 0
    var_47 = [var_46]
    var_48 = []
    var_49 = True
    var_50 = module_0.get_in(var_47, var_48, no_default=var_49)
    var_51 = [var_11]
    var_52 = {}
    var_53 = module_0.get_in(var_51, var_52, var_23)
    assert var_53 == 0
    var_54 = [var_23]
    var_55 = []
    var_56 = module_0.get_in(var_54, var_55, var_23)
    assert var_56 == 0
    var_57 = []
    var_58 = module_0.get_in(var_57, var_34)



# Parsed testcases at query #48
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
    var_40 = [var_0, var_17]
    var_41 = module_0.get_in(var_40, var_33, var_15)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = [var_0]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 1
    var_45 = 'b'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_42, no_default=var_47)
    var_49 = {var_46: var_48}
    var_50 = {var_45: var_49}
    var_51 = [var_45, var_47]
    var_52 = module_0.get_in(var_51, var_50, var_15)
    assert var_52 == 0
    var_53 = [var_45, var_47]
    var_54 = None
    var_55 = module_0.get_in(var_53, var_50, var_54)
    assert var_55 is None
    var_56 = [var_45, var_47]
    var_57 = 'default'
    var_58 = module_0.get_in(var_56, var_50, var_57)
    assert var_58 == 'default'
    var_59 = {var_45: var_48}
    var_60 = []
    var_61 = module_0.get_in(var_60, var_59)
    var_62 = 123
    var_63 = [var_45]
    var_64 = module_0.get_in(var_63, var_62)
    assert var_64 is None
    var_65 = [var_45]
    var_66 = module_0.get_in(var_65, var_62, var_15)
    assert var_66 == 0



# Parsed testcases at query #49
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
    assert var_20 == 'Apple'
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 is None
    var_22 = {var_0: var_3}
    var_23 = 'b'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.get_in(var_24, var_22, no_default=var_25)
    var_27 = [var_26, var_9, var_10]
    var_28 = 5
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_27, no_default=var_30)
    var_32 = 'string'
    var_33 = 0
    var_34 = [var_33, var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_32, no_default=var_35)
    var_37 = {var_33: var_36}
    var_38 = []
    assert var_38 == 'Alice'
    var_39 = module_0.get_in(var_38, var_37)
    var_40 = 'name'
    var_41 = 'purchase'
    var_42 = 'credit card'
    var_43 = 'Alice'
    var_44 = 'items'
    var_45 = 'costs'
    var_46 = 'Apple'
    var_47 = 'Orange'
    var_48 = [var_46, var_47]
    var_49 = 0.5
    var_50 = 1.25
    var_51 = [var_49, var_50]
    var_52 = {var_44: var_48, var_45: var_51}
    var_53 = '5555-1234-1234-1234'
    var_54 = {var_40: var_43, var_41: var_52, var_42: var_53}
    var_55 = 0
    var_56 = [var_41, var_44, var_55]
    var_57 = [var_40]
    var_58 = 'total'
    var_59 = [var_41, var_58]
    var_60 = [var_41, var_58]



# Parsed testcases at query #50
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
    var_19 = 2
    var_20 = 3
    var_21 = 4
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = [var_3, var_23]
    var_25 = 0
    var_26 = [var_3, var_3, var_25]
    var_27 = module_0.get_in(var_26, var_24)
    assert var_27 == 3
    var_28 = [var_3, var_3]
    var_29 = module_0.get_in(var_28, var_24)
    var_30 = [var_3]
    var_31 = module_0.get_in(var_30, var_24)
    var_32 = [var_19]
    var_33 = module_0.get_in(var_32, var_24)
    assert var_33 is None
    var_34 = [var_19]
    var_35 = module_0.get_in(var_34, var_24, var_17)
    assert var_35 == 'default'
    var_36 = {var_1: var_19}
    var_37 = [var_3, var_36]
    var_38 = {var_0: var_37}
    var_39 = [var_0, var_3, var_1]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 2
    var_41 = [var_0, var_3]
    var_42 = module_0.get_in(var_41, var_38)
    var_43 = [var_0, var_25]
    var_44 = module_0.get_in(var_43, var_38)
    assert var_44 == 1
    var_45 = [var_13]
    var_46 = module_0.get_in(var_45, var_38)
    assert var_46 is None
    var_47 = {var_0: var_3}
    var_48 = 'x'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = {var_48: var_51}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = None
    var_56 = {var_49: var_55}
    var_57 = {var_48: var_56}
    var_58 = [var_48, var_49]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_48, var_49]
    var_61 = module_0.get_in(var_60, var_57, var_17)
    assert var_61 is None
    var_62 = [var_48, var_50]
    var_63 = module_0.get_in(var_62, var_57, var_17)
    assert var_63 == 'default'



# Parsed testcases at query #51
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
    var_41 = {var_0: var_3}
    var_42 = [var_1]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_41, no_default=var_43)
    assert var_44 is None
    var_45 = 'b'
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_41, no_default=var_47)
    var_49 = 'y'
    var_50 = [var_11, var_49]
    var_51 = module_0.get_in(var_50, var_41, var_15)
    assert var_51 == 'default'
    var_52 = []
    var_53 = module_0.get_in(var_52, var_41)
    var_54 = 'string'
    var_55 = {var_45: var_54}
    var_56 = [var_45, var_23]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_45, var_23]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 == 'default'



# Parsed testcases at query #52
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
    var_10 = [var_3, var_9]
    var_11 = 3
    var_12 = [var_10, var_11]
    var_13 = 4
    var_14 = [var_12, var_13]
    var_15 = 0
    var_16 = [var_15, var_15, var_3]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 2
    var_18 = [var_3, var_9, var_11]
    var_19 = {var_1: var_18}
    var_20 = [var_19, var_13]
    var_21 = {var_0: var_20}
    var_22 = [var_0, var_15, var_1, var_3]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 == 2
    var_24 = {var_1: var_3}
    var_25 = {var_0: var_24}
    var_26 = [var_0, var_2]
    var_27 = module_0.get_in(var_26, var_25)
    assert var_27 is None
    var_28 = [var_0, var_2]
    var_29 = 'default'
    var_30 = module_0.get_in(var_28, var_25, var_29)
    assert var_30 == 'default'
    var_31 = {var_0: var_3}
    var_32 = 'b'
    var_33 = [var_32]
    var_34 = True
    var_35 = module_0.get_in(var_33, var_31, no_default=var_34)
    var_36 = [var_35, var_9, var_11]
    var_37 = 5
    var_38 = [var_37]
    var_39 = True
    var_40 = module_0.get_in(var_38, var_36, no_default=var_39)
    var_41 = {var_37: var_40}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_38: var_9}
    var_45 = {var_37: var_44}
    var_46 = [var_37, var_38]
    var_47 = module_0.get_in(var_46, var_41)
    assert var_47 == 2
    var_48 = {var_37: var_40}
    var_49 = [var_37, var_38]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 is None
    var_51 = [var_37, var_38]
    var_52 = module_0.get_in(var_51, var_48, var_29)
    assert var_52 == 'default'
    var_53 = (var_40, var_9)
    var_54 = {var_37: var_40}
    var_55 = {var_53: var_54}
    var_56 = (var_40, var_9)
    var_57 = [var_56, var_37]
    var_58 = module_0.get_in(var_57, var_55)
    assert var_58 == 1



# Parsed testcases at query #53
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
    var_39 = [var_0, var_23]
    var_40 = module_0.get_in(var_39, var_34)
    assert var_40 == 1
    var_41 = [var_11]
    var_42 = module_0.get_in(var_41, var_34)
    assert var_42 is None
    var_43 = {var_0: var_3}
    var_44 = 'x'
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



# Parsed testcases at query #54
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
    var_30 = [var_17]
    var_31 = module_0.get_in(var_30, var_22, var_23)
    assert var_31 == 0
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_23]
    var_38 = module_0.get_in(var_37, var_34)
    assert var_38 == 1
    var_39 = [var_0, var_3, var_2]
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
    var_49 = {var_42: var_45}
    var_50 = [var_42, var_43]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 is None
    var_52 = [var_42, var_43]
    var_53 = module_0.get_in(var_52, var_49, var_15)
    assert var_53 == 'default'



# Parsed testcases at query #55
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
    var_29 = 5
    var_30 = [var_23, var_29]
    var_31 = module_0.get_in(var_30, var_28)
    assert var_31 is None
    var_32 = [var_23, var_29]
    var_33 = module_0.get_in(var_32, var_28, var_20)
    assert var_33 == 'default'
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
    var_42 = [var_26, var_9, var_10]
    var_43 = {var_24: var_42}
    var_44 = [var_43]
    var_45 = {var_23: var_44}
    var_46 = 0
    var_47 = [var_23, var_46, var_24, var_26]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 2



# Parsed testcases at query #56
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
    var_36 = [var_0, var_15]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 == 1
    var_38 = [var_0, var_3]
    var_39 = module_0.get_in(var_38, var_33)
    var_40 = [var_11]
    var_41 = module_0.get_in(var_40, var_33)
    assert var_41 is None
    var_42 = {var_0: var_3}
    var_43 = 'x'
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



# Parsed testcases at query #57
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
    var_29 = {var_1: var_17}
    var_30 = [var_3, var_29]
    var_31 = {var_0: var_30}
    var_32 = [var_0, var_3, var_1]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 2
    var_34 = [var_0, var_3, var_2]
    var_35 = module_0.get_in(var_34, var_31)
    assert var_35 is None
    var_36 = {var_0: var_3}
    var_37 = 'b'
    var_38 = [var_37]
    var_39 = True
    var_40 = module_0.get_in(var_38, var_36, no_default=var_39)
    var_41 = {var_37: var_40}
    var_42 = [var_37, var_38]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 is None
    var_44 = None
    var_45 = {var_38: var_44}
    var_46 = {var_37: var_45}
    var_47 = [var_37, var_38]
    var_48 = module_0.get_in(var_47, var_46, var_15)
    assert var_48 is None
    var_49 = [var_37, var_39]
    var_50 = module_0.get_in(var_49, var_46, var_15)
    assert var_50 == 0



# Parsed testcases at query #58
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
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_12, var_18)
    assert var_19 == 'not found'
    var_20 = [var_0, var_1, var_2]
    var_21 = {var_0: var_3}
    var_22 = module_0.get_in(var_20, var_21, var_18)
    assert var_22 == 'not found'
    var_23 = 'x'
    var_24 = 'y'
    var_25 = [var_23, var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_12, no_default=var_26)
    var_28 = 'a'
    var_29 = 10
    var_30 = [var_28, var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_12, no_default=var_31)
    var_33 = [var_15, var_16]
    var_34 = module_0.get_in(var_33, var_12)
    assert var_34 is None
    var_35 = []
    var_36 = module_0.get_in(var_35, var_12)
    var_37 = {var_29: var_31}
    var_38 = {var_30: var_9}
    var_39 = [var_37, var_38]
    var_40 = {var_28: var_39}
    var_41 = [var_28, var_31, var_30]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 == 2
    var_43 = 123
    var_44 = {var_28: var_43}
    var_45 = [var_28, var_29]
    var_46 = 'error'
    var_47 = module_0.get_in(var_45, var_44, var_46)
    assert var_47 == 'error'



# Parsed testcases at query #59
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
    var_28 = 5
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_27, no_default=var_30)
    var_32 = 'string'
    var_33 = 0
    var_34 = [var_33, var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_32, no_default=var_35)
    var_37 = {var_33: var_36}
    var_38 = []
    var_39 = module_0.get_in(var_38, var_37)
    var_40 = (var_36, var_9)
    var_41 = {var_34: var_40}
    var_42 = [var_41]
    var_43 = {var_33: var_42}
    var_44 = 0
    var_45 = [var_33, var_44, var_34, var_36]
    var_46 = module_0.get_in(var_45, var_43)
    assert var_46 == 2
    var_47 = {var_34: var_36}
    var_48 = {var_33: var_47}
    var_49 = 'd'
    var_50 = [var_33, var_34, var_35, var_49]
    var_51 = 'not found'
    var_52 = module_0.get_in(var_50, var_48, var_51)
    assert var_52 == 'not found'



# Parsed testcases at query #60
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
    var_10 = [var_3, var_9]
    var_11 = 3
    var_12 = 4
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = 0
    var_16 = [var_3, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 3
    var_18 = {var_1: var_3}
    var_19 = {var_0: var_18}
    var_20 = [var_0, var_2]
    var_21 = module_0.get_in(var_20, var_19, var_15)
    assert var_21 == 0
    var_22 = {var_1: var_3}
    var_23 = {var_0: var_22}
    var_24 = 'a'
    var_25 = 'c'
    var_26 = [var_24, var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_23, no_default=var_27)
    var_29 = [var_27, var_9, var_11]
    var_30 = 5
    var_31 = [var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_29, no_default=var_32)
    var_34 = 1
    var_35 = 0
    var_36 = [var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_34, no_default=var_37)
    var_39 = {var_35: var_38}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_35: var_38}
    var_43 = [var_35]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 == 1
    var_45 = {var_35: var_38}
    var_46 = [var_36]
    var_47 = None
    var_48 = module_0.get_in(var_46, var_45, var_47)
    assert var_48 is None
    var_49 = {var_36: var_9}
    var_50 = [var_38, var_49]
    var_51 = {var_35: var_50}
    var_52 = [var_35, var_38, var_36]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 == 2



# Parsed testcases at query #61
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
    var_18 = 3
    var_19 = 4
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = [var_3, var_21]
    var_23 = [var_3, var_3, var_15]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 == 3
    var_25 = [var_3, var_3]
    var_26 = module_0.get_in(var_25, var_22)
    var_27 = [var_3, var_3, var_17]
    var_28 = module_0.get_in(var_27, var_22)
    assert var_28 is None
    var_29 = [var_3, var_3, var_17]
    var_30 = -1
    var_31 = module_0.get_in(var_29, var_22, var_30)
    assert var_31 == -1
    var_32 = {var_1: var_17}
    var_33 = [var_3, var_32]
    var_34 = {var_0: var_33}
    var_35 = [var_0, var_3, var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 2
    var_37 = [var_0, var_3]
    var_38 = module_0.get_in(var_37, var_34)
    var_39 = [var_0, var_3, var_2]
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
    var_49 = {var_42: var_45}
    var_50 = [var_42]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 1
    var_52 = [var_43]
    var_53 = module_0.get_in(var_52, var_49)
    assert var_53 is None
    var_54 = None
    var_55 = {var_42: var_54}
    var_56 = [var_42]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_42]
    var_59 = module_0.get_in(var_58, var_55, var_15)
    assert var_59 is None
    var_60 = [var_43]
    var_61 = module_0.get_in(var_60, var_55, var_15)
    assert var_61 == 0



