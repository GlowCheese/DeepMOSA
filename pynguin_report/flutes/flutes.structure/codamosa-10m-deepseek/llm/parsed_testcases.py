# Parsed testcases at query #21
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x * var_0
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x * var_0
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = lambda x: x * var_0
    var_16 = (var_2, var_0)
    var_17 = (var_3, var_8)
    var_18 = (var_16, var_17)
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_2, var_22: var_0}
    var_24 = module_0.map_structure(var_20, var_23)
    var_25 = lambda x: x * var_0
    var_26 = 'c'
    var_27 = {var_26: var_2}
    var_28 = {var_21: var_27, var_22: var_0}
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_2, var_0, var_3}
    var_32 = module_0.map_structure(var_30, var_31)
    var_33 = lambda x: x * var_0
    var_34 = [var_2, var_0]
    var_35 = (var_3, var_8)
    var_36 = {var_21: var_34, var_22: var_35}
    var_37 = module_0.map_structure(var_33, var_36)
    var_38 = [var_2, var_0, var_3]
    var_39 = module_0.no_map_instance(var_38)
    var_40 = lambda x: x * var_0
    var_41 = module_0.map_structure(var_40, var_39)
    var_42 = [var_2, var_0, var_3]
    var_43 = lambda x: x * var_0
    var_44 = 'All tests passed!'
    var_45 = print(var_44)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = lambda x, y, z: x + y + z
    var_10 = [var_1, var_2]
    var_11 = [var_4, var_5]
    var_12 = 5
    var_13 = 6
    var_14 = [var_12, var_13]
    var_15 = [var_10, var_11, var_14]
    var_16 = module_0.map_structure_zip(var_9, var_15)
    var_17 = lambda x, y: x + y
    var_18 = (var_1, var_2)
    var_19 = (var_4, var_5)
    var_20 = [var_18, var_19]
    var_21 = module_0.map_structure_zip(var_17, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_4, var_5)
    var_25 = (var_12, var_13)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x + y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_4, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y, z: x + y + z
    var_36 = {var_29: var_1, var_30: var_2}
    var_37 = {var_29: var_4, var_30: var_5}
    var_38 = {var_29: var_12, var_30: var_13}
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.map_structure_zip(var_35, var_39)
    var_41 = [var_1, var_2]
    var_42 = (var_4, var_5)
    var_43 = {var_29: var_41, var_30: var_42}
    var_44 = [var_12, var_13]
    var_45 = 7
    var_46 = 8
    var_47 = (var_45, var_46)
    var_48 = {var_29: var_44, var_30: var_47}
    var_49 = [var_13, var_46]
    var_50 = 10
    var_51 = 12
    var_52 = (var_50, var_51)
    var_53 = {var_29: var_49, var_30: var_52}
    var_54 = lambda x, y: x + y
    var_55 = [var_43, var_48]
    var_56 = module_0.map_structure_zip(var_54, var_55)
    var_57 = [var_1, var_2, var_4]
    var_58 = module_0.no_map_instance(var_57)
    var_59 = lambda x, y: x + y
    var_60 = [var_58, var_58]
    var_61 = module_0.map_structure_zip(var_59, var_60)
    var_62 = [var_1, var_2, var_4]
    var_63 = lambda x, y: x + y
    var_64 = lambda x, y: x + y
    var_65 = 1
    var_66 = 2
    var_67 = {var_65, var_66}
    var_68 = 3
    var_69 = 4
    var_70 = {var_68, var_69}
    var_71 = [var_67, var_70]
    var_72 = module_0.map_structure_zip(var_64, var_71)



# Parsed testcases at query #2
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = (var_0, var_1)
    var_8 = (var_3, var_4)
    var_9 = (var_7, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = {var_10: var_3, var_11: var_4}
    var_14 = (var_12, var_13)
    var_15 = [var_0, var_1]
    var_16 = (var_3, var_4)
    var_17 = (var_15, var_16)
    var_18 = [var_0, var_1]
    var_19 = [var_3, var_4]
    var_20 = [var_18, var_19]
    var_21 = 5
    var_22 = 6
    var_23 = [var_21, var_22]
    var_24 = 7
    var_25 = 8
    var_26 = [var_24, var_25]
    var_27 = [var_23, var_26]
    var_28 = (var_20, var_27)
    var_29 = 'Point'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = 1
    var_34 = 2
    var_35 = {var_33, var_34}
    var_36 = 3
    var_37 = 4
    var_38 = {var_36, var_37}
    var_39 = (var_35, var_38)
    var_40 = 'Expected ValueError for unordered set'
    var_41 = AssertionError(var_40)
    var_42 = [var_40, var_41]
    var_43 = module_0.no_map_instance(var_42)
    var_44 = [var_36, var_37]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = (var_43, var_45)
    var_47 = [var_40, var_41]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_36, var_37]
    var_50 = (var_48, var_49)
    var_51 = [var_40, var_41]
    var_52 = [var_36, var_37]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = (var_51, var_53)



# Parsed testcases at query #3
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = [var_0]
    var_8 = [var_1]
    var_9 = [var_7, var_8]
    var_10 = [var_3]
    var_11 = [var_4]
    var_12 = [var_10, var_11]
    var_13 = (var_9, var_12)
    var_14 = (var_0, var_1)
    var_15 = (var_3, var_4)
    var_16 = (var_14, var_15)
    var_17 = (var_0,)
    var_18 = (var_1,)
    var_19 = (var_17, var_18)
    var_20 = (var_3,)
    var_21 = (var_4,)
    var_22 = (var_20, var_21)
    var_23 = (var_19, var_22)
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_0, var_25: var_1}
    var_27 = {var_24: var_3, var_25: var_4}
    var_28 = (var_26, var_27)
    var_29 = 'x'
    var_30 = {var_29: var_0}
    var_31 = {var_29: var_1}
    var_32 = {var_24: var_30, var_25: var_31}
    var_33 = {var_29: var_3}
    var_34 = {var_29: var_4}
    var_35 = {var_24: var_33, var_25: var_34}
    var_36 = (var_32, var_35)
    var_37 = [var_0, var_1]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = (var_38, var_38)
    var_40 = [var_0, var_1]
    var_41 = 1
    var_42 = 2
    var_43 = {var_41, var_42}
    var_44 = 3
    var_45 = 4
    var_46 = {var_44, var_45}
    var_47 = (var_43, var_46)



# Parsed testcases at query #4
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = [var_0, var_1]
    var_10 = [var_2, var_4]
    var_11 = [var_9, var_10]
    var_12 = [var_5, var_6]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = [var_11, var_16]
    var_18 = (var_0, var_1, var_2)
    var_19 = (var_4, var_5, var_6)
    var_20 = [var_18, var_19]
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = 'a'
    var_26 = 'b'
    var_27 = {var_25: var_0, var_26: var_1}
    var_28 = {var_25: var_2, var_26: var_4}
    var_29 = [var_27, var_28]
    var_30 = [var_0, var_1, var_2]
    var_31 = module_0.no_map_instance(var_30)
    var_32 = [var_4, var_5, var_6]
    var_33 = module_0.no_map_instance(var_32)
    var_34 = [var_31, var_33]
    var_35 = [var_3, var_7]
    var_36 = {var_0, var_1, var_2}
    var_37 = {var_4, var_5, var_6}
    var_38 = [var_36, var_37]
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #5
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = (var_0, var_1, var_2)
    var_10 = (var_4, var_5, var_6)
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = {var_12: var_2, var_13: var_4}
    var_16 = [var_14, var_15]
    var_17 = [var_0, var_1]
    var_18 = 'c'
    var_19 = 'd'
    var_20 = {var_18: var_2, var_19: var_4}
    var_21 = {var_12: var_17, var_13: var_20}
    var_22 = [var_5, var_6]
    var_23 = 7
    var_24 = 8
    var_25 = {var_18: var_23, var_19: var_24}
    var_26 = {var_12: var_22, var_13: var_25}
    var_27 = [var_21, var_26]
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = [var_4, var_5, var_6]
    var_31 = module_0.no_map_instance(var_30)
    var_32 = [var_29, var_31]
    var_33 = 0
    var_34 = [var_0, var_1]
    var_35 = (var_2, var_4)
    var_36 = {var_12: var_34, var_13: var_35}
    var_37 = [var_5, var_6]
    var_38 = (var_23, var_24)
    var_39 = {var_12: var_37, var_13: var_38}
    var_40 = [var_36, var_39]
    var_41 = 'All tests passed.'
    var_42 = print(var_41)



# Parsed testcases at query #6
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_5: var_2}
    var_9 = (var_1, var_8)
    var_10 = 4
    var_11 = [var_0, var_9, var_10]
    var_12 = [var_0, var_1, var_2]
    var_13 = module_0.no_map_instance(var_12)
    var_14 = [var_0, var_1, var_2]



# Parsed testcases at query #7
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = 4
    var_6 = [var_2, var_5]
    var_7 = [var_4, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = 'c'
    var_12 = {var_11: var_0}
    var_13 = 'd'
    var_14 = {var_13: var_1}
    var_15 = {var_8: var_12, var_9: var_14}
    var_16 = (var_0, var_1, var_2)
    var_17 = 'Point'
    var_18 = 'x'
    var_19 = 'y'
    var_20 = [var_18, var_19]
    var_21 = {var_0, var_1, var_2}
    var_22 = [var_0, var_1, var_2]
    var_23 = [var_0, var_1, var_2]
    var_24 = module_0.no_map_instance(var_23)



# Parsed testcases at query #8
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = [var_0, var_1]
    var_8 = [var_3, var_4]
    var_9 = [var_7, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = (var_9, var_16)
    var_18 = (var_0, var_1)
    var_19 = (var_3, var_4)
    var_20 = (var_18, var_19)
    var_21 = (var_0, var_1)
    var_22 = (var_3, var_4)
    var_23 = (var_21, var_22)
    var_24 = (var_10, var_11)
    var_25 = (var_13, var_14)
    var_26 = (var_24, var_25)
    var_27 = (var_23, var_26)
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_0, var_29: var_1}
    var_31 = {var_28: var_3, var_29: var_4}
    var_32 = (var_30, var_31)
    var_33 = 'x'
    var_34 = {var_33: var_0}
    var_35 = {var_33: var_1}
    var_36 = {var_28: var_34, var_29: var_35}
    var_37 = {var_33: var_3}
    var_38 = {var_33: var_4}
    var_39 = {var_28: var_37, var_29: var_38}
    var_40 = (var_36, var_39)
    var_41 = [var_0, var_1]
    var_42 = module_0.no_map_instance(var_41)
    var_43 = (var_42, var_42)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45]
    var_47 = (var_44, var_45)
    var_48 = (var_46, var_47)
    var_49 = 1
    var_50 = 2
    var_51 = {var_49, var_50}
    var_52 = 3
    var_53 = 4
    var_54 = {var_52, var_53}
    var_55 = (var_51, var_54)
    var_56 = 'All tests passed for map_structure_zip'
    var_57 = print(var_56)



# Parsed testcases at query #9
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = (var_0, var_1, var_2)
    var_10 = (var_4, var_5, var_6)
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = {var_12: var_2, var_13: var_4}
    var_16 = [var_14, var_15]
    var_17 = [var_0, var_1]
    var_18 = 'c'
    var_19 = {var_18: var_2}
    var_20 = {var_12: var_17, var_13: var_19}
    var_21 = [var_4, var_5]
    var_22 = {var_18: var_6}
    var_23 = {var_12: var_21, var_13: var_22}
    var_24 = [var_20, var_23]
    var_25 = [var_0, var_1, var_2]
    var_26 = module_0.no_map_instance(var_25)
    var_27 = [var_4, var_5, var_6]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = [var_26, var_28]
    var_30 = (var_0, var_1)
    var_31 = [var_2, var_4]
    var_32 = {var_12: var_30, var_13: var_31}
    var_33 = (var_5, var_6)
    var_34 = 7
    var_35 = 8
    var_36 = [var_34, var_35]
    var_37 = {var_12: var_33, var_13: var_36}
    var_38 = [var_32, var_37]



# Parsed testcases at query #10
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = lambda x, y: x + y
    var_9 = [var_3, var_7]
    var_10 = module_0.map_structure_zip(var_8, var_9)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = lambda x, y: x + y
    var_20 = [var_13, var_18]
    var_21 = module_0.map_structure_zip(var_19, var_20)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = {var_22: var_0, var_23: var_1}
    var_25 = {var_22: var_2, var_23: var_4}
    var_26 = lambda x, y: x + y
    var_27 = [var_24, var_25]
    var_28 = module_0.map_structure_zip(var_26, var_27)
    var_29 = (var_0, var_1, var_2)
    var_30 = (var_4, var_5, var_6)
    var_31 = lambda x, y: x + y
    var_32 = [var_29, var_30]
    var_33 = module_0.map_structure_zip(var_31, var_32)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = [var_0, var_1, var_2]
    var_40 = module_0.no_map_instance(var_39)
    var_41 = [var_4, var_5, var_6]
    var_42 = module_0.no_map_instance(var_41)
    var_43 = lambda x, y: x + y
    var_44 = [var_40, var_42]
    var_45 = module_0.map_structure_zip(var_43, var_44)
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = {var_22: var_1}
    var_48 = (var_2,)
    var_49 = [var_0, var_47, var_48]
    var_50 = {var_22: var_5}
    var_51 = (var_6,)
    var_52 = [var_4, var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = [var_49, var_52]
    var_55 = module_0.map_structure_zip(var_53, var_54)



# Parsed testcases at query #11
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_0, var_1, var_2}
    var_9 = {var_5: var_2}
    var_10 = (var_1, var_9)
    var_11 = 4
    var_12 = [var_0, var_10, var_11]
    var_13 = [var_0, var_1, var_2]
    var_14 = module_0.no_map_instance(var_13)



# Parsed testcases at query #12
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = {var_13: var_2, var_14: var_0}
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = lambda x: x ** var_0
    var_22 = {var_0, var_3}
    var_23 = module_0.map_structure(var_21, var_22)
    var_24 = [var_2, var_0, var_3]
    var_25 = module_0.no_map_instance(var_24)
    var_26 = lambda x: x * var_0
    var_27 = module_0.map_structure(var_26, var_25)
    var_28 = [var_2, var_0, var_3]
    var_29 = lambda x: x * var_0



# Parsed testcases at query #13
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = (var_0, var_1)
    var_8 = (var_3, var_4)
    var_9 = (var_7, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = {var_10: var_3, var_11: var_4}
    var_14 = (var_12, var_13)
    var_15 = {var_10: var_0}
    var_16 = {var_11: var_1}
    var_17 = [var_15, var_16]
    var_18 = {var_10: var_3}
    var_19 = {var_11: var_4}
    var_20 = [var_18, var_19]
    var_21 = (var_17, var_20)
    var_22 = 1
    var_23 = 2
    var_24 = {var_22, var_23}
    var_25 = 3
    var_26 = 4
    var_27 = {var_25, var_26}
    var_28 = (var_24, var_27)
    var_29 = [var_22, var_23]
    var_30 = module_0.no_map_instance(var_29)
    var_31 = [var_25, var_26]
    var_32 = (var_30, var_31)
    var_33 = [var_25, var_26]
    var_34 = [var_22, var_23]
    var_35 = [var_25, var_26]
    var_36 = (var_34, var_35)
    var_37 = [var_22, var_23]
    var_38 = [var_25, var_26]



# Parsed testcases at query #14
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = (var_0, var_1)
    var_8 = (var_3, var_4)
    var_9 = (var_7, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = {var_10: var_3, var_11: var_4}
    var_14 = (var_12, var_13)
    var_15 = {var_10: var_0}
    var_16 = {var_11: var_1}
    var_17 = [var_15, var_16]
    var_18 = {var_10: var_3}
    var_19 = {var_11: var_4}
    var_20 = [var_18, var_19]
    var_21 = (var_17, var_20)
    var_22 = 1
    var_23 = 2
    var_24 = {var_22, var_23}
    var_25 = 3
    var_26 = 4
    var_27 = {var_25, var_26}
    var_28 = (var_24, var_27)
    var_29 = [var_22, var_23]
    var_30 = module_0.no_map_instance(var_29)
    var_31 = (var_30, var_30)



# Parsed testcases at query #15
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = (var_0, var_1, var_2)
    var_10 = (var_4, var_5, var_6)
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = {var_12: var_2, var_13: var_4}
    var_16 = [var_14, var_15]
    var_17 = [var_0, var_1]
    var_18 = (var_2, var_4)
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = [var_5, var_6]
    var_21 = 7
    var_22 = 8
    var_23 = (var_21, var_22)
    var_24 = {var_12: var_20, var_13: var_23}
    var_25 = [var_19, var_24]
    var_26 = [var_0, var_1, var_2]
    var_27 = module_0.no_map_instance(var_26)
    var_28 = [var_4, var_5, var_6]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = [var_27, var_29]



# Parsed testcases at query #16
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = [var_0]
    var_8 = [var_1]
    var_9 = [var_7, var_8]
    var_10 = [var_3]
    var_11 = [var_4]
    var_12 = [var_10, var_11]
    var_13 = (var_9, var_12)
    var_14 = (var_0, var_1)
    var_15 = (var_3, var_4)
    var_16 = (var_14, var_15)
    var_17 = (var_0,)
    var_18 = (var_1,)
    var_19 = (var_17, var_18)
    var_20 = (var_3,)
    var_21 = (var_4,)
    var_22 = (var_20, var_21)
    var_23 = (var_19, var_22)
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_0, var_25: var_1}
    var_27 = {var_24: var_3, var_25: var_4}
    var_28 = (var_26, var_27)
    var_29 = [var_0, var_1]
    var_30 = (var_3, var_4)
    var_31 = {var_24: var_29, var_25: var_30}
    var_32 = 5
    var_33 = 6
    var_34 = [var_32, var_33]
    var_35 = 7
    var_36 = 8
    var_37 = (var_35, var_36)
    var_38 = {var_24: var_34, var_25: var_37}
    var_39 = [var_33, var_36]
    var_40 = 10
    var_41 = 12
    var_42 = (var_40, var_41)
    var_43 = {var_24: var_39, var_25: var_42}
    var_44 = (var_31, var_38)
    var_45 = [var_0, var_1]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = (var_46, var_46)
    var_48 = [var_0, var_1]
    var_49 = 1
    var_50 = 2
    var_51 = {var_49, var_50}
    var_52 = 3
    var_53 = 4
    var_54 = {var_52, var_53}
    var_55 = (var_51, var_54)
    var_56 = 'All tests passed!'
    var_57 = print(var_56)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_5: var_2}
    var_9 = (var_1, var_8)
    var_10 = [var_0, var_9]



# Parsed testcases at query #18
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x * var_2
    var_7 = [var_0, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = 10
    var_19 = lambda x: x + var_18
    var_20 = {var_13: var_0, var_14: var_2}
    var_21 = module_0.map_structure(var_19, var_20)
    var_22 = lambda x: x * var_2
    var_23 = {var_0, var_2, var_3}
    var_24 = module_0.map_structure(var_22, var_23)
    var_25 = [var_0, var_2, var_3]
    var_26 = module_0.no_map_instance(var_25)
    var_27 = lambda x: x + var_0
    var_28 = module_0.map_structure(var_27, var_26)
    var_29 = [var_0, var_2, var_3]
    var_30 = lambda x: x + var_0



# Parsed testcases at query #19
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_1, var_2]
    var_5 = 4
    var_6 = [var_0, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {var_7: var_0, var_8: var_1}
    var_10 = 'c'
    var_11 = 'd'
    var_12 = {var_10: var_1, var_11: var_2}
    var_13 = {var_7: var_0, var_8: var_12}
    var_14 = (var_0, var_1, var_2)
    var_15 = (var_1, var_2)
    var_16 = (var_0, var_15, var_5)
    var_17 = {var_0, var_1, var_2}
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.no_map_instance(var_18)
    var_20 = [var_0, var_1, var_2]



# Parsed testcases at query #20
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = [var_0, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = [var_4, var_11]
    var_13 = [var_8, var_12]
    var_14 = (var_0, var_1)
    var_15 = (var_3, var_4)
    var_16 = [var_14, var_15]
    var_17 = (var_1, var_3)
    var_18 = (var_0, var_17)
    var_19 = (var_9, var_10)
    var_20 = (var_4, var_19)
    var_21 = [var_18, var_20]
    var_22 = 'a'
    var_23 = 'b'
    var_24 = {var_22: var_0, var_23: var_1}
    var_25 = {var_22: var_3, var_23: var_4}
    var_26 = [var_24, var_25]
    var_27 = 'c'
    var_28 = {var_27: var_1}
    var_29 = {var_22: var_0, var_23: var_28}
    var_30 = {var_27: var_4}
    var_31 = {var_22: var_3, var_23: var_30}
    var_32 = [var_29, var_31]
    var_33 = 1
    var_34 = 2
    var_35 = {var_33, var_34}
    var_36 = 3
    var_37 = 4
    var_38 = {var_36, var_37}
    var_39 = [var_35, var_38]
    var_40 = [var_33, var_34]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = [var_36, var_37]
    var_43 = [var_41, var_42]
    var_44 = [var_36, var_37]
    var_45 = {var_22: var_33, var_23: var_34}
    var_46 = module_0.no_map_instance(var_45)
    var_47 = {var_22: var_36, var_23: var_37}
    var_48 = [var_46, var_47]
    var_49 = {var_22: var_36, var_23: var_37}



# Parsed testcases at query #21
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = [var_0, var_1]
    var_10 = [var_2, var_4]
    var_11 = [var_9, var_10]
    var_12 = [var_5, var_6]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = [var_11, var_16]
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_0, var_19: var_1}
    var_21 = {var_18: var_2, var_19: var_4}
    var_22 = [var_20, var_21]
    var_23 = (var_0, var_1)
    var_24 = (var_2, var_4)
    var_25 = [var_23, var_24]
    var_26 = [var_0, var_1]
    var_27 = (var_2, var_4)
    var_28 = {var_18: var_26, var_19: var_27}
    var_29 = [var_5, var_6]
    var_30 = (var_13, var_14)
    var_31 = {var_18: var_29, var_19: var_30}
    var_32 = [var_28, var_31]
    var_33 = [var_0, var_1]
    var_34 = module_0.no_map_instance(var_33)
    var_35 = [var_2, var_4]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = [var_34, var_36]
    var_38 = 'All tests passed.'
    var_39 = print(var_38)



# Parsed testcases at query #22
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = (var_0, var_1)
    var_8 = (var_3, var_4)
    var_9 = (var_7, var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = {var_10: var_3, var_11: var_4}
    var_14 = (var_12, var_13)
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = 6
    var_20 = [var_0, var_1]
    var_21 = module_0.no_map_instance(var_20)
    var_22 = (var_21, var_21)
    var_23 = 1
    var_24 = 2
    var_25 = {var_23, var_24}
    var_26 = 3
    var_27 = 4
    var_28 = {var_26, var_27}
    var_29 = (var_25, var_28)



# Parsed testcases at query #23
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = (var_0, var_1, var_2)
    var_10 = (var_4, var_5, var_6)
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = {var_12: var_2, var_13: var_4}
    var_16 = [var_14, var_15]
    var_17 = [var_0, var_1]
    var_18 = (var_2, var_4)
    var_19 = {var_12: var_17, var_13: var_18}
    var_20 = [var_5, var_6]
    var_21 = 7
    var_22 = 8
    var_23 = (var_21, var_22)
    var_24 = {var_12: var_20, var_13: var_23}
    var_25 = [var_19, var_24]
    var_26 = [var_0, var_1, var_2]
    var_27 = module_0.no_map_instance(var_26)
    var_28 = [var_4, var_5, var_6]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = [var_27, var_29]
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = {var_31, var_32, var_33}
    var_35 = 4
    var_36 = 5
    var_37 = 6
    var_38 = {var_35, var_36, var_37}
    var_39 = [var_34, var_38]
    var_40 = 'All tests passed!'
    var_41 = print(var_40)



# Parsed testcases at query #24
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = (var_0, var_1, var_2)
    var_10 = (var_4, var_5, var_6)
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = {var_12: var_2, var_13: var_4}
    var_16 = [var_14, var_15]
    var_17 = [var_0, var_1]
    var_18 = 'x'
    var_19 = 'y'
    var_20 = {var_18: var_2, var_19: var_4}
    var_21 = {var_12: var_17, var_13: var_20}
    var_22 = [var_5, var_6]
    var_23 = 7
    var_24 = 8
    var_25 = {var_18: var_23, var_19: var_24}
    var_26 = {var_12: var_22, var_13: var_25}
    var_27 = [var_21, var_26]
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = [var_29, var_29]
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = {var_31, var_32, var_33}
    var_35 = 4
    var_36 = 5
    var_37 = 6
    var_38 = {var_35, var_36, var_37}
    var_39 = [var_34, var_38]



# Parsed testcases at query #25
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_0
    var_7 = [var_0, var_2]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x + var_0
    var_13 = (var_0, var_2, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = lambda x: x + var_0
    var_16 = (var_0, var_2)
    var_17 = (var_3, var_8)
    var_18 = (var_16, var_17)
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x + var_0
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_2}
    var_24 = module_0.map_structure(var_20, var_23)
    var_25 = lambda x: x + var_0
    var_26 = 'c'
    var_27 = {var_26: var_0}
    var_28 = {var_21: var_27, var_22: var_2}
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = lambda x: x + var_0
    var_31 = {var_0, var_2, var_3}
    var_32 = module_0.map_structure(var_30, var_31)
    var_33 = [var_0, var_2, var_3]
    var_34 = module_0.no_map_instance(var_33)
    var_35 = lambda x: x + var_0
    var_36 = [var_0, var_2, var_3]
    var_37 = module_0.map_structure(var_35, var_36)



# Parsed testcases at query #26
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = [var_0, var_1]
    var_8 = [var_3, var_4]
    var_9 = [var_7, var_8]
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = (var_9, var_16)
    var_18 = (var_0, var_1)
    var_19 = (var_3, var_4)
    var_20 = (var_18, var_19)
    var_21 = (var_0, var_1)
    var_22 = (var_3, var_4)
    var_23 = (var_21, var_22)
    var_24 = (var_10, var_11)
    var_25 = (var_13, var_14)
    var_26 = (var_24, var_25)
    var_27 = (var_23, var_26)
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_0, var_29: var_1}
    var_31 = {var_28: var_3, var_29: var_4}
    var_32 = (var_30, var_31)
    var_33 = [var_0, var_1]
    var_34 = [var_3, var_4]
    var_35 = {var_28: var_33, var_29: var_34}
    var_36 = [var_10, var_11]
    var_37 = [var_13, var_14]
    var_38 = {var_28: var_36, var_29: var_37}
    var_39 = (var_35, var_38)
    var_40 = [var_0, var_1, var_3]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = (var_41, var_41)
    var_43 = [var_0, var_1, var_3]
    var_44 = 1
    var_45 = 2
    var_46 = {var_44, var_45}
    var_47 = 3
    var_48 = 4
    var_49 = {var_47, var_48}
    var_50 = (var_46, var_49)



# Parsed testcases at query #27
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = 4
    var_6 = [var_2, var_5]
    var_7 = [var_4, var_6]
    var_8 = (var_0, var_1, var_2)
    var_9 = (var_0, var_1)
    var_10 = (var_2, var_5)
    var_11 = (var_9, var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = 'c'
    var_16 = {var_15: var_0}
    var_17 = {var_12: var_16, var_13: var_1}
    var_18 = {var_0, var_1, var_2}
    var_19 = {var_0, var_1}
    var_20 = {var_2, var_5}
    var_21 = [var_19, var_20]
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.no_map_instance(var_22)
    var_24 = [var_0, var_1, var_2]



# Parsed testcases at query #28
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_5: var_0, var_6: var_1}
    var_9 = 'c'
    var_10 = {var_9: var_2}
    var_11 = [var_8, var_10]
    var_12 = [var_0, var_1, var_2]
    var_13 = [var_0, var_1, var_2]
    var_14 = [var_0, var_1, var_2]
    var_15 = module_0.no_map_instance(var_14)



# Parsed testcases at query #29
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = lambda x, y: x + y
    var_10 = (var_1, var_2)
    var_11 = (var_4, var_5)
    var_12 = [var_10, var_11]
    var_13 = module_0.map_structure_zip(var_9, var_12)
    var_14 = lambda x, y: x + y
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_1, var_16: var_2}
    var_18 = {var_15: var_4, var_16: var_5}
    var_19 = [var_17, var_18]
    var_20 = module_0.map_structure_zip(var_14, var_19)
    var_21 = lambda x, y: x + y
    var_22 = [var_1]
    var_23 = [var_2]
    var_24 = [var_22, var_23]
    var_25 = [var_4]
    var_26 = [var_5]
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = module_0.map_structure_zip(var_21, var_28)
    var_30 = [var_1, var_2]
    var_31 = module_0.no_map_instance(var_30)
    var_32 = lambda x, y: x + y
    var_33 = [var_31, var_31]
    var_34 = module_0.map_structure_zip(var_32, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1]
    var_37 = {var_15: var_2}
    var_38 = (var_36, var_37)
    var_39 = [var_4]
    var_40 = {var_15: var_5}
    var_41 = (var_39, var_40)
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = lambda x, y: x + y
    var_45 = 1
    var_46 = 2
    var_47 = {var_45, var_46}
    var_48 = 3
    var_49 = 4
    var_50 = {var_48, var_49}
    var_51 = [var_47, var_50]
    var_52 = module_0.map_structure_zip(var_44, var_51)



# Parsed testcases at query #30
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = {var_0, var_1, var_2}
    var_9 = [var_0, var_1]
    var_10 = 4
    var_11 = [var_2, var_10]
    var_12 = [var_9, var_11]
    var_13 = {var_5: var_0, var_6: var_1}
    var_14 = 'c'
    var_15 = 'd'
    var_16 = {var_14: var_2, var_15: var_10}
    var_17 = (var_13, var_16)
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.no_map_instance(var_18)
    var_20 = [var_0, var_1, var_2]



