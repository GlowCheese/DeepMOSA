####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0}
    var_45 = {var_38: var_1}
    var_46 = [var_44, var_45]
    var_47 = {var_37: var_2}
    var_48 = {var_38: var_4}
    var_49 = [var_47, var_48]
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = type(var_53)
    var_60 = module_0.register_no_map_class(var_59)
    var_61 = [var_4, var_5, var_6]
    var_62 = [var_53, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)
    var_65 = {var_0, var_1}
    var_66 = {var_2, var_4}
    var_67 = [var_65, var_66]
    var_68 = lambda x, y: x + y
    var_69 = module_0.map_structure_zip(var_68, var_67)



# Parsed testcases at query #2
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0, var_39: var_1}
    var_46 = 'c'
    var_47 = {var_46: var_2}
    var_48 = [var_45, var_47]
    var_49 = {var_38: var_4, var_39: var_5}
    var_50 = {var_46: var_6}
    var_51 = [var_49, var_50]
    var_52 = [var_48, var_51]
    var_53 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = type(var_55)
    var_62 = module_0.register_no_map_class(var_61)
    var_63 = [var_4, var_5, var_6]
    var_64 = [var_55, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)
    var_67 = {var_0, var_1}
    var_68 = {var_2, var_4}
    var_69 = [var_67, var_68]
    var_70 = lambda x, y: x + y
    var_71 = module_0.map_structure_zip(var_70, var_69)



# Parsed testcases at query #3
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
    var_11 = (var_0, var_1, var_2)
    var_12 = (var_4, var_5, var_6)
    var_13 = lambda x, y: x + y
    var_14 = [var_11, var_12]
    var_15 = module_0.map_structure_zip(var_13, var_14)
    var_16 = 'Point'
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = lambda x, y: x + y
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = {var_21: var_2, var_22: var_4}
    var_25 = lambda x, y: x + y
    var_26 = [var_23, var_24]
    var_27 = module_0.map_structure_zip(var_25, var_26)
    var_28 = [var_0, var_1]
    var_29 = (var_2, var_4)
    var_30 = {var_21: var_28, var_22: var_29}
    var_31 = [var_5, var_6]
    var_32 = 7
    var_33 = 8
    var_34 = (var_32, var_33)
    var_35 = {var_21: var_31, var_22: var_34}
    var_36 = lambda x, y: x + y
    var_37 = [var_30, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1, var_2]
    var_40 = module_0.no_map_instance(var_39)
    var_41 = lambda x, y: x + y
    var_42 = [var_4, var_5, var_6]
    var_43 = [var_40, var_42]
    var_44 = module_0.map_structure_zip(var_41, var_43)
    var_45 = [var_0, var_1, var_2]
    var_46 = type(var_45)
    var_47 = module_0.register_no_map_class(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_0, var_1, var_2]
    var_50 = [var_4, var_5, var_6]
    var_51 = [var_49, var_50]
    var_52 = module_0.map_structure_zip(var_48, var_51)
    var_53 = lambda x, y: x + y
    var_54 = 1
    var_55 = 2
    var_56 = {var_54, var_55}
    var_57 = 3
    var_58 = 4
    var_59 = {var_57, var_58}
    var_60 = [var_56, var_59]
    var_61 = module_0.map_structure_zip(var_53, var_60)



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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_1}
    var_45 = [var_0, var_44]
    var_46 = {var_37: var_4}
    var_47 = [var_2, var_46]
    var_48 = [var_45, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = module_0.no_map_instance(var_51)
    var_54 = module_0.no_map_instance(var_52)
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = [var_0, var_1, var_2]
    var_59 = [var_4, var_5, var_6]
    var_60 = [var_58, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)
    var_63 = {var_0, var_1}
    var_64 = {var_2, var_4}
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)



# Parsed testcases at query #5
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = 'd'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = {var_19: var_2, var_20: var_26}
    var_28 = module_0.map_structure(var_23, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = lambda x: x * var_0
    var_33 = (var_0, var_3)
    var_34 = {var_19: var_8}
    var_35 = [var_2, var_33, var_34]
    var_36 = module_0.map_structure(var_32, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = type(var_38)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_38)
    var_45 = 'Point'
    var_46 = 'x'
    var_47 = 'y'
    var_48 = [var_46, var_47]
    var_49 = lambda x: x * var_0
    var_50 = lambda x: x * var_0
    var_51 = 6



# Parsed testcases at query #6
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = 'c'
    var_26 = {var_25: var_0}
    var_27 = {var_20: var_2, var_21: var_26}
    var_28 = module_0.map_structure(var_24, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.no_map_instance(var_32)
    var_34 = lambda x: x * var_0
    var_35 = module_0.map_structure(var_34, var_33)
    var_36 = lambda x: x * var_0
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.map_structure(var_36, var_37)
    var_39 = (var_0, var_3)
    var_40 = {var_20: var_39}
    var_41 = 5
    var_42 = {var_8, var_41}
    var_43 = [var_2, var_40, var_42]
    var_44 = lambda x: x + var_2
    var_45 = module_0.map_structure(var_44, var_43)



# Parsed testcases at query #7
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
    var_7 = (var_2, var_0, var_3)
    var_8 = module_0.map_structure(var_6, var_7)
    var_9 = 'Point'
    var_10 = 'x'
    var_11 = 'y'
    var_12 = [var_10, var_11]
    var_13 = lambda x: x * var_0
    var_14 = 4
    var_15 = lambda x: x * var_0
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = {var_3, var_14}
    var_24 = (var_0, var_23)
    var_25 = 5
    var_26 = 6
    var_27 = {var_16: var_26}
    var_28 = [var_25, var_27]
    var_29 = [var_2, var_24, var_28]
    var_30 = 8
    var_31 = {var_26, var_30}
    var_32 = (var_14, var_31)
    var_33 = 10
    var_34 = 12
    var_35 = {var_16: var_34}
    var_36 = [var_33, var_35]
    var_37 = [var_0, var_32, var_36]
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_29)
    var_40 = [var_2, var_0, var_3]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = type(var_40)
    var_45 = module_0.register_no_map_class(var_44)
    var_46 = lambda x: x * var_0
    var_47 = module_0.map_structure(var_46, var_40)
    var_48 = lambda x: x * var_0
    var_49 = module_0.map_structure(var_48, var_25)
    assert var_49 == 10



# Parsed testcases at query #8
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = (var_0, var_3)
    var_28 = {var_20: var_27}
    var_29 = 5
    var_30 = {var_8, var_29}
    var_31 = [var_2, var_28, var_30]
    var_32 = 6
    var_33 = (var_8, var_32)
    var_34 = {var_20: var_33}
    var_35 = 8
    var_36 = 10
    var_37 = {var_35, var_36}
    var_38 = [var_0, var_34, var_37]
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_31)
    var_41 = [var_2, var_0, var_3]
    var_42 = module_0.no_map_instance(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_42)
    var_45 = type(var_41)
    var_46 = module_0.register_no_map_class(var_45)
    var_47 = lambda x: x * var_0
    var_48 = module_0.map_structure(var_47, var_41)
    var_49 = lambda x: x * var_0
    var_50 = module_0.map_structure(var_49, var_29)
    assert var_50 == 10



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
    var_9 = [var_0, var_1]
    var_10 = [var_2, var_4]
    var_11 = [var_9, var_10]
    var_12 = [var_5, var_6]
    var_13 = 7
    var_14 = 8
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = [var_11, var_16]
    var_18 = (var_0, var_1)
    var_19 = (var_2, var_4)
    var_20 = (var_18, var_19)
    var_21 = (var_5, var_6)
    var_22 = (var_13, var_14)
    var_23 = (var_21, var_22)
    var_24 = [var_20, var_23]
    var_25 = 'Point'
    var_26 = 'x'
    var_27 = 'y'
    var_28 = [var_26, var_27]
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_0, var_30: var_1}
    var_32 = {var_29: var_2, var_30: var_4}
    var_33 = [var_31, var_32]
    var_34 = {var_29: var_0, var_30: var_1}
    var_35 = {var_29: var_2, var_30: var_4}
    var_36 = [var_34, var_35]
    var_37 = {var_29: var_5, var_30: var_6}
    var_38 = {var_29: var_13, var_30: var_14}
    var_39 = [var_37, var_38]
    var_40 = [var_36, var_39]
    var_41 = [var_0, var_1, var_2]
    var_42 = module_0.no_map_instance(var_41)
    var_43 = [var_4, var_5, var_6]
    var_44 = [var_42, var_43]
    var_45 = type(var_41)
    var_46 = module_0.register_no_map_class(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_41, var_47]
    var_49 = {var_0, var_1}
    var_50 = {var_2, var_4}
    var_51 = [var_49, var_50]



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
    var_8 = [var_3, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = [var_4, var_5, var_6]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_52)
    var_59 = {var_0, var_1, var_2}
    var_60 = {var_4, var_5, var_6}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #11
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #12
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_48)
    var_55 = {var_0, var_1}
    var_56 = {var_2, var_4}
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)



# Parsed testcases at query #13
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = type(var_45)
    var_52 = module_0.register_no_map_class(var_51)
    var_53 = [var_4, var_5, var_6]
    var_54 = [var_45, var_53]
    var_55 = lambda x, y: x + y
    var_56 = module_0.map_structure_zip(var_55, var_54)
    var_57 = {var_0, var_1}
    var_58 = {var_2, var_4}
    var_59 = [var_57, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)



# Parsed testcases at query #14
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x * y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #16
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = [var_4, var_5, var_6]
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = {var_0, var_1}
    var_59 = {var_2, var_4}
    var_60 = [var_58, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)



# Parsed testcases at query #17
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = 4
    var_31 = 5
    var_32 = [var_30, var_31]
    var_33 = {var_18: var_3, var_19: var_32}
    var_34 = (var_0, var_33)
    var_35 = [var_2, var_34]
    var_36 = 6
    var_37 = 8
    var_38 = 10
    var_39 = [var_37, var_38]
    var_40 = {var_18: var_36, var_19: var_39}
    var_41 = (var_30, var_40)
    var_42 = [var_0, var_41]
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_35)
    var_45 = [var_2, var_0, var_3]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = lambda x: x * var_0
    var_48 = module_0.map_structure(var_47, var_46)
    var_49 = type(var_46)
    var_50 = module_0.register_no_map_class(var_49)
    var_51 = lambda x: x * var_0
    var_52 = module_0.map_structure(var_51, var_46)
    var_53 = 'Point'
    var_54 = 'x'
    var_55 = 'y'
    var_56 = [var_54, var_55]
    var_57 = lambda x: x * var_0
    var_58 = lambda x: x * var_0
    var_59 = lambda x: x * var_0
    var_60 = []
    var_61 = module_0.map_structure(var_59, var_60)
    var_62 = lambda x: x * var_0
    var_63 = ()
    var_64 = module_0.map_structure(var_62, var_63)
    var_65 = lambda x: x * var_0
    var_66 = {}
    var_67 = module_0.map_structure(var_65, var_66)
    var_68 = lambda x: x * var_0
    var_69 = set()
    var_70 = module_0.map_structure(var_68, var_69)
    var_71 = set()



# Parsed testcases at query #18
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = (var_0, var_3)
    var_32 = 4
    var_33 = {var_18: var_32}
    var_34 = [var_2, var_31, var_33]
    var_35 = module_0.map_structure(var_30, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = type(var_37)
    var_41 = module_0.register_no_map_class(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_37)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda x: x * var_0
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #19
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = 'd'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = {var_19: var_2, var_20: var_26}
    var_28 = module_0.map_structure(var_23, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = lambda x: x * var_0
    var_33 = (var_0, var_3)
    var_34 = {var_19: var_8}
    var_35 = [var_2, var_33, var_34]
    var_36 = module_0.map_structure(var_32, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = type(var_38)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_38)
    var_45 = 'Point'
    var_46 = 'x'
    var_47 = 'y'
    var_48 = [var_46, var_47]
    var_49 = lambda x: x * var_0
    var_50 = lambda x: x * var_0
    var_51 = 6



# Parsed testcases at query #20
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



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
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = lambda x, y: x * y
    var_25 = [var_22, var_23]
    var_26 = module_0.map_structure_zip(var_24, var_25)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_0, var_33: var_1}
    var_35 = {var_32: var_2, var_33: var_4}
    var_36 = lambda x, y: x + y
    var_37 = [var_34, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1]
    var_40 = (var_2, var_4)
    var_41 = {var_32: var_39, var_33: var_40}
    var_42 = [var_5, var_6]
    var_43 = (var_15, var_16)
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = lambda x, y: x + y
    var_46 = [var_41, var_44]
    var_47 = module_0.map_structure_zip(var_45, var_46)
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x, y: x + y
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_49, var_51]
    var_53 = module_0.map_structure_zip(var_50, var_52)
    var_54 = [var_0, var_1, var_2]
    var_55 = type(var_54)
    var_56 = module_0.register_no_map_class(var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = lambda x, y: x + y
    var_59 = [var_4, var_5, var_6]
    var_60 = [var_57, var_59]
    var_61 = module_0.map_structure_zip(var_58, var_60)
    var_62 = {var_0, var_1, var_2}
    var_63 = {var_4, var_5, var_6}
    var_64 = lambda x, y: x + y
    var_65 = [var_62, var_63]
    var_66 = module_0.map_structure_zip(var_64, var_65)



# Parsed testcases at query #22
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_1}
    var_45 = [var_0, var_44]
    var_46 = {var_37: var_4}
    var_47 = [var_2, var_46]
    var_48 = [var_45, var_47]
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = type(var_49)
    var_56 = module_0.register_no_map_class(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_49, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = {var_0, var_1}
    var_62 = {var_2, var_4}
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)



# Parsed testcases at query #23
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = [var_0, var_1]
    var_7 = 4
    var_8 = [var_2, var_7]
    var_9 = [var_6, var_8]
    var_10 = lambda x: x * var_1
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = lambda x: x * var_1
    var_20 = module_0.map_structure(var_19, var_12)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = lambda x: x * var_1
    var_25 = module_0.map_structure(var_24, var_23)
    var_26 = {var_0, var_1, var_2}
    var_27 = lambda x: x * var_1
    var_28 = module_0.map_structure(var_27, var_26)
    var_29 = [var_0, var_1, var_2]
    var_30 = module_0.no_map_instance(var_29)
    var_31 = lambda x: x * var_1
    var_32 = module_0.map_structure(var_31, var_30)
    var_33 = [var_0, var_1, var_2]
    var_34 = lambda x: x * var_1
    var_35 = module_0.map_structure(var_34, var_33)



# Parsed testcases at query #24
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = 'c'
    var_26 = {var_25: var_0}
    var_27 = {var_20: var_2, var_21: var_26}
    var_28 = module_0.map_structure(var_24, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.no_map_instance(var_32)
    var_34 = lambda x: x * var_0
    var_35 = module_0.map_structure(var_34, var_33)
    var_36 = lambda x: x * var_0
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.map_structure(var_36, var_37)
    var_39 = lambda x: x * var_0
    var_40 = 5
    var_41 = module_0.map_structure(var_39, var_40)
    assert var_41 == 10



# Parsed testcases at query #25
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #26
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
    var_7 = [var_2, var_3]
    var_8 = [var_0, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x + var_0
    var_11 = (var_0, var_2, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x + var_0
    var_14 = (var_2, var_3)
    var_15 = (var_0, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x + var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_0, var_19: var_2}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x + var_0
    var_23 = 'c'
    var_24 = {var_23: var_2}
    var_25 = {var_18: var_0, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x + var_0
    var_28 = {var_0, var_2, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x + var_0
    var_31 = {var_18: var_3}
    var_32 = (var_2, var_31)
    var_33 = [var_0, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_0, var_2, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x + var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_36)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x + var_0
    var_42 = module_0.map_structure(var_41, var_36)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x + var_0



# Parsed testcases at query #27
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = (var_0, var_3)
    var_33 = {var_19: var_8}
    var_34 = [var_2, var_32, var_33]
    var_35 = module_0.map_structure(var_31, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = type(var_37)
    var_41 = module_0.register_no_map_class(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_37)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda x: x * var_0
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #28
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
    var_11 = (var_0, var_1, var_2)
    var_12 = (var_4, var_5, var_6)
    var_13 = lambda x, y: x + y
    var_14 = [var_11, var_12]
    var_15 = module_0.map_structure_zip(var_13, var_14)
    var_16 = 'Point'
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = lambda x, y: x + y
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = {var_21: var_2, var_22: var_4}
    var_25 = lambda x, y: x + y
    var_26 = [var_23, var_24]
    var_27 = module_0.map_structure_zip(var_25, var_26)
    var_28 = [var_0, var_1]
    var_29 = (var_2, var_4)
    var_30 = {var_21: var_28, var_22: var_29}
    var_31 = [var_5, var_6]
    var_32 = 7
    var_33 = 8
    var_34 = (var_32, var_33)
    var_35 = {var_21: var_31, var_22: var_34}
    var_36 = lambda x, y: x + y
    var_37 = [var_30, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1, var_2]
    var_40 = module_0.no_map_instance(var_39)
    var_41 = lambda x, y: x + y
    var_42 = [var_4, var_5, var_6]
    var_43 = [var_40, var_42]
    var_44 = module_0.map_structure_zip(var_41, var_43)
    var_45 = lambda x, y: x + y
    var_46 = [var_0, var_1, var_2]
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = module_0.map_structure_zip(var_45, var_48)



# Parsed testcases at query #29
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
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)
    var_9 = [var_0, var_1]
    var_10 = [var_3, var_4]
    var_11 = [var_9, var_10]
    var_12 = 5
    var_13 = 6
    var_14 = [var_12, var_13]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_11, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_3, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = (var_0, var_1)
    var_28 = (var_3, var_4)
    var_29 = (var_27, var_28)
    var_30 = (var_12, var_13)
    var_31 = (var_15, var_16)
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = lambda x, y: x + y
    var_35 = module_0.map_structure_zip(var_34, var_33)
    var_36 = 'Point'
    var_37 = 'x'
    var_38 = 'y'
    var_39 = [var_37, var_38]
    var_40 = lambda x, y: x + y
    var_41 = module_0.map_structure_zip(var_40, var_33)
    var_42 = 'a'
    var_43 = 'b'
    var_44 = {var_42: var_0, var_43: var_1}
    var_45 = {var_42: var_3, var_43: var_4}
    var_46 = [var_44, var_45]
    var_47 = lambda x, y: x + y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = 'c'
    var_50 = 'd'
    var_51 = {var_49: var_0, var_50: var_1}
    var_52 = {var_42: var_51, var_43: var_3}
    var_53 = {var_49: var_4, var_50: var_12}
    var_54 = {var_42: var_53, var_43: var_13}
    var_55 = [var_52, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = [var_0, var_1]
    var_59 = (var_3, var_4)
    var_60 = {var_42: var_58, var_43: var_59}
    var_61 = [var_12, var_13]
    var_62 = (var_15, var_16)
    var_63 = {var_42: var_61, var_43: var_62}
    var_64 = [var_60, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)
    var_67 = {var_0, var_1}
    var_68 = {var_3, var_4}
    var_69 = [var_67, var_68]
    var_70 = lambda x, y: x + y
    var_71 = module_0.map_structure_zip(var_70, var_69)
    var_72 = [var_70, var_71]
    var_73 = [var_3, var_4]
    var_74 = lambda x, y: x + y
    var_75 = module_0.map_structure_zip(var_74, var_69)
    var_76 = [var_70, var_71]
    var_77 = module_0.no_map_instance(var_76)
    var_78 = [var_3, var_4]
    var_79 = module_0.no_map_instance(var_78)
    var_80 = [var_77, var_79]
    var_81 = lambda x, y: x + y
    var_82 = module_0.map_structure_zip(var_81, var_80)



# Parsed testcases at query #30
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = [var_4, var_5, var_6]
    var_47 = module_0.no_map_instance(var_45)
    var_48 = lambda x, y: x + y
    var_49 = [var_45, var_46]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = (var_33, var_0)
    var_52 = (var_34, var_1)
    var_53 = [var_51, var_52]
    var_54 = (var_33, var_2)
    var_55 = (var_34, var_4)
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_44)
    var_59 = (var_33, var_0)
    var_60 = (var_34, var_1)
    var_61 = (var_33, var_2)
    var_62 = (var_34, var_4)
    var_63 = [var_59, var_60, var_61, var_62]
    var_64 = {var_0, var_1}
    var_65 = {var_2, var_4}
    var_66 = [var_64, var_65]
    var_67 = lambda x, y: x + y
    var_68 = module_0.map_structure_zip(var_67, var_66)



# Parsed testcases at query #31
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_21: var_2}
    var_26 = {var_20: var_25}
    var_27 = module_0.map_structure(var_24, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = [var_2, var_0, var_3]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = lambda x: x * var_0
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.map_structure(var_35, var_36)



# Parsed testcases at query #32
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = 'd'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = {var_19: var_2, var_20: var_26}
    var_28 = module_0.map_structure(var_23, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = lambda x: x * var_0
    var_33 = (var_3, var_8)
    var_34 = {var_19: var_0, var_20: var_33}
    var_35 = [var_2, var_34]
    var_36 = module_0.map_structure(var_32, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = type(var_38)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_38)
    var_45 = 'Point'
    var_46 = 'x'
    var_47 = 'y'
    var_48 = [var_46, var_47]
    var_49 = lambda x: x * var_0
    var_50 = lambda x: x * var_0
    var_51 = 6
    var_52 = lambda x: x * var_0
    var_53 = []
    var_54 = module_0.map_structure(var_52, var_53)
    var_55 = lambda x: x * var_0
    var_56 = ()
    var_57 = module_0.map_structure(var_55, var_56)
    var_58 = lambda x: x * var_0
    var_59 = {}
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x * var_0
    var_62 = set()
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = set()



# Parsed testcases at query #33
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
    var_11 = (var_0, var_1, var_2)
    var_12 = (var_4, var_5, var_6)
    var_13 = lambda x, y: x + y
    var_14 = [var_11, var_12]
    var_15 = module_0.map_structure_zip(var_13, var_14)
    var_16 = 'Point'
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = lambda x, y: x + y
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = {var_21: var_2, var_22: var_4}
    var_25 = lambda x, y: x + y
    var_26 = [var_23, var_24]
    var_27 = module_0.map_structure_zip(var_25, var_26)
    var_28 = [var_0, var_1]
    var_29 = (var_2, var_4)
    var_30 = {var_21: var_28, var_22: var_29}
    var_31 = [var_5, var_6]
    var_32 = 7
    var_33 = 8
    var_34 = (var_32, var_33)
    var_35 = {var_21: var_31, var_22: var_34}
    var_36 = lambda x, y: x + y
    var_37 = [var_30, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1, var_2]
    var_40 = [var_4, var_5, var_6]
    var_41 = lambda x, y: x + y
    var_42 = 'c'
    var_43 = [var_0, var_1]
    var_44 = (var_2, var_4)
    var_45 = {var_21: var_43, var_22: var_44, var_42: var_5}
    var_46 = [var_6, var_32]
    var_47 = 9
    var_48 = (var_33, var_47)
    var_49 = 10
    var_50 = {var_21: var_46, var_22: var_48, var_42: var_49}
    var_51 = lambda x, y: x + y
    var_52 = [var_45, var_50]
    var_53 = module_0.map_structure_zip(var_51, var_52)
    var_54 = {var_0, var_1, var_2}
    var_55 = {var_4, var_5, var_6}
    var_56 = lambda x, y: x + y
    var_57 = [var_54, var_55]
    var_58 = module_0.map_structure_zip(var_56, var_57)



# Parsed testcases at query #34
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #35
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = type(var_2)
    var_28 = module_0.register_no_map_class(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_2)
    assert var_30 == 2
    var_31 = [var_2, var_0, var_3]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = (var_0, var_3)
    var_36 = {var_20: var_8}
    var_37 = [var_2, var_35, var_36]
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)



# Parsed testcases at query #36
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)
    var_66 = [var_0]
    var_67 = [var_1]
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)
    var_71 = []
    var_72 = []
    var_73 = [var_71, var_72]
    var_74 = lambda x, y: x + y
    var_75 = module_0.map_structure_zip(var_74, var_73)
    var_76 = {var_0, var_1}
    var_77 = {var_2, var_4}
    var_78 = [var_76, var_77]
    var_79 = lambda x, y: x + y
    var_80 = module_0.map_structure_zip(var_79, var_78)



# Parsed testcases at query #37
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = lambda x, y: x.__name__ + y.__name__
    var_56 = module_0.map_structure_zip(var_55, var_52)
    assert var_56 == 'intstr'



# Parsed testcases at query #38
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = (var_3, var_8)
    var_33 = {var_19: var_0, var_20: var_32}
    var_34 = [var_2, var_33]
    var_35 = module_0.map_structure(var_31, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = [var_2, var_0, var_3]
    var_41 = type(var_40)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = [var_2, var_0, var_3]
    var_45 = module_0.map_structure(var_43, var_44)
    var_46 = 'Point'
    var_47 = 'x'
    var_48 = 'y'
    var_49 = [var_47, var_48]
    var_50 = lambda x: x * var_0
    var_51 = lambda x: x * var_0
    var_52 = 6
    var_53 = lambda x: x * var_0
    var_54 = []
    var_55 = module_0.map_structure(var_53, var_54)
    var_56 = lambda x: x * var_0
    var_57 = ()
    var_58 = module_0.map_structure(var_56, var_57)
    var_59 = lambda x: x * var_0
    var_60 = {}
    var_61 = module_0.map_structure(var_59, var_60)
    var_62 = lambda x: x * var_0
    var_63 = set()
    var_64 = module_0.map_structure(var_62, var_63)
    var_65 = set()



# Parsed testcases at query #39
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0, var_38: var_1}
    var_45 = 'c'
    var_46 = {var_45: var_2}
    var_47 = [var_44, var_46]
    var_48 = {var_37: var_4, var_38: var_5}
    var_49 = {var_45: var_6}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = [var_0, var_1, var_2]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = [var_4, var_5, var_6]
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = type(var_52)
    var_59 = module_0.register_no_map_class(var_58)
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_52, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)
    var_64 = {var_0, var_1}
    var_65 = {var_2, var_4}
    var_66 = [var_64, var_65]
    var_67 = lambda x, y: x + y
    var_68 = module_0.map_structure_zip(var_67, var_66)



# Parsed testcases at query #40
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_0, var_33: var_1}
    var_35 = {var_32: var_2, var_33: var_4}
    var_36 = [var_34, var_35]
    var_37 = lambda x, y: x + y
    var_38 = module_0.map_structure_zip(var_37, var_36)
    var_39 = {var_32: var_0, var_33: var_1}
    var_40 = 'c'
    var_41 = {var_40: var_2}
    var_42 = [var_39, var_41]
    var_43 = {var_32: var_4, var_33: var_5}
    var_44 = {var_40: var_6}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = type(var_55)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_0, var_1, var_2]
    var_59 = [var_4, var_5, var_6]
    var_60 = [var_58, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)
    var_63 = {var_0, var_1}
    var_64 = {var_2, var_4}
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)



# Parsed testcases at query #41
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda x, y, z: x + y + z
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_4]
    var_17 = [var_15, var_16]
    var_18 = [var_5, var_6]
    var_19 = [var_8, var_9]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = 0
    var_23 = lambda x, y: [x[var_22] + y[var_22], x[var_0] + y[var_0]]
    var_24 = module_0.map_structure_zip(var_23, var_21)
    var_25 = (var_0, var_1)
    var_26 = (var_2, var_4)
    var_27 = [var_25, var_26]
    var_28 = lambda x, y: x + y
    var_29 = module_0.map_structure_zip(var_28, var_27)
    var_30 = 'Point'
    var_31 = 'x'
    var_32 = 'y'
    var_33 = [var_31, var_32]
    var_34 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_35 = module_0.map_structure_zip(var_34, var_27)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = {var_36: var_0, var_37: var_1}
    var_39 = {var_36: var_2, var_37: var_4}
    var_40 = [var_38, var_39]
    var_41 = lambda x, y: x + y
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = [var_0, var_1]
    var_44 = (var_2, var_4)
    var_45 = [var_43, var_44]
    var_46 = lambda x, y: x + y
    var_47 = module_0.map_structure_zip(var_46, var_45)
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = [var_4, var_5, var_6]
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = type(var_48)
    var_55 = module_0.register_no_map_class(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_48, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = {var_0, var_1}
    var_61 = {var_2, var_4}
    var_62 = [var_60, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_35)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_35)



# Parsed testcases at query #2
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = lambda x: x * var_0
    var_28 = 5
    var_29 = module_0.map_structure(var_27, var_28)
    assert var_29 == 10
    var_30 = [var_2, var_0, var_3]
    var_31 = module_0.no_map_instance(var_30)
    var_32 = lambda x: x * var_0
    var_33 = module_0.map_structure(var_32, var_31)
    var_34 = (var_0, var_3)
    var_35 = {var_20: var_18}
    var_36 = [var_2, var_34, var_35]
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)



# Parsed testcases at query #3
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = type(var_49)
    var_56 = module_0.register_no_map_class(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_49, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = {var_0, var_1}
    var_62 = {var_2, var_4}
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)



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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0, var_39: var_1}
    var_46 = 'c'
    var_47 = {var_46: var_2}
    var_48 = [var_45, var_47]
    var_49 = {var_38: var_4, var_39: var_5}
    var_50 = {var_46: var_6}
    var_51 = [var_49, var_50]
    var_52 = [var_48, var_51]
    var_53 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = type(var_55)
    var_62 = module_0.register_no_map_class(var_61)
    var_63 = [var_4, var_5, var_6]
    var_64 = [var_55, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)
    var_67 = {var_0, var_1}
    var_68 = {var_2, var_4}
    var_69 = [var_67, var_68]
    var_70 = lambda x, y: x + y
    var_71 = module_0.map_structure_zip(var_70, var_69)



# Parsed testcases at query #5
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = {var_0, var_3}
    var_33 = {var_2, var_32}
    var_34 = module_0.map_structure(var_31, var_33)
    var_35 = lambda x: x * var_0
    var_36 = [var_3, var_8]
    var_37 = {var_19: var_0, var_20: var_36}
    var_38 = [var_2, var_37]
    var_39 = module_0.map_structure(var_35, var_38)
    var_40 = [var_2, var_0, var_3]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = [var_2, var_0, var_3]
    var_45 = type(var_44)
    var_46 = module_0.register_no_map_class(var_45)
    var_47 = lambda x: x * var_0
    var_48 = [var_2, var_0, var_3]
    var_49 = module_0.map_structure(var_47, var_48)
    var_50 = 'Point'
    var_51 = 'x'
    var_52 = 'y'
    var_53 = [var_51, var_52]
    var_54 = lambda x: x * var_0
    var_55 = lambda x: x * var_0
    var_56 = 6



# Parsed testcases at query #6
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = [var_4, var_5, var_6]
    var_51 = module_0.no_map_instance(var_49)
    var_52 = lambda x, y: x + y
    var_53 = [var_49, var_50]
    var_54 = module_0.map_structure_zip(var_52, var_53)
    var_55 = [var_0, var_1, var_2]
    var_56 = type(var_55)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = lambda x, y: x + y
    var_59 = [var_0, var_1, var_2]
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_59, var_60]
    var_62 = module_0.map_structure_zip(var_58, var_61)
    var_63 = {var_0, var_1}
    var_64 = {var_2, var_4}
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)



# Parsed testcases at query #7
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = lambda x: x * var_0
    var_28 = 5
    var_29 = module_0.map_structure(var_27, var_28)
    assert var_29 == 10
    var_30 = [var_2, var_0, var_3]
    var_31 = module_0.no_map_instance(var_30)
    var_32 = lambda x: x * var_0
    var_33 = module_0.map_structure(var_32, var_31)
    var_34 = (var_0, var_3)
    var_35 = {var_20: var_8}
    var_36 = [var_2, var_34, var_35]
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)



# Parsed testcases at query #8
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = [var_4, var_5, var_6]
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = {var_0, var_1}
    var_61 = {var_2, var_4}
    var_62 = [var_60, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_35)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_35)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = 4



# Parsed testcases at query #2
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_36)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_36)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = 4
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #3
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_48)
    var_55 = {var_0, var_1}
    var_56 = {var_2, var_4}
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)



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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #5
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = 'c'
    var_26 = {var_25: var_0}
    var_27 = {var_20: var_2, var_21: var_26}
    var_28 = module_0.map_structure(var_24, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = (var_20, var_2)
    var_33 = (var_21, var_0)
    var_34 = [var_32, var_33]
    var_35 = lambda x: x * var_0
    var_36 = (var_20, var_0)
    var_37 = (var_21, var_18)
    var_38 = [var_36, var_37]
    var_39 = lambda x: x * var_0
    var_40 = [var_2, var_0, var_3]
    var_41 = module_0.map_structure(var_39, var_40)



# Parsed testcases at query #6
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_36)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_36)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = 4
    var_49 = lambda x: x * var_0
    var_50 = []
    var_51 = module_0.map_structure(var_49, var_50)
    var_52 = lambda x: x * var_0
    var_53 = ()
    var_54 = module_0.map_structure(var_52, var_53)
    var_55 = lambda x: x * var_0
    var_56 = {}
    var_57 = module_0.map_structure(var_55, var_56)
    var_58 = lambda x: x * var_0
    var_59 = set()
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = set()



# Parsed testcases at query #7
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
    var_20 = 'Point'
    var_21 = 'x'
    var_22 = 'y'
    var_23 = [var_21, var_22]
    var_24 = lambda x: x + var_0
    var_25 = lambda x: x + var_0
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_0, var_27: var_2}
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = lambda x: x + var_0
    var_31 = 'd'
    var_32 = 'c'
    var_33 = {var_27: var_0, var_32: var_2}
    var_34 = {var_26: var_33, var_31: var_3}
    var_35 = module_0.map_structure(var_30, var_34)
    var_36 = lambda x: x + var_0
    var_37 = {var_0, var_2, var_3}
    var_38 = module_0.map_structure(var_36, var_37)
    var_39 = (var_2, var_3)
    var_40 = 5
    var_41 = 6
    var_42 = [var_40, var_41]
    var_43 = {var_26: var_8, var_27: var_42}
    var_44 = [var_0, var_39, var_43]
    var_45 = (var_3, var_8)
    var_46 = 7
    var_47 = [var_41, var_46]
    var_48 = {var_26: var_40, var_27: var_47}
    var_49 = [var_2, var_45, var_48]
    var_50 = lambda x: x + var_0
    var_51 = module_0.map_structure(var_50, var_44)
    var_52 = [var_0, var_2, var_3]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = lambda x: x + var_0
    var_55 = module_0.map_structure(var_54, var_53)
    var_56 = lambda x: x + var_0
    var_57 = [var_0, var_2, var_3]
    var_58 = module_0.map_structure(var_56, var_57)



# Parsed testcases at query #8
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
    var_10 = [var_1, var_2]
    var_11 = [var_4, var_5]
    var_12 = [var_10, var_11]
    var_13 = 5
    var_14 = 6
    var_15 = [var_13, var_14]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_12, var_19]
    var_21 = module_0.map_structure_zip(var_9, var_20)
    var_22 = lambda x, y: x + y
    var_23 = (var_1, var_2)
    var_24 = (var_4, var_5)
    var_25 = [var_23, var_24]
    var_26 = module_0.map_structure_zip(var_22, var_25)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = lambda x, y: x + y
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_1, var_34: var_2}
    var_36 = {var_33: var_4, var_34: var_5}
    var_37 = [var_35, var_36]
    var_38 = module_0.map_structure_zip(var_32, var_37)
    var_39 = lambda x, y: x + y
    var_40 = {var_33: var_2}
    var_41 = [var_1, var_40]
    var_42 = {var_33: var_5}
    var_43 = [var_4, var_42]
    var_44 = [var_41, var_43]
    var_45 = module_0.map_structure_zip(var_39, var_44)
    var_46 = [var_1, var_2]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_4, var_5]
    var_50 = [var_47, var_49]
    var_51 = module_0.map_structure_zip(var_48, var_50)
    var_52 = lambda x, y: x + y
    var_53 = [var_1, var_2]
    var_54 = [var_4, var_5]
    var_55 = [var_53, var_54]
    var_56 = module_0.map_structure_zip(var_52, var_55)
    var_57 = lambda x, y: x + y
    var_58 = 1
    var_59 = 2
    var_60 = {var_58, var_59}
    var_61 = 3
    var_62 = 4
    var_63 = {var_61, var_62}
    var_64 = [var_60, var_63]
    var_65 = module_0.map_structure_zip(var_57, var_64)



# Parsed testcases at query #9
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
    var_7 = (var_2, var_0, var_3)
    var_8 = module_0.map_structure(var_6, var_7)
    var_9 = lambda x: x * var_0
    var_10 = [var_0, var_3]
    var_11 = [var_2, var_10]
    var_12 = module_0.map_structure(var_9, var_11)
    var_13 = lambda x: x * var_0
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_2, var_15: var_0}
    var_17 = module_0.map_structure(var_13, var_16)
    var_18 = lambda x: x * var_0
    var_19 = {var_2, var_0, var_3}
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = lambda x: x * var_0
    var_26 = 4
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = lambda x: x * var_0
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.map_structure(var_31, var_32)



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
    var_8 = [var_3, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #11
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = (var_3, var_8)
    var_33 = {var_19: var_0, var_20: var_32}
    var_34 = [var_2, var_33]
    var_35 = module_0.map_structure(var_31, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = type(var_37)
    var_41 = module_0.register_no_map_class(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_37)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda x: x * var_0
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #12
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *args: sum(args)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_4]
    var_17 = [var_15, var_16]
    var_18 = [var_5, var_6]
    var_19 = [var_8, var_9]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = lambda *args: sum(args)
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = (var_0, var_1)
    var_25 = (var_2, var_4)
    var_26 = [var_24, var_25]
    var_27 = lambda *args: sum(args)
    var_28 = module_0.map_structure_zip(var_27, var_26)
    var_29 = 'Point'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = lambda *args: sum(args)
    var_34 = module_0.map_structure_zip(var_33, var_26)
    var_35 = 'a'
    var_36 = 'b'
    var_37 = {var_35: var_0, var_36: var_1}
    var_38 = {var_35: var_2, var_36: var_4}
    var_39 = [var_37, var_38]
    var_40 = lambda *args: sum(args)
    var_41 = module_0.map_structure_zip(var_40, var_39)
    var_42 = {var_35: var_0}
    var_43 = {var_36: var_1}
    var_44 = [var_42, var_43]
    var_45 = {var_35: var_2}
    var_46 = {var_36: var_4}
    var_47 = [var_45, var_46]
    var_48 = [var_44, var_47]
    var_49 = lambda *args: {k: sum(v) for (k, v) in zip(*args)}
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = module_0.no_map_instance(var_51)
    var_53 = [var_4, var_5, var_6]
    var_54 = [var_52, var_53]
    var_55 = lambda *args: sum(args)
    var_56 = module_0.map_structure_zip(var_55, var_54)
    var_57 = [var_0, var_1, var_2]
    var_58 = [var_4, var_5, var_6]
    var_59 = lambda *args: sum(args)
    var_60 = module_0.map_structure_zip(var_59, var_54)
    var_61 = [var_0, var_1, var_2]
    var_62 = [var_4, var_5, var_6]
    var_63 = {var_0, var_1}
    var_64 = {var_2, var_4}
    var_65 = [var_63, var_64]
    var_66 = lambda *args: sum(args)
    var_67 = module_0.map_structure_zip(var_66, var_65)



# Parsed testcases at query #13
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0}
    var_45 = {var_38: var_1}
    var_46 = [var_44, var_45]
    var_47 = {var_37: var_2}
    var_48 = {var_38: var_4}
    var_49 = [var_47, var_48]
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = [var_4, var_5, var_6]
    var_55 = module_0.no_map_instance(var_53)
    var_56 = module_0.no_map_instance(var_54)
    var_57 = lambda x, y: x + y
    var_58 = [var_55, var_56]
    var_59 = module_0.map_structure_zip(var_57, var_58)
    var_60 = [var_0, var_1, var_2]
    var_61 = [var_4, var_5, var_6]
    var_62 = lambda x, y: x + y
    var_63 = [var_53, var_54]
    var_64 = module_0.map_structure_zip(var_62, var_63)
    var_65 = {var_0, var_1}
    var_66 = {var_2, var_4}
    var_67 = [var_65, var_66]
    var_68 = lambda x, y: x + y
    var_69 = module_0.map_structure_zip(var_68, var_67)



# Parsed testcases at query #14
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0}
    var_45 = {var_38: var_1}
    var_46 = [var_44, var_45]
    var_47 = {var_37: var_2}
    var_48 = {var_38: var_4}
    var_49 = [var_47, var_48]
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = type(var_53)
    var_60 = module_0.register_no_map_class(var_59)
    var_61 = [var_4, var_5, var_6]
    var_62 = [var_53, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)
    var_65 = {var_0, var_1}
    var_66 = {var_2, var_4}
    var_67 = [var_65, var_66]
    var_68 = lambda x, y: x + y
    var_69 = module_0.map_structure_zip(var_68, var_67)



# Parsed testcases at query #15
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = (var_0, var_3)
    var_33 = {var_19: var_8}
    var_34 = [var_2, var_32, var_33]
    var_35 = module_0.map_structure(var_31, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = lambda x: x * var_0
    var_41 = [var_2, var_0, var_3]
    var_42 = module_0.map_structure(var_40, var_41)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0



# Parsed testcases at query #16
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *x: sum(x)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = (var_0, var_1)
    var_16 = (var_2, var_4)
    var_17 = (var_5, var_6)
    var_18 = [var_15, var_16, var_17]
    var_19 = lambda *x: sum(x)
    var_20 = module_0.map_structure_zip(var_19, var_18)
    var_21 = [var_0, var_1]
    var_22 = [var_2, var_4]
    var_23 = [var_5, var_6]
    var_24 = [var_21, var_22, var_23]
    var_25 = lambda *x: sum(x)
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = {var_27: var_5, var_28: var_6}
    var_32 = [var_29, var_30, var_31]
    var_33 = lambda *x: sum(x)
    var_34 = module_0.map_structure_zip(var_33, var_32)
    var_35 = [var_0, var_1]
    var_36 = (var_2, var_4)
    var_37 = {var_27: var_35, var_28: var_36}
    var_38 = [var_5, var_6]
    var_39 = (var_8, var_9)
    var_40 = {var_27: var_38, var_28: var_39}
    var_41 = [var_37, var_40]
    var_42 = lambda *x: sum(x)
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = [var_0, var_1, var_2]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = [var_4, var_5, var_6]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda *x: sum(x)
    var_49 = [var_45, var_47]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    assert var_50 == 21
    var_51 = [var_0, var_1, var_2]
    var_52 = type(var_51)
    var_53 = module_0.register_no_map_class(var_52)
    var_54 = lambda *x: sum(x)
    var_55 = [var_0, var_1, var_2]
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = module_0.map_structure_zip(var_54, var_57)
    assert var_58 == 21



# Parsed testcases at query #17
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = [var_4, var_5, var_6]
    var_56 = module_0.no_map_instance(var_54)
    var_57 = module_0.no_map_instance(var_55)
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: sum(x) + sum(y)
    var_60 = module_0.map_structure_zip(var_59, var_58)
    assert var_60 == 21
    var_61 = [var_0, var_1, var_2]
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: sum(x) + sum(y)
    var_65 = module_0.map_structure_zip(var_64, var_63)
    assert var_65 == 21
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #18
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)



# Parsed testcases at query #19
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
    var_30 = {var_25: var_1}
    var_31 = [var_0, var_30]
    var_32 = {var_25: var_4}
    var_33 = [var_2, var_32]
    var_34 = [var_31, var_33]
    var_35 = [var_0, var_1, var_2]
    var_36 = [var_4, var_5, var_6]
    var_37 = module_0.no_map_instance(var_35)
    var_38 = [var_35, var_36]
    var_39 = type(var_35)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = [var_35, var_36]
    var_42 = {var_0, var_1}
    var_43 = {var_2, var_4}
    var_44 = [var_42, var_43]



# Parsed testcases at query #20
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = type(var_45)
    var_52 = module_0.register_no_map_class(var_51)
    var_53 = [var_4, var_5, var_6]
    var_54 = [var_45, var_53]
    var_55 = lambda x, y: x + y
    var_56 = module_0.map_structure_zip(var_55, var_54)
    var_57 = {var_0, var_1}
    var_58 = {var_2, var_4}
    var_59 = [var_57, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)



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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = lambda x, y: x + y
    var_54 = module_0.map_structure_zip(var_53, var_48)
    var_55 = {var_0, var_1}
    var_56 = {var_2, var_4}
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)



# Parsed testcases at query #22
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0, var_38: var_1}
    var_45 = 'c'
    var_46 = {var_45: var_2}
    var_47 = [var_44, var_46]
    var_48 = {var_37: var_4, var_38: var_5}
    var_49 = {var_45: var_6}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = [var_0, var_1, var_2]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = [var_4, var_5, var_6]
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = type(var_52)
    var_59 = module_0.register_no_map_class(var_58)
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_52, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)
    var_64 = {var_0, var_1}
    var_65 = {var_2, var_4}
    var_66 = [var_64, var_65]
    var_67 = lambda x, y: x + y
    var_68 = module_0.map_structure_zip(var_67, var_66)



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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *x: sum(x)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_4]
    var_17 = [var_15, var_16]
    var_18 = [var_5, var_6]
    var_19 = [var_8, var_9]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = lambda *x: sum(x)
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = (var_0, var_1)
    var_25 = (var_2, var_4)
    var_26 = (var_24, var_25)
    var_27 = (var_5, var_6)
    var_28 = (var_8, var_9)
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = lambda *x: sum(x)
    var_32 = module_0.map_structure_zip(var_31, var_30)
    var_33 = 'Point'
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = lambda *x: sum(x)
    var_38 = module_0.map_structure_zip(var_37, var_30)
    var_39 = 'a'
    var_40 = 'b'
    var_41 = {var_39: var_0, var_40: var_1}
    var_42 = {var_39: var_2, var_40: var_4}
    var_43 = [var_41, var_42]
    var_44 = lambda *x: sum(x)
    var_45 = module_0.map_structure_zip(var_44, var_43)
    var_46 = {var_39: var_0, var_40: var_1}
    var_47 = 'c'
    var_48 = {var_47: var_2}
    var_49 = [var_46, var_48]
    var_50 = {var_39: var_4, var_40: var_5}
    var_51 = {var_47: var_6}
    var_52 = [var_50, var_51]
    var_53 = [var_49, var_52]
    var_54 = lambda *x: sum(x)
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = None
    var_57 = type(var_56)
    var_58 = module_0.register_no_map_class(var_57)
    var_59 = [var_56, var_56]
    var_60 = lambda *x: var_56
    var_61 = module_0.map_structure_zip(var_60, var_59)
    assert var_61 is None
    var_62 = [var_0, var_1, var_2]
    var_63 = module_0.no_map_instance(var_62)
    var_64 = [var_4, var_5, var_6]
    var_65 = [var_63, var_64]
    var_66 = lambda *x: sum(x)
    var_67 = module_0.map_structure_zip(var_66, var_65)
    var_68 = {var_0, var_1}
    var_69 = {var_2, var_4}
    var_70 = [var_68, var_69]
    var_71 = lambda *x: sum(x)
    var_72 = module_0.map_structure_zip(var_71, var_70)



# Parsed testcases at query #24
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = 'c'
    var_26 = {var_25: var_0}
    var_27 = {var_20: var_2, var_21: var_26}
    var_28 = module_0.map_structure(var_24, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.no_map_instance(var_32)
    var_34 = lambda x: x * var_0
    var_35 = module_0.map_structure(var_34, var_33)
    var_36 = type(var_32)
    var_37 = module_0.register_no_map_class(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_32)
    var_40 = (var_0, var_3)
    var_41 = {var_20: var_18}
    var_42 = [var_2, var_40, var_41]
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_42)



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
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = lambda x: x + var_0
    var_20 = lambda x: x + var_0
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_2}
    var_24 = module_0.map_structure(var_20, var_23)
    var_25 = lambda x: x + var_0
    var_26 = {var_0, var_2, var_3}
    var_27 = module_0.map_structure(var_25, var_26)
    var_28 = lambda x: x + var_0
    var_29 = 5
    var_30 = module_0.map_structure(var_28, var_29)
    assert var_30 == 6
    var_31 = [var_0, var_2, var_3]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: x + var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = (var_2, var_3)
    var_36 = {var_21: var_8}
    var_37 = [var_0, var_35, var_36]
    var_38 = lambda x: x + var_0
    var_39 = module_0.map_structure(var_38, var_37)



# Parsed testcases at query #26
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = [var_4, var_5, var_6]
    var_47 = module_0.no_map_instance(var_45)
    var_48 = lambda x, y: x + y
    var_49 = [var_45, var_46]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)



# Parsed testcases at query #27
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *x: sum(x)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_4]
    var_17 = [var_15, var_16]
    var_18 = [var_5, var_6]
    var_19 = [var_8, var_9]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = lambda *x: sum(x)
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = (var_0, var_1)
    var_25 = (var_2, var_4)
    var_26 = (var_24, var_25)
    var_27 = (var_5, var_6)
    var_28 = (var_8, var_9)
    var_29 = (var_27, var_28)
    var_30 = [var_26, var_29]
    var_31 = lambda *x: sum(x)
    var_32 = module_0.map_structure_zip(var_31, var_30)
    var_33 = 'Point'
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = lambda *x: sum(x)
    var_38 = module_0.map_structure_zip(var_37, var_30)
    var_39 = 'a'
    var_40 = 'b'
    var_41 = {var_39: var_0, var_40: var_1}
    var_42 = {var_39: var_2, var_40: var_4}
    var_43 = [var_41, var_42]
    var_44 = lambda *x: sum(x)
    var_45 = module_0.map_structure_zip(var_44, var_43)
    var_46 = [var_0, var_1, var_2]
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda *x: sum(x)
    var_50 = module_0.map_structure_zip(var_49, var_48)
    assert var_50 == 21
    var_51 = [var_0, var_1, var_2]
    var_52 = module_0.no_map_instance(var_51)
    var_53 = [var_4, var_5, var_6]
    var_54 = [var_52, var_53]
    var_55 = lambda *x: sum(x)
    var_56 = module_0.map_structure_zip(var_55, var_54)
    assert var_56 == 21
    var_57 = {var_0, var_1}
    var_58 = {var_2, var_4}
    var_59 = [var_57, var_58]
    var_60 = lambda *x: sum(x)
    var_61 = module_0.map_structure_zip(var_60, var_59)



# Parsed testcases at query #28
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = [var_4, var_5, var_6]
    var_47 = module_0.no_map_instance(var_45)
    var_48 = lambda x, y: x + y
    var_49 = [var_45, var_46]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = type(var_45)
    var_52 = module_0.register_no_map_class(var_51)
    var_53 = lambda x, y: x + y
    var_54 = [var_45, var_46]
    var_55 = module_0.map_structure_zip(var_53, var_54)
    var_56 = {var_0, var_1}
    var_57 = {var_2, var_4}
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)



# Parsed testcases at query #29
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
    var_7 = [var_2, var_3]
    var_8 = 4
    var_9 = [var_0, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x + var_0
    var_12 = (var_0, var_2, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x + var_0
    var_15 = (var_2, var_3)
    var_16 = (var_0, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x + var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_0, var_20: var_2}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x + var_0
    var_24 = 'c'
    var_25 = {var_24: var_2}
    var_26 = {var_19: var_0, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x + var_0
    var_29 = {var_0, var_2, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = [var_0, var_2, var_3]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: x + var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = lambda x: x + var_0
    var_36 = [var_0, var_2, var_3]
    var_37 = module_0.map_structure(var_35, var_36)
    var_38 = 'Point'
    var_39 = 'x'
    var_40 = 'y'
    var_41 = [var_39, var_40]
    var_42 = lambda x: x + var_0
    var_43 = lambda x: x + var_0



# Parsed testcases at query #30
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = 'd'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = {var_19: var_2, var_20: var_26}
    var_28 = module_0.map_structure(var_23, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = lambda x: x * var_0
    var_33 = (var_0, var_3)
    var_34 = {var_19: var_8}
    var_35 = [var_2, var_33, var_34]
    var_36 = module_0.map_structure(var_32, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = type(var_38)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_38)
    var_45 = 'Point'
    var_46 = 'x'
    var_47 = 'y'
    var_48 = [var_46, var_47]
    var_49 = lambda x: x * var_0
    var_50 = lambda x: x * var_0
    var_51 = 6



# Parsed testcases at query #31
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = 'Point'
    var_19 = 'x'
    var_20 = 'y'
    var_21 = [var_19, var_20]
    var_22 = lambda x: x * var_0
    var_23 = lambda x: x * var_0
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_2, var_25: var_0}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = 'c'
    var_30 = {var_29: var_0}
    var_31 = {var_24: var_2, var_25: var_30}
    var_32 = module_0.map_structure(var_28, var_31)
    var_33 = lambda x: x * var_0
    var_34 = {var_2, var_0, var_3}
    var_35 = module_0.map_structure(var_33, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = type(var_36)
    var_41 = module_0.register_no_map_class(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_36)
    var_44 = (var_0, var_3)
    var_45 = {var_24: var_8}
    var_46 = [var_2, var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = module_0.map_structure(var_47, var_46)



# Parsed testcases at query #32
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
    var_11 = (var_0, var_1, var_2)
    var_12 = (var_4, var_5, var_6)
    var_13 = lambda x, y: x + y
    var_14 = [var_11, var_12]
    var_15 = module_0.map_structure_zip(var_13, var_14)
    var_16 = 'Point'
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = lambda x, y: x + y
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = {var_21: var_2, var_22: var_4}
    var_25 = lambda x, y: x + y
    var_26 = [var_23, var_24]
    var_27 = module_0.map_structure_zip(var_25, var_26)
    var_28 = [var_0, var_1]
    var_29 = (var_2, var_4)
    var_30 = {var_21: var_28, var_22: var_29}
    var_31 = [var_5, var_6]
    var_32 = 7
    var_33 = 8
    var_34 = (var_32, var_33)
    var_35 = {var_21: var_31, var_22: var_34}
    var_36 = [var_6, var_33]
    var_37 = 10
    var_38 = 12
    var_39 = (var_37, var_38)
    var_40 = {var_21: var_36, var_22: var_39}
    var_41 = lambda x, y: x + y
    var_42 = [var_30, var_35]
    var_43 = module_0.map_structure_zip(var_41, var_42)
    var_44 = lambda x, y: x + y
    var_45 = [var_0]
    var_46 = [var_1]
    var_47 = [var_45, var_46]
    var_48 = module_0.map_structure_zip(var_44, var_47)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = lambda x, y: x + y
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_50, var_52]
    var_54 = module_0.map_structure_zip(var_51, var_53)
    var_55 = [var_0, var_1, var_2]
    var_56 = type(var_55)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = lambda x, y: x + y
    var_59 = [var_0, var_1, var_2]
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_59, var_60]
    var_62 = module_0.map_structure_zip(var_58, var_61)
    var_63 = lambda x, y: x + y
    var_64 = 1
    var_65 = 2
    var_66 = {var_64, var_65}
    var_67 = 3
    var_68 = 4
    var_69 = {var_67, var_68}
    var_70 = [var_66, var_69]
    var_71 = module_0.map_structure_zip(var_63, var_70)



# Parsed testcases at query #33
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #34
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = (var_0, var_1)
    var_28 = (var_2, var_4)
    var_29 = (var_27, var_28)
    var_30 = (var_5, var_6)
    var_31 = (var_15, var_16)
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = lambda x, y: x + y
    var_35 = module_0.map_structure_zip(var_34, var_33)
    var_36 = 'a'
    var_37 = 'b'
    var_38 = {var_36: var_0, var_37: var_1}
    var_39 = {var_36: var_2, var_37: var_4}
    var_40 = [var_38, var_39]
    var_41 = lambda x, y: x + y
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = 'x'
    var_44 = 'y'
    var_45 = {var_43: var_0, var_44: var_1}
    var_46 = {var_36: var_45, var_37: var_2}
    var_47 = {var_43: var_2, var_44: var_4}
    var_48 = {var_36: var_47, var_37: var_5}
    var_49 = [var_46, var_48]
    var_50 = lambda x, y: x + y
    var_51 = module_0.map_structure_zip(var_50, var_49)
    var_52 = (var_1, var_2)
    var_53 = [var_0, var_52]
    var_54 = (var_5, var_6)
    var_55 = [var_4, var_54]
    var_56 = [var_53, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = [var_0, var_1, var_2]
    var_60 = [var_4, var_5, var_6]
    var_61 = module_0.no_map_instance(var_59)
    var_62 = module_0.no_map_instance(var_60)
    var_63 = lambda x, y: x + y
    var_64 = [var_59, var_60]
    var_65 = module_0.map_structure_zip(var_63, var_64)
    var_66 = lambda x, y: x + y
    var_67 = [var_0, var_1, var_2]
    var_68 = [var_4, var_5, var_6]
    var_69 = [var_67, var_68]
    var_70 = module_0.map_structure_zip(var_66, var_69)
    var_71 = {var_0, var_1}
    var_72 = {var_2, var_4}
    var_73 = [var_71, var_72]
    var_74 = lambda x, y: x + y
    var_75 = module_0.map_structure_zip(var_74, var_73)



# Parsed testcases at query #35
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
    var_10 = [var_1, var_2]
    var_11 = [var_4, var_5]
    var_12 = [var_10, var_11]
    var_13 = 5
    var_14 = 6
    var_15 = [var_13, var_14]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_12, var_19]
    var_21 = module_0.map_structure_zip(var_9, var_20)
    var_22 = lambda x, y: x + y
    var_23 = (var_1, var_2)
    var_24 = (var_4, var_5)
    var_25 = [var_23, var_24]
    var_26 = module_0.map_structure_zip(var_22, var_25)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = lambda x, y: x + y
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_1, var_34: var_2}
    var_36 = {var_33: var_4, var_34: var_5}
    var_37 = [var_35, var_36]
    var_38 = module_0.map_structure_zip(var_32, var_37)
    var_39 = lambda x, y: x + y
    var_40 = {var_33: var_2}
    var_41 = [var_1, var_40]
    var_42 = {var_33: var_5}
    var_43 = [var_4, var_42]
    var_44 = [var_41, var_43]
    var_45 = module_0.map_structure_zip(var_39, var_44)
    var_46 = [var_1, var_2, var_4]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_1, var_1, var_1]
    var_50 = [var_47, var_49]
    var_51 = module_0.map_structure_zip(var_48, var_50)
    var_52 = type(var_46)
    var_53 = module_0.register_no_map_class(var_52)
    var_54 = lambda x, y: x + y
    var_55 = [var_1, var_1, var_1]
    var_56 = [var_46, var_55]
    var_57 = module_0.map_structure_zip(var_54, var_56)
    var_58 = lambda x: x * var_2
    var_59 = [var_13]
    var_60 = [var_59]
    var_61 = module_0.map_structure_zip(var_58, var_60)
    var_62 = lambda x, y: x + y
    var_63 = []
    var_64 = []
    var_65 = [var_63, var_64]
    var_66 = module_0.map_structure_zip(var_62, var_65)
    var_67 = lambda x, y: x + y
    var_68 = 1
    var_69 = 2
    var_70 = {var_68, var_69}
    var_71 = 3
    var_72 = 4
    var_73 = {var_71, var_72}
    var_74 = [var_70, var_73]
    var_75 = module_0.map_structure_zip(var_67, var_74)



# Parsed testcases at query #36
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_36)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_36)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = 4
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #37
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #38
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = [var_0, var_1]
    var_46 = (var_2, var_4)
    var_47 = {var_38: var_45, var_39: var_46}
    var_48 = [var_5, var_6]
    var_49 = (var_15, var_16)
    var_50 = {var_38: var_48, var_39: var_49}
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #39
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #40
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = [var_0, var_1]
    var_45 = (var_2, var_4)
    var_46 = {var_37: var_44, var_38: var_45}
    var_47 = [var_5, var_6]
    var_48 = (var_15, var_16)
    var_49 = {var_37: var_47, var_38: var_48}
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = type(var_53)
    var_60 = module_0.register_no_map_class(var_59)
    var_61 = [var_4, var_5, var_6]
    var_62 = [var_53, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)
    var_65 = {var_0, var_1}
    var_66 = {var_2, var_4}
    var_67 = [var_65, var_66]
    var_68 = lambda x, y: x + y
    var_69 = module_0.map_structure_zip(var_68, var_67)



# Parsed testcases at query #41
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = lambda x: x * var_0
    var_32 = (var_0, var_3)
    var_33 = {var_19: var_8}
    var_34 = [var_2, var_32, var_33]
    var_35 = module_0.map_structure(var_31, var_34)
    var_36 = [var_2, var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = type(var_37)
    var_41 = module_0.register_no_map_class(var_40)
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_37)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda x: x * var_0
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #42
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = lambda x, y: [x[var_29] + y[var_29], {k: x[var_0][k] + y[var_0][k] for k in x[var_0]}]
    var_51 = module_0.map_structure_zip(var_50, var_49)
    var_52 = [var_0, var_1, var_2]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = [var_4, var_5, var_6]
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = type(var_52)
    var_59 = module_0.register_no_map_class(var_58)
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_52, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)
    var_64 = {var_0, var_1}
    var_65 = {var_2, var_4}
    var_66 = [var_64, var_65]
    var_67 = lambda x, y: x + y
    var_68 = module_0.map_structure_zip(var_67, var_66)



# Parsed testcases at query #43
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0, var_39: var_1}
    var_46 = 'c'
    var_47 = {var_46: var_2}
    var_48 = [var_45, var_47]
    var_49 = {var_38: var_4, var_39: var_5}
    var_50 = {var_46: var_6}
    var_51 = [var_49, var_50]
    var_52 = [var_48, var_51]
    var_53 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = type(var_55)
    var_62 = module_0.register_no_map_class(var_61)
    var_63 = [var_4, var_5, var_6]
    var_64 = [var_55, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)
    var_67 = {var_0, var_1}
    var_68 = {var_2, var_4}
    var_69 = [var_67, var_68]
    var_70 = lambda x, y: x + y
    var_71 = module_0.map_structure_zip(var_70, var_69)



# Parsed testcases at query #44
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = lambda x: x * var_0
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.map_structure(var_31, var_32)
    var_34 = {var_20: var_3}
    var_35 = (var_0, var_34)
    var_36 = [var_2, var_35]
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)



# Parsed testcases at query #45
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #46
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = {var_0, var_1}
    var_57 = {var_2, var_4}
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)



# Parsed testcases at query #47
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = (var_38, var_0)
    var_55 = (var_39, var_1)
    var_56 = [var_54, var_55]
    var_57 = (var_38, var_2)
    var_58 = (var_39, var_4)
    var_59 = [var_57, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_51)
    assert var_61 == 4
    var_62 = [var_0, var_1, var_2]
    var_63 = module_0.no_map_instance(var_62)
    var_64 = [var_4, var_5, var_6]
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)
    assert var_67 == 10
    var_68 = {var_0, var_1}
    var_69 = {var_2, var_4}
    var_70 = [var_68, var_69]
    var_71 = lambda x, y: x + y
    var_72 = module_0.map_structure_zip(var_71, var_70)



# Parsed testcases at query #48
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = [var_4, var_5, var_6]
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = {var_0, var_1}
    var_59 = {var_2, var_4}
    var_60 = [var_58, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)



# Parsed testcases at query #49
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = type(var_27)
    var_32 = module_0.register_no_map_class(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_27)
    var_35 = (var_0, var_3)
    var_36 = {var_20: var_18}
    var_37 = [var_2, var_35, var_36]
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)



# Parsed testcases at query #50
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
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = lambda x, y: x + y
    var_25 = [var_22, var_23]
    var_26 = module_0.map_structure_zip(var_24, var_25)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_0, var_33: var_1}
    var_35 = {var_32: var_2, var_33: var_4}
    var_36 = lambda x, y: x + y
    var_37 = [var_34, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1]
    var_40 = (var_2, var_4)
    var_41 = {var_32: var_39, var_33: var_40}
    var_42 = [var_5, var_6]
    var_43 = (var_15, var_16)
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = lambda x, y: x + y
    var_46 = [var_41, var_44]
    var_47 = module_0.map_structure_zip(var_45, var_46)
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x, y: x + y
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_49, var_51]
    var_53 = module_0.map_structure_zip(var_50, var_52)
    var_54 = [var_0, var_1, var_2]
    var_55 = type(var_54)
    var_56 = module_0.register_no_map_class(var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = lambda x, y: x + y
    var_59 = [var_4, var_5, var_6]
    var_60 = [var_57, var_59]
    var_61 = module_0.map_structure_zip(var_58, var_60)
    var_62 = {var_0, var_1, var_2}
    var_63 = {var_4, var_5, var_6}
    var_64 = lambda x, y: x + y
    var_65 = [var_62, var_63]
    var_66 = module_0.map_structure_zip(var_64, var_65)



# Parsed testcases at query #51
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = lambda x: x * var_0
    var_14 = (var_0, var_3)
    var_15 = (var_2, var_14)
    var_16 = module_0.map_structure(var_13, var_15)
    var_17 = lambda x: x * var_0
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_2, var_19: var_0}
    var_21 = module_0.map_structure(var_17, var_20)
    var_22 = lambda x: x * var_0
    var_23 = 'c'
    var_24 = {var_23: var_0}
    var_25 = {var_18: var_2, var_19: var_24}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = lambda x: x * var_0
    var_28 = {var_2, var_0, var_3}
    var_29 = module_0.map_structure(var_27, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_18: var_3}
    var_32 = (var_0, var_31)
    var_33 = [var_2, var_32]
    var_34 = module_0.map_structure(var_30, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = type(var_36)
    var_40 = module_0.register_no_map_class(var_39)
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_36)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda x: x * var_0
    var_48 = 4
    var_49 = lambda x: x * var_0
    var_50 = 6



# Parsed testcases at query #52
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = lambda x: x * var_0
    var_32 = [var_2, var_0, var_3]
    var_33 = module_0.map_structure(var_31, var_32)
    var_34 = (var_0, var_3)
    var_35 = {var_20: var_8}
    var_36 = [var_2, var_34, var_35]
    var_37 = lambda x: x * var_0
    var_38 = module_0.map_structure(var_37, var_36)



# Parsed testcases at query #53
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = {var_24: var_0}
    var_26 = {var_19: var_2, var_20: var_25}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = lambda x: x * var_0
    var_29 = {var_2, var_0, var_3}
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = (var_0, var_3)
    var_32 = {var_19: var_31}
    var_33 = 5
    var_34 = {var_8, var_33}
    var_35 = [var_2, var_32, var_34]
    var_36 = 6
    var_37 = (var_8, var_36)
    var_38 = {var_19: var_37}
    var_39 = 8
    var_40 = 10
    var_41 = {var_39, var_40}
    var_42 = [var_0, var_38, var_41]
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_35)
    var_45 = [var_2, var_0, var_3]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = lambda x: x * var_0
    var_48 = module_0.map_structure(var_47, var_46)
    var_49 = [var_2, var_0, var_3]
    var_50 = type(var_49)
    var_51 = module_0.register_no_map_class(var_50)
    var_52 = lambda x: x * var_0
    var_53 = [var_2, var_0, var_3]
    var_54 = module_0.map_structure(var_52, var_53)
    var_55 = 'Point'
    var_56 = 'x'
    var_57 = 'y'
    var_58 = [var_56, var_57]
    var_59 = lambda x: x * var_0
    var_60 = lambda x: x * var_0
    var_61 = []
    var_62 = module_0.map_structure(var_60, var_61)
    var_63 = lambda x: x * var_0
    var_64 = ()
    var_65 = module_0.map_structure(var_63, var_64)
    var_66 = lambda x: x * var_0
    var_67 = {}
    var_68 = module_0.map_structure(var_66, var_67)
    var_69 = lambda x: x * var_0
    var_70 = set()
    var_71 = module_0.map_structure(var_69, var_70)
    var_72 = set()



# Parsed testcases at query #54
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'Point'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_15, var_16]
    var_18 = lambda x: x * var_0
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = type(var_27)
    var_32 = module_0.register_no_map_class(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_27)
    var_35 = (var_0, var_3)
    var_36 = {var_20: var_8}
    var_37 = [var_2, var_35, var_36]
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)



# Parsed testcases at query #55
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0}
    var_46 = {var_39: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_38: var_2}
    var_49 = {var_39: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_4, var_5, var_6]
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = type(var_54)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_4, var_5, var_6]
    var_63 = [var_54, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #56
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
    var_7 = (var_2, var_0, var_3)
    var_8 = module_0.map_structure(var_6, var_7)
    var_9 = lambda x: x * var_0
    var_10 = [var_0, var_3]
    var_11 = [var_2, var_10]
    var_12 = module_0.map_structure(var_9, var_11)
    var_13 = lambda x: x * var_0
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_2, var_15: var_0}
    var_17 = module_0.map_structure(var_13, var_16)
    var_18 = lambda x: x * var_0
    var_19 = {var_2, var_0, var_3}
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = [var_2, var_0, var_3]
    var_22 = module_0.no_map_instance(var_21)
    var_23 = lambda x: x * var_0
    var_24 = module_0.map_structure(var_23, var_22)
    var_25 = lambda x: x * var_0
    var_26 = [var_2, var_0, var_3]
    var_27 = module_0.map_structure(var_25, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda x: x * var_0
    var_33 = 4
    var_34 = lambda x: x * var_0
    var_35 = 5
    var_36 = module_0.map_structure(var_34, var_35)
    assert var_36 == 10



# Parsed testcases at query #57
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_0, var_33: var_1}
    var_35 = {var_32: var_2, var_33: var_4}
    var_36 = [var_34, var_35]
    var_37 = lambda x, y: x + y
    var_38 = module_0.map_structure_zip(var_37, var_36)
    var_39 = {var_32: var_0, var_33: var_1}
    var_40 = 'c'
    var_41 = {var_40: var_2}
    var_42 = [var_39, var_41]
    var_43 = {var_32: var_4, var_33: var_5}
    var_44 = {var_40: var_6}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = [var_4, var_5, var_6]
    var_51 = module_0.no_map_instance(var_49)
    var_52 = module_0.no_map_instance(var_50)
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = [var_0, var_1, var_2]
    var_57 = [var_4, var_5, var_6]
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = module_0.map_structure_zip(var_59, var_58)
    var_61 = {var_0, var_1}
    var_62 = {var_2, var_4}
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)



# Parsed testcases at query #58
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_1}
    var_45 = [var_0, var_44]
    var_46 = {var_37: var_4}
    var_47 = [var_2, var_46]
    var_48 = [var_45, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = module_0.no_map_instance(var_51)
    var_54 = lambda x, y: x + y
    var_55 = [var_51, var_52]
    var_56 = module_0.map_structure_zip(var_54, var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_57, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #59
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_0}
    var_45 = {var_38: var_1}
    var_46 = [var_44, var_45]
    var_47 = {var_37: var_2}
    var_48 = {var_38: var_4}
    var_49 = [var_47, var_48]
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_0, var_1, var_2]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = [var_0, var_1, var_2]
    var_60 = type(var_59)
    var_61 = module_0.register_no_map_class(var_60)
    var_62 = [var_0, var_1, var_2]
    var_63 = [var_4, var_5, var_6]
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #60
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = type(var_47)
    var_54 = module_0.register_no_map_class(var_53)
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_47, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #61
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
    var_10 = [var_1, var_2]
    var_11 = [var_4]
    var_12 = [var_10, var_11]
    var_13 = 5
    var_14 = [var_5, var_13]
    var_15 = 6
    var_16 = [var_15]
    var_17 = [var_14, var_16]
    var_18 = [var_12, var_17]
    var_19 = module_0.map_structure_zip(var_9, var_18)
    var_20 = lambda x, y: x + y
    var_21 = (var_1, var_2)
    var_22 = (var_4, var_5)
    var_23 = [var_21, var_22]
    var_24 = module_0.map_structure_zip(var_20, var_23)
    var_25 = 'Point'
    var_26 = 'x'
    var_27 = 'y'
    var_28 = [var_26, var_27]
    var_29 = lambda x, y: x + y
    var_30 = lambda x, y: x + y
    var_31 = 'a'
    var_32 = 'b'
    var_33 = {var_31: var_1, var_32: var_2}
    var_34 = {var_31: var_4, var_32: var_5}
    var_35 = [var_33, var_34]
    var_36 = module_0.map_structure_zip(var_30, var_35)
    var_37 = lambda x, y: x + y
    var_38 = {var_31: var_2}
    var_39 = [var_1, var_38]
    var_40 = {var_31: var_5}
    var_41 = [var_4, var_40]
    var_42 = [var_39, var_41]
    var_43 = module_0.map_structure_zip(var_37, var_42)
    var_44 = [var_1, var_2, var_4]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x, y: x + y
    var_47 = [var_4, var_5, var_13]
    var_48 = [var_45, var_47]
    var_49 = module_0.map_structure_zip(var_46, var_48)
    var_50 = type(var_44)
    var_51 = module_0.register_no_map_class(var_50)
    var_52 = lambda x, y: x + y
    var_53 = [var_4, var_5, var_13]
    var_54 = [var_44, var_53]
    var_55 = module_0.map_structure_zip(var_52, var_54)
    var_56 = lambda x, y: x + y
    var_57 = 1
    var_58 = 2
    var_59 = {var_57, var_58}
    var_60 = 3
    var_61 = 4
    var_62 = {var_60, var_61}
    var_63 = [var_59, var_62]
    var_64 = module_0.map_structure_zip(var_56, var_63)



# Parsed testcases at query #62
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = [var_0, var_1]
    var_7 = 4
    var_8 = [var_2, var_7]
    var_9 = [var_6, var_8]
    var_10 = lambda x: x * var_1
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = lambda x: x * var_1
    var_20 = module_0.map_structure(var_19, var_12)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = lambda x: x * var_1
    var_25 = module_0.map_structure(var_24, var_23)
    var_26 = {var_0, var_1, var_2}
    var_27 = lambda x: x * var_1
    var_28 = module_0.map_structure(var_27, var_26)
    var_29 = [var_0, var_1, var_2]
    var_30 = module_0.no_map_instance(var_29)
    var_31 = lambda x: x * var_1
    var_32 = module_0.map_structure(var_31, var_30)
    var_33 = [var_0, var_1, var_2]
    var_34 = lambda x: x * var_1
    var_35 = module_0.map_structure(var_34, var_33)



# Parsed testcases at query #63
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
    var_7 = [var_0, var_3]
    var_8 = 4
    var_9 = [var_2, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_0
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = lambda x: x * var_0
    var_15 = (var_0, var_3)
    var_16 = (var_2, var_15, var_8)
    var_17 = module_0.map_structure(var_14, var_16)
    var_18 = lambda x: x * var_0
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_2, var_20: var_0}
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = 'c'
    var_25 = 'd'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = {var_19: var_2, var_20: var_26}
    var_28 = module_0.map_structure(var_23, var_27)
    var_29 = lambda x: x * var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = lambda x: x * var_0
    var_33 = (var_0, var_3)
    var_34 = {var_19: var_8}
    var_35 = [var_2, var_33, var_34]
    var_36 = module_0.map_structure(var_32, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_0
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = type(var_37)
    var_42 = module_0.register_no_map_class(var_41)
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_37)
    var_45 = 'Point'
    var_46 = 'x'
    var_47 = 'y'
    var_48 = [var_46, var_47]
    var_49 = lambda x: x * var_0



# Parsed testcases at query #64
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = module_0.map_structure_zip(var_35, var_28)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = {var_37: var_0, var_38: var_1}
    var_40 = {var_37: var_2, var_38: var_4}
    var_41 = [var_39, var_40]
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = {var_37: var_1}
    var_45 = [var_0, var_44]
    var_46 = {var_37: var_4}
    var_47 = [var_2, var_46]
    var_48 = [var_45, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = [var_0, var_1, var_2]
    var_52 = [var_4, var_5, var_6]
    var_53 = module_0.no_map_instance(var_51)
    var_54 = module_0.no_map_instance(var_52)
    var_55 = lambda x, y: x + y
    var_56 = [var_53, var_54]
    var_57 = module_0.map_structure_zip(var_55, var_56)
    var_58 = lambda x, y: x + y
    var_59 = [var_0, var_1, var_2]
    var_60 = [var_4, var_5, var_6]
    var_61 = [var_59, var_60]
    var_62 = module_0.map_structure_zip(var_58, var_61)
    var_63 = {var_0, var_1}
    var_64 = {var_2, var_4}
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)



# Parsed testcases at query #65
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = (var_33, var_0)
    var_50 = (var_34, var_1)
    var_51 = [var_49, var_50]
    var_52 = (var_33, var_2)
    var_53 = (var_34, var_4)
    var_54 = [var_52, var_53]
    var_55 = lambda x, y: x + y
    var_56 = module_0.map_structure_zip(var_55, var_46)
    assert var_56 == 4
    var_57 = 0
    var_58 = var_46[var_57]
    var_59 = type(var_58)
    var_60 = module_0.register_no_map_class(var_59)
    var_61 = {var_33: var_0}
    var_62 = {var_33: var_1}
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: len(x) + len(y)
    var_65 = module_0.map_structure_zip(var_64, var_63)
    assert var_65 == 2
    var_66 = {var_0, var_1}
    var_67 = {var_2, var_4}
    var_68 = [var_66, var_67]
    var_69 = lambda x, y: x + y
    var_70 = module_0.map_structure_zip(var_69, var_68)



# Parsed testcases at query #66
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = (var_33, var_0)
    var_46 = (var_34, var_1)
    var_47 = [var_45, var_46]
    var_48 = (var_33, var_2)
    var_49 = (var_34, var_4)
    var_50 = [var_48, var_49]
    var_51 = lambda x, y: OrderedDict([(k, x[k] + y[k]) for k in x.keys()])
    var_52 = module_0.map_structure_zip(var_51, var_44)
    var_53 = (var_33, var_4)
    var_54 = (var_34, var_6)
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: OrderedDict([(k, x[k] + y[k]) for k in x.keys()])
    var_57 = module_0.map_structure_zip(var_56, var_44)
    var_58 = (var_33, var_4)
    var_59 = (var_34, var_6)
    var_60 = [var_58, var_59]
    var_61 = {var_0, var_1}
    var_62 = {var_2, var_4}
    var_63 = [var_61, var_62]
    var_64 = lambda x, y: x + y
    var_65 = module_0.map_structure_zip(var_64, var_63)



# Parsed testcases at query #67
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
    var_11 = (var_0, var_1, var_2)
    var_12 = (var_4, var_5, var_6)
    var_13 = lambda x, y: x + y
    var_14 = [var_11, var_12]
    var_15 = module_0.map_structure_zip(var_13, var_14)
    var_16 = 'Point'
    var_17 = 'x'
    var_18 = 'y'
    var_19 = [var_17, var_18]
    var_20 = lambda x, y: x + y
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_0, var_22: var_1}
    var_24 = {var_21: var_2, var_22: var_4}
    var_25 = lambda x, y: x + y
    var_26 = [var_23, var_24]
    var_27 = module_0.map_structure_zip(var_25, var_26)
    var_28 = [var_0, var_1]
    var_29 = (var_2, var_4)
    var_30 = {var_21: var_28, var_22: var_29}
    var_31 = [var_5, var_6]
    var_32 = 7
    var_33 = 8
    var_34 = (var_32, var_33)
    var_35 = {var_21: var_31, var_22: var_34}
    var_36 = lambda x, y: x + y
    var_37 = [var_30, var_35]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_0, var_1, var_2]
    var_40 = [var_4, var_5, var_6]
    var_41 = lambda x, y: x + y
    var_42 = [var_39, var_40]
    var_43 = module_0.map_structure_zip(var_41, var_42)
    var_44 = [var_0, var_1, var_2]
    var_45 = [var_4, var_5, var_6]
    var_46 = module_0.no_map_instance(var_44)
    var_47 = module_0.no_map_instance(var_45)
    var_48 = lambda x, y: x + y
    var_49 = [var_44, var_45]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = {var_0, var_1, var_2}
    var_52 = {var_4, var_5, var_6}
    var_53 = lambda x, y: x + y
    var_54 = [var_51, var_52]
    var_55 = module_0.map_structure_zip(var_53, var_54)



# Parsed testcases at query #68
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'Point'
    var_28 = 'x'
    var_29 = 'y'
    var_30 = [var_28, var_29]
    var_31 = lambda x, y: x + y
    var_32 = module_0.map_structure_zip(var_31, var_24)
    var_33 = 'a'
    var_34 = 'b'
    var_35 = {var_33: var_0, var_34: var_1}
    var_36 = {var_33: var_2, var_34: var_4}
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_1}
    var_41 = [var_0, var_40]
    var_42 = {var_33: var_4}
    var_43 = [var_2, var_42]
    var_44 = [var_41, var_43]
    var_45 = [var_0, var_1, var_2]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = [var_4, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = lambda x, y: x + y
    var_50 = module_0.map_structure_zip(var_49, var_48)
    var_51 = []
    var_52 = type(var_51)
    var_53 = module_0.register_no_map_class(var_52)
    var_54 = [var_0, var_1, var_2]
    var_55 = [var_4, var_5, var_6]
    var_56 = [var_54, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)



# Parsed testcases at query #69
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *x: sum(x)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_0, var_1]
    var_16 = [var_2, var_4]
    var_17 = [var_15, var_16]
    var_18 = [var_5, var_6]
    var_19 = [var_8, var_9]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = lambda *x: sum(x)
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = (var_0, var_1)
    var_25 = (var_2, var_4)
    var_26 = [var_24, var_25]
    var_27 = lambda *x: sum(x)
    var_28 = module_0.map_structure_zip(var_27, var_26)
    var_29 = 'Point'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = lambda *x: sum(x)
    var_34 = module_0.map_structure_zip(var_33, var_26)
    var_35 = 'a'
    var_36 = 'b'
    var_37 = {var_35: var_0, var_36: var_1}
    var_38 = {var_35: var_2, var_36: var_4}
    var_39 = [var_37, var_38]
    var_40 = lambda *x: sum(x)
    var_41 = module_0.map_structure_zip(var_40, var_39)
    var_42 = (var_1, var_2)
    var_43 = [var_0, var_42]
    var_44 = (var_5, var_6)
    var_45 = [var_4, var_44]
    var_46 = [var_43, var_45]
    var_47 = lambda *x: sum(x)
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_0, var_1, var_2]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_4, var_5, var_6]
    var_52 = [var_50, var_51]
    var_53 = lambda *x: sum(x)
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1]
    var_56 = [var_2, var_4]
    var_57 = lambda *x: sum(x)
    var_58 = module_0.map_structure_zip(var_57, var_52)
    var_59 = {var_0, var_1}
    var_60 = {var_2, var_4}
    var_61 = [var_59, var_60]
    var_62 = lambda *x: sum(x)
    var_63 = module_0.map_structure_zip(var_62, var_61)



# Parsed testcases at query #70
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
    var_7 = (var_2, var_0, var_3)
    var_8 = module_0.map_structure(var_6, var_7)
    var_9 = lambda x: x * var_0
    var_10 = [var_0, var_3]
    var_11 = [var_2, var_10]
    var_12 = module_0.map_structure(var_9, var_11)
    var_13 = lambda x: x * var_0
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_2, var_15: var_0}
    var_17 = module_0.map_structure(var_13, var_16)
    var_18 = lambda x: x * var_0
    var_19 = {var_2, var_0, var_3}
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = [var_2, var_0, var_3]
    var_22 = module_0.no_map_instance(var_21)
    var_23 = lambda x: x * var_0
    var_24 = module_0.map_structure(var_23, var_22)
    var_25 = lambda x: x * var_0
    var_26 = [var_2, var_0, var_3]
    var_27 = module_0.map_structure(var_25, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda x: x * var_0
    var_33 = 4
    var_34 = lambda x: x * var_0
    var_35 = 5
    var_36 = module_0.map_structure(var_34, var_35)
    assert var_36 == 10



# Parsed testcases at query #71
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
    var_7 = [var_0, var_3]
    var_8 = [var_2, var_7]
    var_9 = module_0.map_structure(var_6, var_8)
    var_10 = lambda x: x * var_0
    var_11 = (var_2, var_0, var_3)
    var_12 = module_0.map_structure(var_10, var_11)
    var_13 = 'Point'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = lambda x: x * var_0
    var_18 = 4
    var_19 = lambda x: x * var_0
    var_20 = 'a'
    var_21 = 'b'
    var_22 = {var_20: var_2, var_21: var_0}
    var_23 = module_0.map_structure(var_19, var_22)
    var_24 = lambda x: x * var_0
    var_25 = {var_2, var_0, var_3}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = [var_2, var_0, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = type(var_27)
    var_32 = module_0.register_no_map_class(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_27)
    var_35 = (var_0, var_3)
    var_36 = {var_20: var_18}
    var_37 = [var_2, var_35, var_36]
    var_38 = lambda x: x * var_0
    var_39 = module_0.map_structure(var_38, var_37)



# Parsed testcases at query #72
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = [var_0, var_1]
    var_23 = [var_2, var_4]
    var_24 = (var_22, var_23)
    var_25 = [var_5, var_6]
    var_26 = [var_15, var_16]
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_1}
    var_46 = [var_0, var_45]
    var_47 = {var_38: var_4}
    var_48 = [var_2, var_47]
    var_49 = [var_46, var_48]
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_4, var_5, var_6]
    var_53 = [var_51, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = type(var_50)
    var_57 = module_0.register_no_map_class(var_56)
    var_58 = [var_4, var_5, var_6]
    var_59 = [var_50, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = {var_0, var_1}
    var_63 = {var_2, var_4}
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)



# Parsed testcases at query #73
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
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = 0
    var_30 = lambda x, y: (x[var_29] + y[var_29], x[var_0] + y[var_0])
    var_31 = module_0.map_structure_zip(var_30, var_28)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_37 = module_0.map_structure_zip(var_36, var_28)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = {var_38: var_0, var_39: var_1}
    var_41 = {var_38: var_2, var_39: var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_38: var_0, var_39: var_1}
    var_46 = 'c'
    var_47 = {var_46: var_2}
    var_48 = [var_45, var_47]
    var_49 = {var_38: var_4, var_39: var_5}
    var_50 = {var_46: var_6}
    var_51 = [var_49, var_50]
    var_52 = [var_48, var_51]
    var_53 = lambda x, y: {k: x[k] + y[k] for k in x}
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = [var_0, var_1, var_2]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_56, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = [var_0, var_1, var_2]
    var_61 = type(var_60)
    var_62 = module_0.register_no_map_class(var_61)
    var_63 = [var_0, var_1, var_2]
    var_64 = [var_4, var_5, var_6]
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)
    var_68 = {var_0, var_1}
    var_69 = {var_2, var_4}
    var_70 = [var_68, var_69]
    var_71 = lambda x, y: x + y
    var_72 = module_0.map_structure_zip(var_71, var_70)



# Parsed testcases at query #74
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
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]
    var_13 = lambda *x: sum(x)
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = (var_0, var_1)
    var_16 = (var_2, var_4)
    var_17 = (var_5, var_6)
    var_18 = [var_15, var_16, var_17]
    var_19 = lambda *x: sum(x)
    var_20 = module_0.map_structure_zip(var_19, var_18)
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = lambda *x: sum(x)
    var_26 = module_0.map_structure_zip(var_25, var_18)
    var_27 = 12
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_0, var_29: var_1}
    var_31 = {var_28: var_2, var_29: var_4}
    var_32 = {var_28: var_5, var_29: var_6}
    var_33 = [var_30, var_31, var_32]
    var_34 = lambda *x: sum(x)
    var_35 = module_0.map_structure_zip(var_34, var_33)
    var_36 = [var_0, var_1]
    var_37 = [var_2, var_4]
    var_38 = [var_5, var_6]
    var_39 = [var_36, var_37, var_38]
    var_40 = lambda *x: sum(x)
    var_41 = module_0.map_structure_zip(var_40, var_39)
    var_42 = [var_0, var_1, var_2]
    var_43 = module_0.no_map_instance(var_42)
    var_44 = [var_4, var_5, var_6]
    var_45 = [var_43, var_44]
    var_46 = lambda *x: sum(x)
    var_47 = module_0.map_structure_zip(var_46, var_45)
    var_48 = [var_0, var_1, var_2]
    var_49 = [var_4, var_5, var_6]
    var_50 = [var_48, var_49]
    var_51 = lambda *x: sum(x)
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = {var_0, var_1}
    var_54 = {var_2, var_4}
    var_55 = [var_53, var_54]
    var_56 = lambda *x: sum(x)
    var_57 = module_0.map_structure_zip(var_56, var_55)



