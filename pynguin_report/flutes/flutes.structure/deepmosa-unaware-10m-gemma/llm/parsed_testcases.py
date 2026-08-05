####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
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
    var_9 = lambda x, y: x * y
    var_10 = [var_1, var_2]
    var_11 = (var_4, var_5)
    var_12 = (var_10, var_11)
    var_13 = 5
    var_14 = 6
    var_15 = [var_13, var_14]
    var_16 = 7
    var_17 = 8
    var_18 = (var_16, var_17)
    var_19 = (var_15, var_18)
    var_20 = [var_12, var_19]
    var_21 = module_0.map_structure_zip(var_9, var_20)
    var_22 = lambda x, y: x - y
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 10
    var_26 = 20
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = {var_23: var_1, var_24: var_2}
    var_29 = [var_27, var_28]
    var_30 = module_0.map_structure_zip(var_22, var_29)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_36 = lambda x, y: (x, y)
    var_37 = [var_1, var_2]
    var_38 = module_0.map_structure_zip(var_36, var_37)
    var_39 = [var_1, var_2]
    var_40 = [var_4, var_5]
    var_41 = [var_39, var_40]
    var_42 = module_0.map_structure_zip(var_0, var_41)
    var_43 = 1
    var_44 = 2
    var_45 = {var_43, var_44}
    var_46 = 3
    var_47 = 4
    var_48 = {var_46, var_47}
    var_49 = [var_45, var_48]
    var_50 = module_0.map_structure_zip(var_0, var_49)
    var_51 = lambda x, y: str(x) + str(y)
    var_52 = {var_23: var_43}
    var_53 = (var_44,)
    var_54 = [var_52, var_53]
    var_55 = {var_24: var_44}
    var_56 = (var_46,)
    var_57 = [var_55, var_56]
    var_58 = [var_54, var_57]
    var_59 = 'val'
    var_60 = {var_59: var_43}
    var_61 = (var_44,)
    var_62 = [var_60, var_61]
    var_63 = {var_59: var_44}
    var_64 = (var_46,)
    var_65 = [var_63, var_64]
    var_66 = [var_62, var_65]
    var_67 = module_0.map_structure_zip(var_51, var_66)



# Parsed testcases at query #2
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
    var_9 = lambda x, y: x * y
    var_10 = (var_2, var_4)
    var_11 = 5
    var_12 = [var_5, var_11]
    var_13 = [var_1, var_10, var_12]
    var_14 = 10
    var_15 = 20
    var_16 = 30
    var_17 = (var_15, var_16)
    var_18 = 40
    var_19 = 50
    var_20 = [var_18, var_19]
    var_21 = [var_14, var_17, var_20]
    var_22 = 90
    var_23 = (var_18, var_22)
    var_24 = 160
    var_25 = 250
    var_26 = [var_24, var_25]
    var_27 = [var_14, var_23, var_26]
    var_28 = [var_13, var_21]
    var_29 = module_0.map_structure_zip(var_9, var_28)
    var_30 = lambda x, y: x + y
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 'c'
    var_34 = {var_33: var_2}
    var_35 = {var_31: var_1, var_32: var_34}
    var_36 = {var_33: var_15}
    var_37 = {var_31: var_14, var_32: var_36}
    var_38 = 11
    var_39 = 22
    var_40 = {var_33: var_39}
    var_41 = {var_31: var_38, var_32: var_40}
    var_42 = [var_35, var_37]
    var_43 = module_0.map_structure_zip(var_30, var_42)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_49 = 6
    var_50 = lambda x, y: x.val + y.val
    var_51 = lambda *args: sum(args)
    var_52 = [var_1, var_2]
    var_53 = [var_4, var_5]
    var_54 = [var_52, var_53]
    var_55 = module_0.map_structure_zip(var_51, var_54)
    assert var_55 == 10
    var_56 = lambda x: x
    var_57 = 1
    var_58 = {var_57}
    var_59 = 2
    var_60 = {var_59}
    var_61 = [var_58, var_60]
    var_62 = module_0.map_structure_zip(var_56, var_61)
    var_63 = lambda x, y: (x, y)
    var_64 = [var_56, var_57]
    var_65 = module_0.map_structure_zip(var_63, var_64)



# Parsed testcases at query #3
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda *args: sum(args)
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = [var_1]
    var_13 = [var_2]
    var_14 = [var_12, var_13]
    var_15 = [var_4]
    var_16 = [var_5]
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_1, var_20: var_2}
    var_22 = 10
    var_23 = 20
    var_24 = {var_19: var_22, var_20: var_23}
    var_25 = [var_21, var_24]
    var_26 = 11
    var_27 = 22
    var_28 = {var_19: var_26, var_20: var_27}
    var_29 = module_0.map_structure_zip(var_0, var_25)
    var_30 = (var_1, var_2)
    var_31 = (var_4, var_5)
    var_32 = [var_30, var_31]
    var_33 = module_0.map_structure_zip(var_0, var_32)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = (var_19, var_1)
    var_39 = [var_38]
    var_40 = (var_19, var_2)
    var_41 = [var_40]
    var_42 = (var_19, var_4)
    var_43 = [var_42]
    var_44 = [var_1, var_2]
    var_45 = [var_4, var_5]
    var_46 = [var_1, var_2]
    var_47 = [var_4, var_5]
    var_48 = {var_1}
    var_49 = {var_2}
    var_50 = [var_48, var_49]
    var_51 = module_0.map_structure_zip(var_0, var_50)



# Parsed testcases at query #4
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = [var_1, var_2]
    var_12 = [var_3, var_5]
    var_13 = [var_11, var_12]
    var_14 = 10
    var_15 = 20
    var_16 = [var_14, var_15]
    var_17 = 30
    var_18 = 40
    var_19 = [var_17, var_18]
    var_20 = [var_16, var_19]
    var_21 = [var_13, var_20]
    var_22 = module_0.map_structure_zip(var_0, var_21)
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_1, var_24: var_2}
    var_26 = {var_23: var_14, var_24: var_15}
    var_27 = [var_25, var_26]
    var_28 = module_0.map_structure_zip(var_0, var_27)
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_5)
    var_31 = [var_29, var_30]
    var_32 = module_0.map_structure_zip(var_0, var_31)
    var_33 = 'Point'
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = 'val'
    var_38 = {var_37: var_1}
    var_39 = {var_37: var_2}
    var_40 = [var_38, var_39]
    var_41 = {var_37: var_14}
    var_42 = {var_37: var_15}
    var_43 = [var_41, var_42]
    var_44 = [var_40, var_43]
    var_45 = module_0.map_structure_zip(var_0, var_44)
    var_46 = lambda x, y, z: x * y * z
    var_47 = [var_1]
    var_48 = [var_2]
    var_49 = [var_3]
    var_50 = [var_1, var_2]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = [var_14, var_15]
    var_53 = lambda x, y: [x, y]
    var_54 = [var_14, var_15]
    var_55 = [var_51, var_54]
    var_56 = module_0.map_structure_zip(var_53, var_55)
    var_57 = {var_1, var_2}
    var_58 = {var_3, var_5}
    var_59 = [var_57, var_58]
    var_60 = module_0.map_structure_zip(var_0, var_59)



# Parsed testcases at query #5
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
    var_7 = [var_0]
    var_8 = [var_3]
    var_9 = [var_2, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_0, var_2)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 0
    var_16 = lambda x: x * var_15
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_0, var_18: var_2}
    var_20 = module_0.map_structure(var_16, var_19)
    var_21 = lambda x: x + var_0
    var_22 = 'c'
    var_23 = {var_22: var_2}
    var_24 = {var_17: var_0, var_18: var_23}
    var_25 = module_0.map_structure(var_21, var_24)
    var_26 = lambda x: x + var_0
    var_27 = {var_0, var_2}
    var_28 = module_0.map_structure(var_26, var_27)
    var_29 = 'Point'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = 10
    var_34 = lambda x: x * var_33
    var_35 = [var_0, var_2, var_3]
    var_36 = lambda x: x.val
    var_37 = 5
    var_38 = lambda x: x.val + var_0
    var_39 = lambda x: x + var_0
    var_40 = module_0.map_structure(var_39, var_37)
    assert var_40 == 6
    var_41 = (var_2, var_3)
    var_42 = {var_17: var_41}
    var_43 = 4
    var_44 = {var_43, var_37}
    var_45 = [var_0, var_42, var_44]
    var_46 = (var_3, var_43)
    var_47 = {var_17: var_46}
    var_48 = 6
    var_49 = {var_37, var_48}
    var_50 = [var_2, var_47, var_49]
    var_51 = lambda x: x + var_0
    var_52 = module_0.map_structure(var_51, var_45)
    var_53 = var_52[var_2]
    var_54 = set(var_53)



# Parsed testcases at query #6
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = lambda x, y, z: x + y + z
    var_10 = [var_1]
    var_11 = [var_3]
    var_12 = [var_5]
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.map_structure_zip(var_9, var_13)
    var_15 = [var_1, var_3]
    var_16 = [var_5]
    var_17 = [var_15, var_16]
    var_18 = 4
    var_19 = 5
    var_20 = [var_18, var_19]
    var_21 = 6
    var_22 = [var_21]
    var_23 = [var_20, var_22]
    var_24 = [var_17, var_23]
    var_25 = 7
    var_26 = [var_19, var_25]
    var_27 = 9
    var_28 = [var_27]
    var_29 = [var_26, var_28]
    var_30 = lambda x, y: x + y
    var_31 = module_0.map_structure_zip(var_30, var_24)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_1, var_33: var_3}
    var_35 = 10
    var_36 = 20
    var_37 = {var_32: var_35, var_33: var_36}
    var_38 = [var_34, var_37]
    var_39 = 11
    var_40 = 22
    var_41 = {var_32: var_39, var_33: var_40}
    var_42 = lambda x, y: x + y
    var_43 = module_0.map_structure_zip(var_42, var_38)
    var_44 = (var_1, var_3)
    var_45 = (var_5, var_18)
    var_46 = [var_44, var_45]
    var_47 = lambda x, y: x + y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = 'Point'
    var_50 = 'x'
    var_51 = 'y'
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: x + y
    var_54 = lambda x, y: x.val + y.val
    var_55 = lambda x, y: x + y
    var_56 = 1
    var_57 = {var_56}
    var_58 = 2
    var_59 = {var_58}
    var_60 = [var_57, var_59]
    var_61 = module_0.map_structure_zip(var_55, var_60)
    var_62 = lambda a, b, c: a * b * c
    var_63 = [var_58]
    var_64 = [var_60]
    var_65 = [var_18]
    var_66 = [var_63, var_64, var_65]
    var_67 = module_0.map_structure_zip(var_62, var_66)



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
    var_6 = lambda x: x + var_2
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 10
    var_16 = lambda x: x * var_15
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_2, var_18: var_0}
    var_20 = module_0.map_structure(var_16, var_19)
    var_21 = [var_2, var_0]
    var_22 = 4
    var_23 = (var_3, var_22)
    var_24 = {var_17: var_21, var_18: var_23}
    var_25 = [var_0, var_22]
    var_26 = 6
    var_27 = 8
    var_28 = (var_26, var_27)
    var_29 = {var_17: var_25, var_18: var_28}
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_24)
    var_32 = lambda x: x + var_2
    var_33 = {var_2, var_0, var_3}
    var_34 = module_0.map_structure(var_32, var_33)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = 5
    var_40 = lambda x: x * var_39
    var_41 = [var_2, var_0]
    var_42 = lambda x: x.value
    var_43 = [var_2, var_0]
    var_44 = lambda x: len(x)
    var_45 = lambda x: x + var_2
    var_46 = module_0.map_structure(var_45, var_39)
    assert var_46 == 6
    var_47 = (var_2, var_0)
    var_48 = {var_17: var_47}
    var_49 = {var_3, var_22}
    var_50 = [var_48, var_49]
    var_51 = lambda x: x * var_0
    var_52 = module_0.map_structure(var_51, var_50)



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
    var_6 = [var_0, var_3]
    var_7 = 4
    var_8 = [var_2, var_6, var_7]
    var_9 = lambda x: x + var_2
    var_10 = module_0.map_structure(var_9, var_8)
    var_11 = lambda x: x * var_3
    var_12 = (var_2, var_0, var_3)
    var_13 = module_0.map_structure(var_11, var_12)
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_0, var_3]
    var_17 = {var_14: var_2, var_15: var_16}
    var_18 = 6
    var_19 = [var_7, var_18]
    var_20 = {var_14: var_0, var_15: var_19}
    var_21 = lambda x: x * var_0
    var_22 = module_0.map_structure(var_21, var_17)
    var_23 = {var_2, var_0, var_3}
    var_24 = lambda x: x + var_2
    var_25 = module_0.map_structure(var_24, var_23)
    var_26 = 'Point'
    var_27 = 'x'
    var_28 = 'y'
    var_29 = [var_27, var_28]
    var_30 = 10
    var_31 = lambda x: x * var_30
    var_32 = 20
    var_33 = 5
    var_34 = lambda x: x + var_33
    var_35 = module_0.map_structure(var_34, var_30)
    assert var_35 == 15
    var_36 = [var_2, var_0, var_3]
    var_37 = [var_2, var_0]
    var_38 = lambda x: x.val
    var_39 = (var_0, var_3)
    var_40 = {var_14: var_39}
    var_41 = {var_7, var_33}
    var_42 = [var_2, var_40, var_41]
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_42)



# Parsed testcases at query #9
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
    var_9 = lambda x, y: x * y
    var_10 = (var_1, var_2)
    var_11 = [var_10, var_4]
    var_12 = 5
    var_13 = (var_5, var_12)
    var_14 = 6
    var_15 = [var_13, var_14]
    var_16 = [var_11, var_15]
    var_17 = module_0.map_structure_zip(var_9, var_16)
    var_18 = lambda x, y: x - y
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 10
    var_22 = 20
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = {var_19: var_1, var_20: var_2}
    var_25 = [var_23, var_24]



# Parsed testcases at query #10
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = [var_1, var_2]
    var_12 = [var_3]
    var_13 = [var_11, var_12]
    var_14 = 10
    var_15 = 20
    var_16 = [var_14, var_15]
    var_17 = 30
    var_18 = [var_17]
    var_19 = [var_16, var_18]
    var_20 = [var_13, var_19]
    var_21 = module_0.map_structure_zip(var_0, var_20)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = {var_22: var_1, var_23: var_2}
    var_25 = {var_22: var_14, var_23: var_15}
    var_26 = [var_24, var_25]
    var_27 = module_0.map_structure_zip(var_0, var_26)
    var_28 = (var_1, var_2)
    var_29 = (var_3, var_5)
    var_30 = [var_28, var_29]
    var_31 = module_0.map_structure_zip(var_0, var_30)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = 7
    var_37 = 8
    var_38 = 9
    var_39 = [var_36, var_37, var_38]
    var_40 = [var_4, var_8, var_39]
    var_41 = module_0.map_structure_zip(var_0, var_40)
    var_42 = [var_1, var_2]
    var_43 = [var_3, var_5]
    var_44 = lambda x, y: x.val + y.val
    var_45 = lambda x, y: x.val + y.val
    var_46 = {var_1, var_2}
    var_47 = {var_3, var_5}
    var_48 = [var_46, var_47]
    var_49 = module_0.map_structure_zip(var_0, var_48)
    var_50 = [var_48]
    var_51 = {var_22: var_50}
    var_52 = (var_49,)
    var_53 = [var_51, var_52]
    var_54 = [var_14]
    var_55 = {var_22: var_54}
    var_56 = (var_15,)
    var_57 = [var_55, var_56]
    var_58 = [var_53, var_57]
    var_59 = module_0.map_structure_zip(var_0, var_58)



# Parsed testcases at query #11
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda *args: args
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = lambda x, y: x + y
    var_10 = (var_2, var_4)
    var_11 = [var_1, var_10]
    var_12 = 10
    var_13 = 20
    var_14 = 30
    var_15 = (var_13, var_14)
    var_16 = [var_12, var_15]
    var_17 = [var_11, var_16]
    var_18 = 11
    var_19 = 22
    var_20 = 33
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = [var_22]
    var_24 = module_0.map_structure_zip(var_9, var_17)
    var_25 = lambda x, y: x * y
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_2, var_27: var_4}
    var_29 = 5
    var_30 = {var_26: var_29, var_27: var_12}
    var_31 = [var_28, var_30]
    var_32 = {var_26: var_12, var_27: var_14}
    var_33 = [var_32]
    var_34 = module_0.map_structure_zip(var_25, var_31)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = lambda x, y: x + y
    var_40 = 6
    var_41 = [var_1, var_2]
    var_42 = [var_4, var_5]
    var_43 = 1
    var_44 = {var_43}
    var_45 = 2
    var_46 = {var_45}
    var_47 = [var_44, var_46]
    var_48 = module_0.map_structure_zip(var_0, var_47)
    var_49 = {var_26: var_44}
    var_50 = [var_43, var_49]
    var_51 = {var_26: var_13}
    var_52 = [var_12, var_51]
    var_53 = [var_50, var_52]
    var_54 = module_0.map_structure_zip(var_9, var_53)
    var_55 = [var_43, var_44]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_12, var_13]
    var_58 = [var_56, var_57]
    var_59 = lambda x, y: x + y
    var_60 = lambda x, y: (len(x), len(y))
    var_61 = [var_12, var_13]
    var_62 = [var_56, var_61]
    var_63 = module_0.map_structure_zip(var_60, var_62)



# Parsed testcases at query #12
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.map_structure_zip(var_0, var_4)
    var_6 = lambda x, y: x + y
    var_7 = [var_1, var_2]
    var_8 = module_0.map_structure_zip(var_6, var_7)
    assert var_8 == 3
    var_9 = [var_1, var_2, var_3]
    var_10 = 4
    var_11 = 5
    var_12 = 6
    var_13 = [var_10, var_11, var_12]
    var_14 = lambda x, y: x + y
    var_15 = [var_9, var_13]
    var_16 = module_0.map_structure_zip(var_14, var_15)
    var_17 = [var_1, var_2]
    var_18 = [var_3]
    var_19 = [var_17, var_18]
    var_20 = 10
    var_21 = 20
    var_22 = [var_20, var_21]
    var_23 = 30
    var_24 = [var_23]
    var_25 = [var_22, var_24]
    var_26 = lambda x, y: x + y
    var_27 = [var_19, var_25]
    var_28 = module_0.map_structure_zip(var_26, var_27)
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_10)
    var_31 = lambda x, y: x * y
    var_32 = [var_29, var_30]
    var_33 = module_0.map_structure_zip(var_31, var_32)
    var_34 = 'a'
    var_35 = 'b'
    var_36 = {var_34: var_1, var_35: var_2}
    var_37 = {var_34: var_20, var_35: var_21}
    var_38 = lambda x, y: x - y
    var_39 = [var_36, var_37]
    var_40 = module_0.map_structure_zip(var_38, var_39)
    var_41 = 'Point'
    var_42 = 'x'
    var_43 = 'y'
    var_44 = [var_42, var_43]
    var_45 = lambda x, y, z: x + y + z
    var_46 = lambda x, y: len(x) + len(y)
    var_47 = [var_1, var_2, var_3]
    var_48 = [var_10, var_11, var_12]
    var_49 = [var_47, var_48]
    var_50 = module_0.map_structure_zip(var_46, var_49)
    assert var_50 == 6
    var_51 = {var_1, var_2}
    var_52 = {var_3, var_10}
    var_53 = lambda x, y: x + y
    var_54 = [var_51, var_52]
    var_55 = module_0.map_structure_zip(var_53, var_54)
    var_56 = [var_54, var_55]
    var_57 = {var_34: var_56}
    var_58 = (var_3, var_10)
    var_59 = [var_57, var_58]
    var_60 = [var_20, var_21]
    var_61 = {var_34: var_60}
    var_62 = 40
    var_63 = (var_23, var_62)
    var_64 = [var_61, var_63]
    var_65 = 11
    var_66 = 22
    var_67 = [var_65, var_66]
    var_68 = {var_34: var_67}
    var_69 = 33
    var_70 = 44
    var_71 = (var_69, var_70)
    var_72 = [var_68, var_71]
    var_73 = lambda x, y: x + y
    var_74 = [var_59, var_64]
    var_75 = module_0.map_structure_zip(var_73, var_74)



# Parsed testcases at query #13
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
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 10
    var_16 = lambda x: x * var_15
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_2, var_18: var_0}
    var_20 = module_0.map_structure(var_16, var_19)
    var_21 = [var_2, var_0]
    var_22 = 4
    var_23 = (var_3, var_22)
    var_24 = {var_17: var_21, var_18: var_23}
    var_25 = [var_0, var_22]
    var_26 = 6
    var_27 = 8
    var_28 = (var_26, var_27)
    var_29 = {var_17: var_25, var_18: var_28}
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_24)
    var_32 = {var_2, var_0, var_3}
    var_33 = lambda x: x + var_2
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = 5
    var_40 = lambda x: x * var_39
    var_41 = lambda x: x + var_2
    var_42 = module_0.map_structure(var_41, var_15)
    assert var_42 == 11
    var_43 = [var_2, var_0]
    var_44 = lambda x: len(x)
    var_45 = [var_2, var_0, var_3]
    var_46 = module_0.no_map_instance(var_45)
    var_47 = lambda x: len(x)
    var_48 = module_0.map_structure(var_47, var_46)
    assert var_48 == 3
    var_49 = (var_2, var_0)
    var_50 = {var_17: var_49}
    var_51 = [var_3, var_22]
    var_52 = {var_18: var_51}
    var_53 = [var_50, var_52, var_39]
    var_54 = (var_0, var_22)
    var_55 = {var_17: var_54}
    var_56 = [var_26, var_27]
    var_57 = {var_18: var_56}
    var_58 = [var_55, var_57, var_15]
    var_59 = lambda x: x * var_0
    var_60 = module_0.map_structure(var_59, var_53)



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
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)
    var_9 = [var_0]
    var_10 = [var_9]
    var_11 = [var_1]
    var_12 = [var_11]
    var_13 = [var_10, var_12]
    var_14 = 0
    var_15 = lambda x: x[var_14][var_14] + var_0
    var_16 = module_0.map_structure_zip(var_15, var_13)
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_0, var_18: var_1}
    var_20 = 10
    var_21 = 20
    var_22 = {var_17: var_20, var_18: var_21}
    var_23 = [var_19, var_22]
    var_24 = lambda x, y: x + y
    var_25 = module_0.map_structure_zip(var_24, var_23)
    var_26 = (var_0, var_1)
    var_27 = (var_3, var_4)
    var_28 = [var_26, var_27]
    var_29 = lambda x, y: x * y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_36 = 6
    var_37 = [var_0, var_1, var_3]
    var_38 = lambda x, y, z: x + y + z
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = [var_0, var_1]
    var_41 = {var_17: var_40}
    var_42 = [var_3, var_4]
    var_43 = {var_17: var_42}
    var_44 = [var_41, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = 99
    var_48 = 100
    var_49 = [var_47, var_48]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = [var_0, var_1]
    var_52 = [var_50, var_51]
    var_53 = lambda x, y: len(x) + sum(y)
    var_54 = module_0.map_structure_zip(var_53, var_52)
    var_55 = {var_0}
    var_56 = {var_1}
    var_57 = [var_55, var_56]
    var_58 = lambda x, y: x + y
    var_59 = module_0.map_structure_zip(var_58, var_57)
    var_60 = [var_58, var_59]
    var_61 = [var_3, var_4]
    var_62 = [var_60, var_61]
    var_63 = lambda x, y: sum(x) + sum(y)
    var_64 = module_0.map_structure_zip(var_63, var_62)



# Parsed testcases at query #15
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 2
    var_3 = 2
    var_4 = lambda x: x * var_3
    var_5 = 3
    var_6 = [var_0, var_3, var_5]
    var_7 = module_0.map_structure(var_4, var_6)
    var_8 = lambda x: x + var_0
    var_9 = [var_0]
    var_10 = [var_5]
    var_11 = [var_3, var_10]
    var_12 = [var_9, var_11]
    var_13 = module_0.map_structure(var_8, var_12)
    var_14 = lambda x: x * var_3
    var_15 = (var_0, var_3)
    var_16 = module_0.map_structure(var_14, var_15)
    var_17 = lambda x: x + var_0
    var_18 = (var_3, var_5)
    var_19 = (var_0, var_18)
    var_20 = module_0.map_structure(var_17, var_19)
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = 10
    var_26 = lambda x: x * var_25
    var_27 = 20
    var_28 = 'a'
    var_29 = 'b'
    var_30 = [var_3, var_5]
    var_31 = {var_28: var_0, var_29: var_30}
    var_32 = 4
    var_33 = 6
    var_34 = [var_32, var_33]
    var_35 = {var_28: var_3, var_29: var_34}
    var_36 = lambda x: x * var_3
    var_37 = module_0.map_structure(var_36, var_31)
    var_38 = (var_0, var_3)
    var_39 = 'c'
    var_40 = {var_39: var_5}
    var_41 = {var_28: var_38, var_29: var_40}
    var_42 = (var_3, var_32)
    var_43 = {var_39: var_33}
    var_44 = {var_28: var_42, var_29: var_43}
    var_45 = lambda x: x * var_3
    var_46 = module_0.map_structure(var_45, var_41)
    var_47 = {var_0, var_3, var_5}
    var_48 = lambda x: x + var_0
    var_49 = module_0.map_structure(var_48, var_47)
    var_50 = lambda x: len(x)
    var_51 = {var_0, var_3, var_5}
    var_52 = module_0.map_structure(var_50, var_51)
    assert var_52 == 3
    var_53 = [var_0, var_3]
    var_54 = lambda x: x.val
    var_55 = (var_3, var_5)
    var_56 = {var_28: var_55}
    var_57 = 5
    var_58 = {var_32, var_57}
    var_59 = [var_0, var_56, var_58]



# Parsed testcases at query #16
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
    var_7 = module_0.map_structure_zip(var_0, var_3)
    var_8 = [var_1, var_2]
    var_9 = [var_4]
    var_10 = [var_8, var_9]
    var_11 = 10
    var_12 = 20
    var_13 = [var_11, var_12]
    var_14 = 30
    var_15 = [var_14]
    var_16 = [var_13, var_15]
    var_17 = lambda x, y: x + y
    var_18 = module_0.map_structure_zip(var_17, var_10)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = {var_21: var_2}
    var_23 = {var_19: var_1, var_20: var_22}
    var_24 = {var_21: var_12}
    var_25 = {var_19: var_11, var_20: var_24}
    var_26 = lambda x, y: x + y
    var_27 = [var_23]
    var_28 = [var_25]
    var_29 = module_0.map_structure_zip(var_26, var_27)
    var_30 = lambda x, y: x + y
    var_31 = {var_19: var_1}
    var_32 = {var_19: var_2}
    var_33 = [var_31, var_32]
    var_34 = {var_19: var_11}
    var_35 = {var_19: var_12}
    var_36 = [var_34, var_35]
    var_37 = module_0.map_structure_zip(var_30, var_33)
    var_38 = (var_2, var_4)
    var_39 = (var_1, var_38)
    var_40 = (var_12, var_14)
    var_41 = (var_11, var_40)
    var_42 = lambda x, y: x + y
    var_43 = [var_39]
    var_44 = [var_41]
    var_45 = module_0.map_structure_zip(var_42, var_43)
    var_46 = 'Point'
    var_47 = 'x'
    var_48 = 'y'
    var_49 = [var_47, var_48]
    var_50 = lambda x, y: x + y
    var_51 = 11
    var_52 = 22
    var_53 = [var_1, var_2]
    var_54 = [var_11, var_12]
    var_55 = lambda x, y: x + y
    var_56 = [var_51, var_52]
    var_57 = lambda x, y: x + y
    var_58 = [var_1, var_2]
    var_59 = [var_11, var_12]
    var_60 = [var_1, var_2]
    var_61 = [var_11, var_12]
    var_62 = lambda x, y: len(x) + len(y)
    var_63 = [var_1]
    var_64 = [var_1, var_2, var_4]
    var_65 = lambda x, y: x + y
    var_66 = 1
    var_67 = {var_66}
    var_68 = 2
    var_69 = {var_68}
    var_70 = [var_67, var_69]
    var_71 = 3
    var_72 = {var_71}
    var_73 = 4
    var_74 = {var_73}
    var_75 = [var_72, var_74]
    var_76 = module_0.map_structure_zip(var_65, var_70)
    var_77 = [var_66, var_67]
    var_78 = module_0.no_map_instance(var_77)
    var_79 = lambda x, y: x + y
    var_80 = [var_78]
    var_81 = [var_75, var_76]
    var_82 = [var_81]
    var_83 = module_0.map_structure_zip(var_79, var_80)
    var_84 = lambda x, y: x + y
    var_85 = [var_66]
    var_86 = module_0.no_map_instance(var_85)
    var_87 = [var_86]
    var_88 = [var_67]
    var_89 = [var_88]
    var_90 = module_0.map_structure_zip(var_84, var_87)



# Parsed testcases at query #17
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = (var_12, var_13)
    var_15 = (var_6, var_7)
    var_16 = [var_14, var_15]
    var_17 = 10
    var_18 = 20
    var_19 = [var_17, var_18]
    var_20 = 30
    var_21 = 40
    var_22 = [var_20, var_21]
    var_23 = (var_19, var_22)
    var_24 = 7
    var_25 = 8
    var_26 = (var_24, var_25)
    var_27 = [var_23, var_26]
    var_28 = [var_16, var_27]
    var_29 = [var_17, var_18]
    var_30 = [var_20, var_21]
    var_31 = (var_29, var_30)
    var_32 = 35
    var_33 = 48
    var_34 = (var_32, var_33)
    var_35 = [var_31, var_34]
    var_36 = [var_35]
    var_37 = module_0.map_structure_zip(var_11, var_28)
    var_38 = lambda x, y: x - y
    var_39 = 'a'
    var_40 = 'b'
    var_41 = {var_39: var_17, var_40: var_18}
    var_42 = {var_39: var_6, var_40: var_2}
    var_43 = [var_41, var_42]
    var_44 = 18
    var_45 = {var_39: var_6, var_40: var_44}
    var_46 = module_0.map_structure_zip(var_38, var_43)



# Parsed testcases at query #18
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_1, var_2, var_5]
    var_7 = module_0.map_structure_zip(var_0, var_6)
    var_8 = [var_2, var_3]
    var_9 = [var_1, var_8, var_4]
    var_10 = 10
    var_11 = 20
    var_12 = 30
    var_13 = [var_11, var_12]
    var_14 = 40
    var_15 = [var_10, var_13, var_14]
    var_16 = lambda x, y: x + y
    var_17 = [var_9, var_15]
    var_18 = module_0.map_structure_zip(var_16, var_17)
    var_19 = (var_2, var_3)
    var_20 = (var_1, var_19)
    var_21 = (var_11, var_12)
    var_22 = (var_10, var_21)
    var_23 = lambda x, y: x + y
    var_24 = [var_20, var_22]
    var_25 = module_0.map_structure_zip(var_23, var_24)
    var_26 = 'Point'
    var_27 = 'x'
    var_28 = 'y'
    var_29 = [var_27, var_28]
    var_30 = 100
    var_31 = 200
    var_32 = lambda a, b, c: a + b + c
    var_33 = 111
    var_34 = 222
    var_35 = 'a'
    var_36 = 'b'
    var_37 = [var_2, var_3]
    var_38 = {var_35: var_1, var_36: var_37}
    var_39 = [var_11, var_12]
    var_40 = {var_35: var_10, var_36: var_39}
    var_41 = lambda x, y: x + y
    var_42 = [var_38, var_40]
    var_43 = module_0.map_structure_zip(var_41, var_42)
    var_44 = [var_1, var_2]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = [var_10, var_11]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_45, var_47]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = {var_1, var_2}
    var_52 = {var_3, var_4}
    var_53 = lambda x, y: x + y
    var_54 = [var_51, var_52]
    var_55 = module_0.map_structure_zip(var_53, var_54)
    var_56 = [var_54]
    var_57 = (var_55, var_3)
    var_58 = {var_35: var_57}
    var_59 = [var_56, var_58]
    var_60 = [var_10]
    var_61 = (var_11, var_12)
    var_62 = {var_35: var_61}
    var_63 = [var_60, var_62]
    var_64 = 11
    var_65 = [var_64]
    var_66 = 22
    var_67 = 33
    var_68 = (var_66, var_67)
    var_69 = {var_35: var_68}
    var_70 = [var_65, var_69]
    var_71 = lambda x, y: x + y
    var_72 = [var_59, var_63]
    var_73 = module_0.map_structure_zip(var_71, var_72)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = 'a'
    var_7 = module_0.map_structure(var_5, var_6)
    assert var_7 == 'aa'
    var_8 = lambda x: x + var_0
    var_9 = 3
    var_10 = [var_0, var_4, var_9]
    var_11 = module_0.map_structure(var_8, var_10)
    var_12 = lambda x: x + var_0
    var_13 = [var_0]
    var_14 = [var_9]
    var_15 = [var_4, var_14]
    var_16 = [var_13, var_15]
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_4
    var_19 = (var_0, var_4)
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = lambda x: x + var_0
    var_22 = (var_4, var_9)
    var_23 = (var_0, var_22)
    var_24 = module_0.map_structure(var_21, var_23)
    var_25 = 'Point'
    var_26 = 'x'
    var_27 = 'y'
    var_28 = [var_26, var_27]
    var_29 = 10
    var_30 = lambda x: x + var_29
    var_31 = 11
    var_32 = 12
    var_33 = 'b'
    var_34 = 'c'
    var_35 = {var_34: var_4}
    var_36 = {var_6: var_0, var_33: var_35}
    var_37 = {var_34: var_9}
    var_38 = {var_6: var_4, var_33: var_37}
    var_39 = lambda x: x + var_0
    var_40 = module_0.map_structure(var_39, var_36)
    var_41 = (var_6, var_0)
    var_42 = (var_33, var_4)
    var_43 = [var_41, var_42]
    var_44 = (var_6, var_4)
    var_45 = (var_33, var_9)
    var_46 = [var_44, var_45]
    var_47 = lambda x: x + var_0
    var_48 = {var_0, var_4, var_9}
    var_49 = lambda x: x + var_0
    var_50 = module_0.map_structure(var_49, var_48)
    var_51 = lambda x: x.val + var_0
    var_52 = [var_0, var_4, var_9]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = lambda x: len(x)
    var_55 = module_0.map_structure(var_54, var_53)
    assert var_55 == 3
    var_56 = {var_0, var_4}
    var_57 = lambda x: len(x)
    var_58 = module_0.map_structure(var_57, var_56)
    assert var_58 == 2
    var_59 = (var_0, var_4)
    var_60 = {var_6: var_59}
    var_61 = 4
    var_62 = 'e'
    var_63 = {var_62: var_2}
    var_64 = [var_61, var_63]
    var_65 = (var_9, var_64)
    var_66 = 6
    var_67 = 7
    var_68 = {var_66, var_67}
    var_69 = [var_60, var_65, var_68]
    var_70 = (var_4, var_9)
    var_71 = {var_6: var_70}
    var_72 = {var_62: var_66}
    var_73 = [var_2, var_72]
    var_74 = (var_61, var_73)
    var_75 = 8
    var_76 = {var_67, var_75}
    var_77 = [var_71, var_74, var_76]
    var_78 = lambda x: x + var_0
    var_79 = module_0.map_structure(var_78, var_69)



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
    var_6 = lambda x: x + var_2
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = 10
    var_20 = lambda x: x * var_19
    var_21 = 20
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_0, var_3]
    var_25 = {var_22: var_2, var_23: var_24}
    var_26 = lambda x: x + var_2
    var_27 = module_0.map_structure(var_26, var_25)
    var_28 = {var_2, var_0, var_3}
    var_29 = lambda x: x * var_0
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = 5
    var_32 = lambda x: x.val * var_0
    var_33 = [var_2, var_0]
    var_34 = lambda x: x.value
    var_35 = 'key'
    var_36 = (var_0, var_3)
    var_37 = {var_35: var_36}
    var_38 = 4
    var_39 = {var_38, var_31}
    var_40 = [var_2, var_37, var_39]
    var_41 = lambda x: x + var_19
    var_42 = module_0.map_structure(var_41, var_40)
    var_43 = 14
    var_44 = 15
    var_45 = [var_43, var_44]
    var_46 = var_42[var_0]
    var_47 = lambda x: x
    var_48 = 42
    var_49 = module_0.map_structure(var_47, var_48)
    assert var_49 == 42



# Parsed testcases at query #3
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.map_structure_zip(var_0, var_4)
    var_10 = [var_1, var_2]
    var_11 = [var_3]
    var_12 = [var_10, var_11]
    var_13 = 10
    var_14 = 20
    var_15 = [var_13, var_14]
    var_16 = 30
    var_17 = [var_16]
    var_18 = [var_15, var_17]
    var_19 = 11
    var_20 = 22
    var_21 = [var_19, var_20]
    var_22 = 33
    var_23 = [var_22]
    var_24 = [var_21, var_23]
    var_25 = lambda x, y: x + y
    var_26 = module_0.map_structure_zip(var_25, var_12)
    var_27 = (var_2, var_3)
    var_28 = (var_1, var_27)
    var_29 = (var_14, var_16)
    var_30 = (var_13, var_29)
    var_31 = (var_20, var_22)
    var_32 = (var_19, var_31)
    var_33 = lambda x, y: x + y
    var_34 = module_0.map_structure_zip(var_33, var_28)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = lambda x, y: x + y
    var_40 = 'a'
    var_41 = 'b'
    var_42 = 'c'
    var_43 = {var_42: var_2}
    var_44 = {var_40: var_1, var_41: var_43}
    var_45 = {var_42: var_14}
    var_46 = {var_40: var_13, var_41: var_45}
    var_47 = {var_42: var_20}
    var_48 = {var_40: var_19, var_41: var_47}
    var_49 = 0
    var_50 = lambda x, y: x + y
    var_51 = [var_44]
    var_52 = [var_46]
    var_53 = map_structure_zip(var_50, var_51, var_52)[var_49]
    var_54 = (var_40, var_1)
    var_55 = (var_41, var_2)
    var_56 = [var_54, var_55]
    var_57 = (var_40, var_13)
    var_58 = (var_41, var_14)
    var_59 = [var_57, var_58]
    var_60 = (var_40, var_19)
    var_61 = (var_41, var_20)
    var_62 = [var_60, var_61]
    var_63 = lambda x, y: x + y
    var_64 = [var_1, var_2]
    var_65 = module_0.no_map_instance(var_64)
    var_66 = [var_13, var_14]
    var_67 = lambda x, y: x + y
    var_68 = [var_65]
    var_69 = [var_66]
    var_70 = module_0.map_structure_zip(var_67, var_68)
    var_71 = {var_1, var_2}
    var_72 = {var_13, var_14}
    var_73 = lambda x, y: x + y
    var_74 = [var_71]
    var_75 = [var_72]
    var_76 = module_0.map_structure_zip(var_73, var_74)
    var_77 = (var_75, var_76)
    var_78 = {var_40: var_77}
    var_79 = [var_5]
    var_80 = [var_74, var_78, var_79]
    var_81 = (var_14, var_16)
    var_82 = {var_40: var_81}
    var_83 = 40
    var_84 = [var_83]
    var_85 = [var_13, var_82, var_84]
    var_86 = (var_20, var_22)
    var_87 = {var_40: var_86}
    var_88 = 44
    var_89 = [var_88]
    var_90 = [var_19, var_87, var_89]
    var_91 = lambda x, y: x + y
    var_92 = module_0.map_structure_zip(var_91, var_80)



# Parsed testcases at query #4
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
    var_9 = lambda x, y: x * y
    var_10 = [var_1]
    var_11 = [var_2]
    var_12 = [var_10, var_11]
    var_13 = [var_4]
    var_14 = [var_5]
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = module_0.map_structure_zip(var_9, var_16)
    var_18 = lambda x, y: x - y
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 10
    var_22 = 20
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = {var_19: var_1, var_20: var_2}
    var_25 = [var_23, var_24]
    var_26 = module_0.map_structure_zip(var_18, var_25)
    var_27 = lambda x, y: x / y
    var_28 = (var_21, var_22)
    var_29 = 5
    var_30 = (var_29, var_2)
    var_31 = [var_28, var_30]
    var_32 = module_0.map_structure_zip(var_27, var_31)
    var_33 = 'Point'
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = lambda x, y: x + y
    var_38 = module_0.map_structure_zip(var_37, var_31)
    var_39 = 6
    var_40 = lambda x, y: f'{x}{y}'
    var_41 = (var_20,)
    var_42 = [var_19, var_41]
    var_43 = '1'
    var_44 = '2'
    var_45 = (var_44,)
    var_46 = [var_43, var_45]
    var_47 = [var_42, var_46]
    var_48 = module_0.map_structure_zip(var_40, var_47)
    var_49 = [var_1, var_2]
    var_50 = [var_4, var_5]
    var_51 = lambda x, y: len(x) + len(y)
    var_52 = lambda x, y: len(x) + len(y)
    var_53 = [var_1]
    var_54 = [var_2, var_4]
    var_55 = module_0.map_structure_zip(var_52, var_47)
    assert var_55 == 3
    var_56 = 1
    var_57 = {var_56}
    var_58 = 2
    var_59 = {var_58}
    var_60 = [var_57, var_59]
    var_61 = module_0.map_structure_zip(var_52, var_60)
    var_62 = lambda x: x * var_57
    var_63 = [var_56, var_57, var_59]
    var_64 = [var_63]
    var_65 = module_0.map_structure_zip(var_62, var_64)



# Parsed testcases at query #5
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 2
    var_3 = 2
    var_4 = lambda x: x * var_3
    var_5 = 3
    var_6 = [var_0, var_3, var_5]
    var_7 = module_0.map_structure(var_4, var_6)
    var_8 = lambda x: x + var_0
    var_9 = [var_0]
    var_10 = [var_5]
    var_11 = [var_3, var_10]
    var_12 = [var_9, var_11]
    var_13 = module_0.map_structure(var_8, var_12)
    var_14 = lambda x: x * var_3
    var_15 = (var_0, var_3)
    var_16 = module_0.map_structure(var_14, var_15)
    var_17 = 'Point'
    var_18 = 'x'
    var_19 = 'y'
    var_20 = [var_18, var_19]
    var_21 = 10
    var_22 = lambda x: x + var_21
    var_23 = 11
    var_24 = 12
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = {var_27: var_3}
    var_29 = {var_25: var_0, var_26: var_28}
    var_30 = lambda x: x * var_5
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = {var_0, var_3, var_5}
    var_33 = lambda x: x + var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = [var_0, var_3, var_5]
    var_36 = [var_0, var_3]
    var_37 = module_0.no_map_instance(var_36)
    var_38 = lambda x: len(x)
    var_39 = module_0.map_structure(var_38, var_37)
    assert var_39 == 2
    var_40 = 5
    var_41 = lambda x: x * var_3
    var_42 = {var_25: var_3}
    var_43 = (var_0, var_42)
    var_44 = 4
    var_45 = {var_5, var_44}
    var_46 = [var_43, var_45]
    var_47 = {var_25: var_44}
    var_48 = (var_3, var_47)
    var_49 = {var_44, var_40}
    var_50 = [var_48, var_49]
    var_51 = lambda x: x + var_0
    var_52 = module_0.map_structure(var_51, var_46)



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
    var_6 = lambda x: x + var_2
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = 10
    var_20 = lambda x: x * var_19
    var_21 = 20
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_0, var_3]
    var_25 = {var_22: var_2, var_23: var_24}
    var_26 = 5
    var_27 = lambda x: x + var_26
    var_28 = module_0.map_structure(var_27, var_25)
    var_29 = (var_22, var_2)
    var_30 = (var_23, var_0)
    var_31 = [var_29, var_30]
    var_32 = lambda x: x * var_3
    var_33 = (var_22, var_3)
    var_34 = 6
    var_35 = (var_23, var_34)
    var_36 = [var_33, var_35]
    var_37 = {var_2, var_0, var_3}
    var_38 = lambda x: x + var_2
    var_39 = module_0.map_structure(var_38, var_37)
    var_40 = [var_2, var_0]
    var_41 = 0
    var_42 = lambda x: x.val[var_41]
    var_43 = lambda x: len(x)
    var_44 = [var_2, var_0]
    var_45 = module_0.map_structure(var_43, var_44)
    assert var_45 == 2
    var_46 = lambda x: x + var_2
    var_47 = module_0.map_structure(var_46, var_19)
    assert var_47 == 11
    var_48 = 'list'
    var_49 = 'tuple'
    var_50 = (var_0, var_3)
    var_51 = [var_2, var_50]
    var_52 = 4
    var_53 = {var_22: var_26}
    var_54 = (var_52, var_53)
    var_55 = {var_48: var_51, var_49: var_54}
    var_56 = (var_52, var_34)
    var_57 = [var_0, var_56]
    var_58 = 8
    var_59 = {var_22: var_19}
    var_60 = (var_58, var_59)
    var_61 = {var_48: var_57, var_49: var_60}
    var_62 = lambda x: x * var_0
    var_63 = module_0.map_structure(var_62, var_55)



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
    var_6 = [var_2, var_3]
    var_7 = 4
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = [var_0, var_6, var_9]
    var_11 = lambda x: x * var_2
    var_12 = module_0.map_structure(var_11, var_10)
    var_13 = (var_3, var_7)
    var_14 = (var_0, var_2, var_13)
    var_15 = lambda x: x ** var_2
    var_16 = module_0.map_structure(var_15, var_14)
    var_17 = 'Point'
    var_18 = 'x'
    var_19 = 'y'
    var_20 = [var_18, var_19]
    var_21 = 10
    var_22 = lambda x: x + var_21
    var_23 = 11
    var_24 = 12
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'c'
    var_28 = {var_27: var_2}
    var_29 = {var_25: var_0, var_26: var_28}
    var_30 = {var_27: var_3}
    var_31 = {var_25: var_2, var_26: var_30}
    var_32 = lambda x: x + var_0
    var_33 = module_0.map_structure(var_32, var_29)
    var_34 = {var_0, var_2, var_3}
    var_35 = lambda x: x * var_21
    var_36 = module_0.map_structure(var_35, var_34)
    var_37 = [var_0, var_2, var_3]
    var_38 = lambda x: len(x)
    var_39 = 5
    var_40 = lambda x: x.value + var_0
    var_41 = 'key'
    var_42 = (var_2, var_3)
    var_43 = {var_41: var_42}
    var_44 = {var_7, var_39}
    var_45 = [var_0, var_43, var_44]
    var_46 = (var_3, var_7)
    var_47 = {var_41: var_46}
    var_48 = 6
    var_49 = {var_39, var_48}
    var_50 = [var_2, var_47, var_49]
    var_51 = lambda x: x + var_0
    var_52 = module_0.map_structure(var_51, var_45)
    var_53 = lambda x: x + var_0
    var_54 = module_0.map_structure(var_53, var_21)
    assert var_54 == 11



# Parsed testcases at query #8
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 2
    var_3 = 2
    var_4 = lambda x: x * var_3
    var_5 = 3
    var_6 = [var_0, var_3, var_5]
    var_7 = module_0.map_structure(var_4, var_6)
    var_8 = lambda x: x + var_0
    var_9 = [var_0]
    var_10 = [var_5]
    var_11 = [var_3, var_10]
    var_12 = [var_9, var_11]
    var_13 = module_0.map_structure(var_8, var_12)
    var_14 = lambda x: x * var_3
    var_15 = (var_0, var_3)
    var_16 = module_0.map_structure(var_14, var_15)
    var_17 = 'Point'
    var_18 = 'x'
    var_19 = 'y'
    var_20 = [var_18, var_19]
    var_21 = 10
    var_22 = lambda x: x + var_21
    var_23 = 11
    var_24 = 12
    var_25 = 'a'
    var_26 = 'b'
    var_27 = [var_3, var_5]
    var_28 = {var_25: var_0, var_26: var_27}
    var_29 = 4
    var_30 = 6
    var_31 = [var_29, var_30]
    var_32 = {var_25: var_3, var_26: var_31}
    var_33 = lambda x: x * var_3
    var_34 = module_0.map_structure(var_33, var_28)
    var_35 = {var_0, var_3, var_5}
    var_36 = lambda x: x + var_0
    var_37 = module_0.map_structure(var_36, var_35)
    var_38 = [var_0, var_3]
    var_39 = lambda x: len(x)
    var_40 = lambda x: x.val * var_3
    var_41 = (var_0, var_3)
    var_42 = {var_25: var_41}
    var_43 = {var_5, var_29}
    var_44 = 5
    var_45 = {var_26: var_30}
    var_46 = [var_44, var_45]
    var_47 = [var_42, var_43, var_46]
    var_48 = (var_3, var_5)
    var_49 = {var_25: var_48}
    var_50 = {var_29, var_44}
    var_51 = 7
    var_52 = {var_26: var_51}
    var_53 = [var_30, var_52]
    var_54 = [var_49, var_50, var_53]
    var_55 = lambda x: x + var_0
    var_56 = module_0.map_structure(var_55, var_47)



# Parsed testcases at query #9
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6
    var_4 = lambda x: x.upper()
    var_5 = 'hello'
    var_6 = module_0.map_structure(var_4, var_5)
    assert var_6 == 'HELLO'
    var_7 = 2
    var_8 = lambda x: x * var_7
    var_9 = 3
    var_10 = [var_0, var_7, var_9]
    var_11 = module_0.map_structure(var_8, var_10)
    var_12 = lambda x: x + var_0
    var_13 = [var_0]
    var_14 = [var_9]
    var_15 = [var_7, var_14]
    var_16 = [var_13, var_15]
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_7
    var_19 = (var_0, var_7)
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = 10
    var_26 = lambda x: x + var_25
    var_27 = 11
    var_28 = 12
    var_29 = 'a'
    var_30 = 'b'
    var_31 = [var_7, var_9]
    var_32 = {var_29: var_0, var_30: var_31}
    var_33 = 4
    var_34 = 6
    var_35 = [var_33, var_34]
    var_36 = {var_29: var_7, var_30: var_35}
    var_37 = lambda x: x * var_7
    var_38 = module_0.map_structure(var_37, var_32)
    var_39 = (var_29, var_0)
    var_40 = (var_30, var_7)
    var_41 = [var_39, var_40]
    var_42 = (var_29, var_7)
    var_43 = (var_30, var_33)
    var_44 = [var_42, var_43]
    var_45 = lambda x: x * var_7
    var_46 = {var_0, var_7, var_9}
    var_47 = lambda x: x + var_0
    var_48 = module_0.map_structure(var_47, var_46)
    var_49 = 'not_traversed'
    var_50 = lambda x: var_49
    var_51 = {var_0, var_7}
    var_52 = module_0.map_structure(var_50, var_51)
    var_53 = lambda x: len(x)
    var_54 = {var_0, var_7}
    var_55 = module_0.map_structure(var_53, var_54)
    assert var_55 == 2
    var_56 = [var_0, var_7, var_9]
    var_57 = 'skip'
    var_58 = lambda x: var_57
    var_59 = (var_0, var_7)
    var_60 = [var_9, var_33]
    var_61 = {var_29: var_59, var_30: var_60}
    var_62 = 'c'
    var_63 = {var_62: var_34}
    var_64 = (var_2, var_63)
    var_65 = [var_61, var_64]
    var_66 = (var_7, var_33)
    var_67 = 8
    var_68 = [var_34, var_67]
    var_69 = {var_29: var_66, var_30: var_68}
    var_70 = {var_62: var_28}
    var_71 = (var_25, var_70)
    var_72 = [var_69, var_71]
    var_73 = lambda x: x * var_7
    var_74 = module_0.map_structure(var_73, var_65)
    var_75 = (var_0, var_7)
    var_76 = module_0.no_map_instance(var_75)
    var_77 = 'found'
    var_78 = lambda x: var_77
    var_79 = module_0.map_structure(var_78, var_76)



# Parsed testcases at query #10
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
    var_13 = 10
    var_14 = 20
    var_15 = [var_13, var_14]
    var_16 = 30
    var_17 = [var_16]
    var_18 = [var_15, var_17]
    var_19 = [var_12, var_18]
    var_20 = module_0.map_structure_zip(var_9, var_19)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = [var_2, var_4]
    var_24 = {var_21: var_1, var_22: var_23}
    var_25 = [var_14, var_16]
    var_26 = {var_21: var_13, var_22: var_25}
    var_27 = [var_24, var_26]
    var_28 = module_0.map_structure_zip(var_9, var_27)
    var_29 = (var_2, var_4)
    var_30 = (var_1, var_29)
    var_31 = (var_14, var_16)
    var_32 = (var_13, var_31)
    var_33 = [var_30, var_32]
    var_34 = module_0.map_structure_zip(var_9, var_33)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = 11
    var_40 = 22
    var_41 = lambda x, y: x.val + y.val
    var_42 = 1
    var_43 = {var_42}
    var_44 = 2
    var_45 = {var_44}
    var_46 = [var_43, var_45]
    var_47 = module_0.map_structure_zip(var_9, var_46)
    var_48 = lambda x, y: str(x) + str(y)
    var_49 = [var_43, var_44]
    var_50 = [var_21, var_22]
    var_51 = [var_49, var_50]
    var_52 = module_0.map_structure_zip(var_48, var_51)



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
    var_6 = lambda x: x + var_2
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 10
    var_16 = lambda x: x * var_15
    var_17 = 'a'
    var_18 = 'b'
    var_19 = [var_0, var_3]
    var_20 = {var_17: var_2, var_18: var_19}
    var_21 = module_0.map_structure(var_16, var_20)
    var_22 = lambda x: x + var_2
    var_23 = {var_2, var_0}
    var_24 = module_0.map_structure(var_22, var_23)
    var_25 = 'Point'
    var_26 = 'x'
    var_27 = 'y'
    var_28 = [var_26, var_27]
    var_29 = 5
    var_30 = lambda x: x * var_29
    var_31 = 'list'
    var_32 = 'tuple'
    var_33 = 'val'
    var_34 = (var_0, var_3)
    var_35 = [var_2, var_34]
    var_36 = 4
    var_37 = 'inner'
    var_38 = {var_37: var_29}
    var_39 = (var_36, var_38)
    var_40 = 6
    var_41 = {var_31: var_35, var_32: var_39, var_33: var_40}
    var_42 = (var_36, var_40)
    var_43 = [var_0, var_42]
    var_44 = 8
    var_45 = {var_37: var_15}
    var_46 = (var_44, var_45)
    var_47 = 12
    var_48 = {var_31: var_43, var_32: var_46, var_33: var_47}
    var_49 = lambda x: x * var_0
    var_50 = module_0.map_structure(var_49, var_41)
    var_51 = lambda x: len(x)
    var_52 = {var_2, var_0, var_3}
    var_53 = module_0.map_structure(var_51, var_52)
    assert var_53 == 3
    var_54 = [var_2, var_0]
    var_55 = lambda x: x + var_2
    var_56 = lambda x: x.val + var_29
    var_57 = lambda x: x
    var_58 = 42
    var_59 = module_0.map_structure(var_57, var_58)
    assert var_59 == 42



# Parsed testcases at query #12
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
    var_8 = 6
    var_9 = [var_5, var_8]
    var_10 = [var_9]
    var_11 = module_0.map_structure_zip(var_0, var_7)
    var_12 = lambda x, y: x * y
    var_13 = [var_1, var_2]
    var_14 = [var_4, var_5]
    var_15 = (var_13, var_14)
    var_16 = 5
    var_17 = [var_16, var_8]
    var_18 = 7
    var_19 = 8
    var_20 = [var_18, var_19]
    var_21 = (var_17, var_20)
    var_22 = [var_15, var_21]
    var_23 = 14
    var_24 = (var_16, var_23)
    var_25 = 21
    var_26 = 48
    var_27 = (var_25, var_26)
    var_28 = (var_24, var_27)
    var_29 = [var_28]
    var_30 = module_0.map_structure_zip(var_12, var_22)
    var_31 = lambda x, y: x + y
    var_32 = 'a'
    var_33 = 'b'
    var_34 = {var_32: var_1, var_33: var_2}
    var_35 = 10
    var_36 = 20
    var_37 = {var_32: var_35, var_33: var_36}
    var_38 = [var_34, var_37]
    var_39 = 11
    var_40 = 22
    var_41 = {var_32: var_39, var_33: var_40}
    var_42 = module_0.map_structure_zip(var_31, var_38)
    var_43 = 'Point'
    var_44 = 'x'
    var_45 = 'y'
    var_46 = [var_44, var_45]
    var_47 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_48 = lambda c1, c2: c1.val + c2.val
    var_49 = {var_1, var_2}
    var_50 = {var_4, var_5}
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_52]
    var_55 = (var_53,)
    var_56 = [var_54, var_55]
    var_57 = [var_52]
    var_58 = [var_53]
    var_59 = [var_57, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = [var_52, var_53, var_4]
    var_63 = lambda x, y, z: x + y + z
    var_64 = module_0.map_structure_zip(var_63, var_62)
    assert var_64 == 6



# Parsed testcases at query #13
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3
    var_5 = lambda x, y, z: x + y + z
    var_6 = 3
    var_7 = [var_1, var_2, var_6]
    var_8 = module_0.map_structure_zip(var_5, var_7)
    assert var_8 == 6
    var_9 = [var_1, var_2]
    var_10 = 4
    var_11 = [var_6, var_10]
    var_12 = [var_9, var_11]
    var_13 = lambda x, y: x + y
    var_14 = module_0.map_structure_zip(var_13, var_12)
    var_15 = [var_1]
    var_16 = [var_2]
    var_17 = [var_15, var_16]
    var_18 = [var_6]
    var_19 = [var_10]
    var_20 = [var_18, var_19]
    var_21 = [var_17, var_20]
    var_22 = lambda x, y: x + y
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = (var_1, var_2)
    var_25 = (var_6, var_10)
    var_26 = [var_24, var_25]
    var_27 = lambda x, y: x * y
    var_28 = module_0.map_structure_zip(var_27, var_26)
    var_29 = 'Point'
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_34 = 6
    var_35 = 'a'
    var_36 = 'b'
    var_37 = {var_35: var_1, var_36: var_2}
    var_38 = 10
    var_39 = 20
    var_40 = {var_35: var_38, var_36: var_39}
    var_41 = [var_37, var_40]
    var_42 = 11
    var_43 = 22
    var_44 = {var_35: var_42, var_36: var_43}
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_41)
    var_47 = 'hit'
    var_48 = lambda x, y: var_47
    var_49 = lambda x, y: x + y
    var_50 = 1
    var_51 = {var_50}
    var_52 = 2
    var_53 = {var_52}
    var_54 = [var_51, var_53]
    var_55 = module_0.map_structure_zip(var_49, var_54)



# Parsed testcases at query #14
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
    var_7 = [var_2]
    var_8 = [var_3]
    var_9 = [var_0, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: str(x)
    var_13 = (var_2, var_0)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = 10
    var_16 = lambda x: x * var_15
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_2, var_18: var_0}
    var_20 = module_0.map_structure(var_16, var_19)
    var_21 = {var_17: var_2, var_18: var_0}
    var_22 = 20
    var_23 = {var_17: var_15, var_18: var_22}
    var_24 = lambda x: x * var_15
    var_25 = module_0.map_structure(var_24, var_21)
    var_26 = {var_2, var_0, var_3}
    var_27 = 4
    var_28 = 6
    var_29 = {var_0, var_27, var_28}
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_26)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]
    var_36 = 5
    var_37 = lambda x: x + var_36
    var_38 = 7
    var_39 = (var_0, var_3)
    var_40 = {var_17: var_39}
    var_41 = {var_27, var_36}
    var_42 = [var_2, var_40, var_41]
    var_43 = lambda x: x * var_0
    var_44 = module_0.map_structure(var_43, var_42)
    var_45 = lambda x: x.val + var_36
    var_46 = [var_2, var_0]
    var_47 = lambda x: len(x.data)
    var_48 = lambda x: x + var_2
    var_49 = module_0.map_structure(var_48, var_36)
    assert var_49 == 6



# Parsed testcases at query #15
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = module_0.map_structure(var_0, var_1)
    assert var_2 == 1
    var_3 = lambda x: x
    var_4 = 'abc'
    var_5 = module_0.map_structure(var_3, var_4)
    assert var_5 == 'abc'
    var_6 = lambda x: x + var_1
    var_7 = 2
    var_8 = 3
    var_9 = [var_1, var_7, var_8]
    var_10 = module_0.map_structure(var_6, var_9)
    var_11 = lambda x: x * var_7
    var_12 = [var_1]
    var_13 = [var_7, var_8]
    var_14 = [var_12, var_13]
    var_15 = module_0.map_structure(var_11, var_14)
    var_16 = lambda x: x + var_1
    var_17 = (var_1, var_7, var_8)
    var_18 = module_0.map_structure(var_16, var_17)
    var_19 = 'Point'
    var_20 = 'x'
    var_21 = 'y'
    var_22 = [var_20, var_21]
    var_23 = 10
    var_24 = lambda x: x + var_23
    var_25 = 11
    var_26 = 12
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = [var_7, var_8]
    var_31 = 'd'
    var_32 = 4
    var_33 = {var_31: var_32}
    var_34 = {var_27: var_1, var_28: var_30, var_29: var_33}
    var_35 = [var_8, var_32]
    var_36 = 5
    var_37 = {var_31: var_36}
    var_38 = {var_27: var_7, var_28: var_35, var_29: var_37}
    var_39 = lambda x: x + var_1
    var_40 = module_0.map_structure(var_39, var_34)
    var_41 = {var_1, var_7, var_8}
    var_42 = {var_7, var_8, var_32}
    var_43 = lambda x: x + var_1
    var_44 = module_0.map_structure(var_43, var_41)
    var_45 = [var_1, var_7]
    var_46 = lambda x: x.value
    var_47 = lambda x: x.val + var_36
    var_48 = lambda x: len(x)
    var_49 = [var_1, var_7]
    var_50 = module_0.map_structure(var_48, var_49)
    assert var_50 == 2

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = [var_3]
    var_5 = 3
    var_6 = [var_5]
    var_7 = [var_2, var_4, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = [var_1, var_3]
    var_10 = 10
    var_11 = 20
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = lambda x, y: x + y
    var_15 = module_0.map_structure_zip(var_14, var_13)
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_1, var_17: var_3}
    var_19 = {var_16: var_10, var_17: var_11}
    var_20 = [var_18, var_19]
    var_21 = lambda x, y: x + y
    var_22 = module_0.map_structure_zip(var_21, var_20)
    var_23 = lambda x: x
    var_24 = 1
    var_25 = {var_24}
    var_26 = 2
    var_27 = {var_26}
    var_28 = [var_25, var_27]
    var_29 = module_0.map_structure_zip(var_23, var_28)



# Parsed testcases at query #16
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.map_structure_zip(var_0, var_4)
    var_10 = [var_1, var_2]
    var_11 = [var_3]
    var_12 = [var_10, var_11]
    var_13 = 10
    var_14 = 20
    var_15 = [var_13, var_14]
    var_16 = 30
    var_17 = [var_16]
    var_18 = [var_15, var_17]
    var_19 = lambda x, y: x + y
    var_20 = module_0.map_structure_zip(var_19, var_12)
    var_21 = (var_2, var_3)
    var_22 = (var_1, var_21)
    var_23 = (var_14, var_16)
    var_24 = (var_13, var_23)
    var_25 = lambda x, y: x + y
    var_26 = [var_22]
    var_27 = [var_24]
    var_28 = module_0.map_structure_zip(var_25, var_26)
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_13, var_30: var_14}
    var_33 = lambda x, y: x + y
    var_34 = [var_31]
    var_35 = [var_32]
    var_36 = module_0.map_structure_zip(var_33, var_34)
    var_37 = 'Point'
    var_38 = 'x'
    var_39 = 'y'
    var_40 = [var_38, var_39]
    var_41 = 100
    var_42 = 200
    var_43 = lambda x, y, z: x + y + z
    var_44 = 111
    var_45 = 222
    var_46 = lambda x, y: x.val + y.val
    var_47 = {var_1, var_2}
    var_48 = {var_13, var_14}
    var_49 = lambda x, y: list(x) + list(y)
    var_50 = [var_47]
    var_51 = [var_48]
    var_52 = module_0.map_structure_zip(var_49, var_50)
    var_53 = lambda x, y: x + y
    var_54 = 1
    var_55 = {var_54}
    var_56 = [var_55]
    var_57 = 2
    var_58 = {var_57}
    var_59 = [var_58]
    var_60 = module_0.map_structure_zip(var_53, var_56)
    var_61 = {var_29: var_55}
    var_62 = (var_56,)
    var_63 = [var_54, var_61, var_62]
    var_64 = {var_29: var_14}
    var_65 = (var_16,)
    var_66 = [var_13, var_64, var_65]
    var_67 = lambda x, y: x + y
    var_68 = [var_63]
    var_69 = [var_66]
    var_70 = module_0.map_structure_zip(var_67, var_68)



# Parsed testcases at query #17
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 3
    var_5 = lambda x, y: x * y
    var_6 = 10
    var_7 = 20
    var_8 = [var_6, var_7]
    var_9 = module_0.map_structure_zip(var_5, var_8)
    assert var_9 == 200
    var_10 = [var_1, var_2]
    var_11 = 3
    var_12 = 4
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_6, var_7]
    var_16 = 30
    var_17 = 40
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = 11
    var_21 = 22
    var_22 = [var_20, var_21]
    var_23 = 33
    var_24 = 44
    var_25 = [var_23, var_24]
    var_26 = [var_22, var_25]
    var_27 = lambda x, y: x + y
    var_28 = [var_14, var_19]
    var_29 = module_0.map_structure_zip(var_27, var_28)
    var_30 = (var_2, var_11)
    var_31 = (var_1, var_30)
    var_32 = (var_7, var_16)
    var_33 = (var_6, var_32)
    var_34 = (var_21, var_23)
    var_35 = (var_20, var_34)
    var_36 = lambda x: x
    var_37 = module_0.map_structure(var_36, var_31)
    var_38 = lambda x, y: x + y
    var_39 = [var_31, var_33]
    var_40 = module_0.map_structure_zip(var_38, var_39)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = [var_2, var_11]
    var_44 = {var_41: var_1, var_42: var_43}
    var_45 = [var_7, var_16]
    var_46 = {var_41: var_6, var_42: var_45}
    var_47 = [var_21, var_23]
    var_48 = {var_41: var_20, var_42: var_47}
    var_49 = lambda x, y: x + y
    var_50 = [var_44, var_46]
    var_51 = module_0.map_structure_zip(var_49, var_50)
    var_52 = 'Point'
    var_53 = 'x'
    var_54 = 'y'
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = [var_1, var_2]
    var_58 = [var_6, var_7]
    var_59 = lambda x, y: len(x) + len(y)
    var_60 = lambda x, y: len(x) + len(y)
    var_61 = {var_1, var_2}
    var_62 = {var_6, var_7}
    var_63 = lambda x, y: x + y
    var_64 = [var_61, var_62]
    var_65 = module_0.map_structure_zip(var_63, var_64)
    var_66 = (var_64, var_65)
    var_67 = {var_41: var_66}
    var_68 = (var_11, var_12)
    var_69 = {var_42: var_68}
    var_70 = [var_67, var_69]
    var_71 = (var_6, var_7)
    var_72 = {var_41: var_71}
    var_73 = (var_16, var_17)
    var_74 = {var_42: var_73}
    var_75 = [var_72, var_74]
    var_76 = (var_20, var_21)
    var_77 = {var_41: var_76}
    var_78 = (var_23, var_24)
    var_79 = {var_42: var_78}
    var_80 = [var_77, var_79]
    var_81 = lambda x, y: x + y
    var_82 = [var_70, var_75]
    var_83 = module_0.map_structure_zip(var_81, var_82)



# Parsed testcases at query #18
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = [var_1]
    var_12 = [var_2]
    var_13 = [var_11, var_12]
    var_14 = 10
    var_15 = [var_14]
    var_16 = 20
    var_17 = [var_16]
    var_18 = [var_15, var_17]
    var_19 = [var_13, var_18]
    var_20 = module_0.map_structure_zip(var_0, var_19)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_1, var_22: var_2}
    var_24 = {var_21: var_14, var_22: var_16}
    var_25 = [var_23, var_24]
    var_26 = module_0.map_structure_zip(var_0, var_25)
    var_27 = (var_1, var_2)
    var_28 = (var_3, var_5)
    var_29 = [var_27, var_28]
    var_30 = module_0.map_structure_zip(var_0, var_29)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = [var_2, var_3]
    var_36 = [var_1, var_35]
    var_37 = 30
    var_38 = [var_16, var_37]
    var_39 = [var_14, var_38]
    var_40 = [var_36, var_39]
    var_41 = module_0.map_structure_zip(var_0, var_40)
    var_42 = [var_1, var_2, var_3]
    var_43 = module_0.map_structure_zip(var_0, var_42)
    assert var_43 == 6
    var_44 = [var_1, var_2]
    var_45 = [var_3, var_5]
    var_46 = lambda x, y: len(x) + len(y)
    var_47 = {var_1, var_2}
    var_48 = {var_3, var_5}
    var_49 = [var_47, var_48]
    var_50 = module_0.map_structure_zip(var_0, var_49)
    var_51 = [var_49, var_50]
    var_52 = (var_3,)
    var_53 = {var_21: var_51, var_22: var_52}
    var_54 = [var_14, var_16]
    var_55 = (var_37,)
    var_56 = {var_21: var_54, var_22: var_55}
    var_57 = 11
    var_58 = 22
    var_59 = [var_57, var_58]
    var_60 = 33
    var_61 = (var_60,)
    var_62 = {var_21: var_59, var_22: var_61}
    var_63 = [var_53, var_56]
    var_64 = module_0.map_structure_zip(var_0, var_63)



