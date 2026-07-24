####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_21 = lambda x: x + var_0
    var_22 = 10
    var_23 = lambda x: x * var_22
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = [var_0, var_3]
    var_29 = 'c'
    var_30 = {var_29: var_5}
    var_31 = {var_24: var_28, var_25: var_30}
    var_32 = 11
    var_33 = 12
    var_34 = [var_32, var_33]
    var_35 = 13
    var_36 = {var_29: var_35}
    var_37 = {var_24: var_34, var_25: var_36}
    var_38 = lambda x: x + var_22
    var_39 = module_0.map_structure(var_38, var_31)
    var_40 = {var_0, var_3, var_5}
    var_41 = lambda x: x * var_3
    var_42 = module_0.map_structure(var_41, var_40)
    var_43 = {var_0, var_3}
    var_44 = lambda x: len(x)
    var_45 = module_0.map_structure(var_44, var_43)
    assert var_45 == 2
    var_46 = [var_0, var_3, var_5]
    var_47 = lambda x: len(x.val)
    var_48 = (var_0, var_3)
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x: len(x)
    var_51 = module_0.map_structure(var_50, var_49)
    assert var_51 == 2



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
    var_9 = [var_1]
    var_10 = [var_2]
    var_11 = [var_9, var_10]
    var_12 = [var_4]
    var_13 = [var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = lambda x, y: x * y
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_1, var_19: var_2}
    var_21 = {var_18: var_4, var_19: var_5}
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure_zip(var_17, var_22)
    var_24 = (var_1, var_2)
    var_25 = (var_4, var_5)
    var_26 = [var_24, var_25]
    var_27 = module_0.map_structure_zip(var_0, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = [var_1, var_2]
    var_33 = {var_18: var_32}
    var_34 = [var_4, var_5]
    var_35 = {var_18: var_34}
    var_36 = [var_33, var_35]
    var_37 = module_0.map_structure_zip(var_0, var_36)
    var_38 = [var_1, var_2, var_4]
    var_39 = module_0.map_structure_zip(var_0, var_38)
    assert var_39 == 6
    var_40 = [var_1, var_2]
    var_41 = [var_1, var_2]
    var_42 = module_0.no_map_instance(var_41)
    var_43 = [var_4, var_5]
    var_44 = [var_42, var_43]
    var_45 = module_0.map_structure_zip(var_0, var_44)
    var_46 = module_0.map_structure_zip(var_0, var_44)
    var_47 = 1
    var_48 = {var_47}
    var_49 = 2
    var_50 = {var_49}
    var_51 = [var_48, var_50]
    var_52 = module_0.map_structure_zip(var_0, var_51)
    var_53 = [var_47, var_48]
    var_54 = [var_50, var_51]



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
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = module_0.map_structure_zip(var_7, var_6)
    var_9 = [var_0]
    var_10 = [var_1]
    var_11 = [var_9, var_10]
    var_12 = [var_3]
    var_13 = [var_4]
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = 0
    var_17 = lambda x, y: x[var_16] + y[var_16]
    var_18 = module_0.map_structure_zip(var_17, var_15)
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_0, var_20: var_1}
    var_22 = 10
    var_23 = 20
    var_24 = {var_19: var_22, var_20: var_23}
    var_25 = [var_21, var_24]
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
    var_37 = 'val'
    var_38 = {var_37: var_0}
    var_39 = {var_37: var_22}
    var_40 = [var_38, var_39]
    var_41 = lambda d1, d2: {var_37: d1[var_37] + d2[var_37]}
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = lambda a, b: a.val + b.val
    var_44 = {var_0, var_1}
    var_45 = {var_3, var_4}
    var_46 = [var_44, var_45]
    var_47 = lambda s1, s2: len(s1) + len(s2)
    var_48 = module_0.map_structure_zip(var_47, var_46)
    assert var_48 == 4
    var_49 = lambda x: x
    var_50 = 1
    var_51 = {var_50}
    var_52 = 2
    var_53 = {var_52}
    var_54 = [var_51, var_53]
    var_55 = module_0.map_structure_zip(var_49, var_54)
    var_56 = [var_49, var_50, var_52]
    var_57 = [var_56]
    var_58 = lambda x: x * var_50
    var_59 = [var_49, var_50, var_52]
    var_60 = [var_59]
    var_61 = module_0.map_structure_zip(var_58, var_60)



# Parsed testcases at query #4
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #5
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 6



# Parsed testcases at query #6
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
    var_25 = lambda x: x * var_5
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_0, var_27: var_3}
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = [var_0, var_3]
    var_31 = 'c'
    var_32 = {var_31: var_5}
    var_33 = {var_26: var_30, var_27: var_32}
    var_34 = 4
    var_35 = [var_3, var_34]
    var_36 = 6
    var_37 = {var_31: var_36}
    var_38 = {var_26: var_35, var_27: var_37}
    var_39 = lambda x: x * var_3
    var_40 = module_0.map_structure(var_39, var_33)
    var_41 = lambda x: x + var_0
    var_42 = {var_0, var_3, var_5}
    var_43 = module_0.map_structure(var_41, var_42)
    var_44 = [var_0, var_3, var_5]
    var_45 = lambda x: len(x)
    var_46 = [var_0, var_3, var_5]
    var_47 = lambda x: x.value
    var_48 = (var_0, var_3)
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x: sum(x)
    var_51 = module_0.map_structure(var_50, var_49)
    assert var_51 == 3



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
    var_6 = [var_2, var_5]
    var_7 = lambda x, y: x + y
    var_8 = [var_0]
    var_9 = [var_1]
    var_10 = [var_8, var_9]
    var_11 = [var_3]
    var_12 = [var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = lambda x, y: x + y
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_0, var_17: var_1}
    var_19 = 10
    var_20 = 20
    var_21 = {var_16: var_19, var_17: var_20}
    var_22 = [var_18, var_21]
    var_23 = lambda x, y: x + y
    var_24 = (var_0, var_1)
    var_25 = (var_3, var_4)
    var_26 = [var_24, var_25]
    var_27 = lambda x, y: x * y
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x + y
    var_33 = 6
    var_34 = [var_0, var_1]
    var_35 = {var_16: var_34}
    var_36 = [var_3, var_4]
    var_37 = {var_16: var_36}
    var_38 = [var_35, var_37]
    var_39 = lambda x, y: x + y
    var_40 = [var_0, var_1]
    var_41 = lambda x, y: x + y
    var_42 = {var_0, var_1}
    var_43 = {var_3, var_4}
    var_44 = [var_42, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_46, var_1]
    var_48 = [var_3, var_4]
    var_49 = lambda x, y: x + y



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
    var_9 = [var_1]
    var_10 = [var_2]
    var_11 = [var_9, var_10]
    var_12 = [var_4]
    var_13 = [var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = lambda x, y: x * y
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_1, var_19: var_2}
    var_21 = {var_18: var_4, var_19: var_5}
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure_zip(var_17, var_22)
    var_24 = (var_1, var_2)
    var_25 = (var_4, var_5)
    var_26 = [var_24, var_25]
    var_27 = module_0.map_structure_zip(var_0, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = [var_1, var_2, var_4]
    var_33 = lambda *args: sum(args)
    var_34 = module_0.map_structure_zip(var_33, var_32)
    assert var_34 == 6
    var_35 = 10
    var_36 = 20
    var_37 = [var_35, var_36]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = [var_1, var_2]
    var_40 = [var_38, var_39]
    var_41 = lambda x, y: x + y
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = (var_1, var_2)
    var_44 = (var_4, var_5)
    var_45 = [var_43, var_44]
    var_46 = 0
    var_47 = lambda x, y: x[var_46] + y[var_46]
    var_48 = module_0.map_structure_zip(var_47, var_45)
    var_49 = {var_1}
    var_50 = {var_2}
    var_51 = [var_49, var_50]
    var_52 = module_0.map_structure_zip(var_0, var_51)
    var_53 = [var_52, var_2]
    var_54 = {var_18: var_53, var_19: var_4}
    var_55 = [var_35, var_36]
    var_56 = 7
    var_57 = {var_18: var_55, var_19: var_56}
    var_58 = [var_54, var_57]
    var_59 = 11
    var_60 = 22
    var_61 = [var_59, var_60]
    var_62 = {var_18: var_61, var_19: var_35}
    var_63 = module_0.map_structure_zip(var_0, var_58)



# Parsed testcases at query #10
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



# Parsed testcases at query #11
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
    var_9 = [var_1]
    var_10 = [var_2]
    var_11 = [var_9, var_10]
    var_12 = [var_4]
    var_13 = [var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = lambda x, y: x * y
    var_18 = 'a'
    var_19 = 'b'
    var_20 = {var_18: var_1, var_19: var_2}
    var_21 = {var_18: var_4, var_19: var_5}
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure_zip(var_17, var_22)
    var_24 = (var_1, var_2)
    var_25 = (var_4, var_5)
    var_26 = [var_24, var_25]
    var_27 = module_0.map_structure_zip(var_0, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = 6
    var_33 = 10
    var_34 = 20
    var_35 = lambda x, y: x.val + y.val
    var_36 = {var_1}
    var_37 = {var_2}
    var_38 = [var_36, var_37]
    var_39 = module_0.map_structure_zip(var_0, var_38)
    var_40 = {var_18: var_39}
    var_41 = (var_2, var_4)
    var_42 = [var_40, var_41]
    var_43 = {var_18: var_33}
    var_44 = 30
    var_45 = (var_34, var_44)
    var_46 = [var_43, var_45]
    var_47 = [var_42, var_46]
    var_48 = 11
    var_49 = {var_18: var_48}
    var_50 = 22
    var_51 = 33
    var_52 = (var_50, var_51)
    var_53 = [var_49, var_52]
    var_54 = module_0.map_structure_zip(var_0, var_47)



# Parsed testcases at query #12
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
    var_12 = 10
    var_13 = 20
    var_14 = [var_12, var_13]
    var_15 = 30
    var_16 = 40
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_11, var_18]
    var_20 = (var_0, var_1)
    var_21 = [var_20]
    var_22 = (var_3, var_4)
    var_23 = [var_22]
    var_24 = [var_21, var_23]
    var_25 = (var_12, var_13)
    var_26 = [var_25]
    var_27 = (var_15, var_16)
    var_28 = [var_27]
    var_29 = [var_26, var_28]
    var_30 = [var_24, var_29]
    var_31 = (var_0, var_1)
    var_32 = (var_3, var_4)
    var_33 = [var_31, var_32]
    var_34 = (var_12, var_13)
    var_35 = (var_15, var_16)
    var_36 = [var_34, var_35]
    var_37 = [var_33, var_36]
    var_38 = 11
    var_39 = 22
    var_40 = (var_38, var_39)
    var_41 = 33
    var_42 = 44
    var_43 = (var_41, var_42)
    var_44 = (var_40, var_43)
    var_45 = [var_44]
    var_46 = lambda x, y: x + y
    var_47 = [var_0, var_1]
    var_48 = [var_12, var_13]
    var_49 = [var_47, var_48]
    var_50 = module_0.map_structure_zip(var_46, var_49)
    var_51 = 'a'
    var_52 = 'b'
    var_53 = {var_51: var_0, var_52: var_1}
    var_54 = {var_51: var_12, var_52: var_13}
    var_55 = [var_53, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = 'Point'
    var_59 = 'x'
    var_60 = 'y'
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = lambda x, y: x.val + y.val
    var_64 = lambda x, y: x + y
    var_65 = 1
    var_66 = {var_65}
    var_67 = 2
    var_68 = {var_67}
    var_69 = [var_66, var_68]
    var_70 = module_0.map_structure_zip(var_64, var_69)
    var_71 = [var_64]
    var_72 = [var_65]
    var_73 = [var_71, var_72]
    var_74 = [var_12]
    var_75 = [var_13]
    var_76 = [var_74, var_75]
    var_77 = [var_73, var_76]
    var_78 = [var_38]
    var_79 = [var_39]
    var_80 = [var_78, var_79]
    var_81 = [var_80]
    var_82 = lambda x, y: x + y
    var_83 = [var_64]
    var_84 = [var_65]
    var_85 = [var_83, var_84]
    var_86 = [var_12]
    var_87 = [var_13]
    var_88 = [var_86, var_87]
    var_89 = [var_85, var_88]
    var_90 = module_0.map_structure_zip(var_82, var_89)



# Parsed testcases at query #13
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
    var_11 = [var_4]
    var_12 = [var_10, var_11]
    var_13 = 5
    var_14 = 6
    var_15 = [var_13, var_14]
    var_16 = 7
    var_17 = [var_16]
    var_18 = [var_15, var_17]
    var_19 = [var_12, var_18]



# Parsed testcases at query #14
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
    var_10 = [var_1]
    var_11 = [var_2]
    var_12 = [var_10, var_11]
    var_13 = [var_4]
    var_14 = [var_5]
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = module_0.map_structure_zip(var_9, var_16)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_36 = [var_1, var_2]
    var_37 = module_0.map_structure_zip(var_0, var_36)
    assert var_37 == 3
    var_38 = [var_1, var_2]
    var_39 = module_0.no_map_instance(var_38)
    var_40 = [var_14, var_15]
    var_41 = lambda x, y: (x, y)
    var_42 = [var_39, var_40]
    var_43 = module_0.map_structure_zip(var_41, var_42)
    var_44 = {var_1, var_2}
    var_45 = {var_3, var_5}
    var_46 = [var_44, var_45]
    var_47 = module_0.map_structure_zip(var_0, var_46)
    var_48 = (var_47, var_3)
    var_49 = {var_22: var_48}
    var_50 = [var_46, var_49, var_5]
    var_51 = (var_15, var_17)
    var_52 = {var_22: var_51}
    var_53 = 40
    var_54 = [var_14, var_52, var_53]
    var_55 = 11
    var_56 = 22
    var_57 = 33
    var_58 = (var_56, var_57)
    var_59 = {var_22: var_58}
    var_60 = 44
    var_61 = [var_55, var_59, var_60]
    var_62 = [var_50, var_54]
    var_63 = module_0.map_structure_zip(var_0, var_62)



# Parsed testcases at query #2
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
    var_25 = 'c'
    var_26 = {var_25: var_2}
    var_27 = {var_23: var_1, var_24: var_26}
    var_28 = {var_25: var_15}
    var_29 = {var_23: var_14, var_24: var_28}
    var_30 = [var_27, var_29]
    var_31 = module_0.map_structure_zip(var_0, var_30)
    var_32 = (var_2, var_3)
    var_33 = (var_1, var_32)
    var_34 = (var_15, var_17)
    var_35 = (var_14, var_34)
    var_36 = [var_33, var_35]
    var_37 = module_0.map_structure_zip(var_0, var_36)
    var_38 = 'Point'
    var_39 = 'x'
    var_40 = 'y'
    var_41 = [var_39, var_40]
    var_42 = 11
    var_43 = 22
    var_44 = lambda x, y: x.val + y.val
    var_45 = lambda x, y: len(x) + len(y)
    var_46 = [var_1, var_2]
    var_47 = [var_3, var_5, var_6]
    var_48 = [var_46, var_47]
    var_49 = module_0.map_structure_zip(var_45, var_48)
    assert var_49 == 5
    var_50 = {var_1, var_2}
    var_51 = {var_3, var_5}
    var_52 = [var_50, var_51]
    var_53 = module_0.map_structure_zip(var_0, var_52)
    var_54 = [var_52, var_53]
    var_55 = module_0.map_structure_zip(var_0, var_54)
    assert var_55 == 3



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
    var_36 = 11
    var_37 = 22
    var_38 = {var_22: var_1}
    var_39 = [var_2, var_3]
    var_40 = [var_38, var_39]
    var_41 = {var_22: var_14}
    var_42 = [var_15, var_17]
    var_43 = [var_41, var_42]
    var_44 = {var_22: var_36}
    var_45 = 33
    var_46 = [var_37, var_45]
    var_47 = [var_44, var_46]
    var_48 = [var_40, var_43]
    var_49 = module_0.map_structure_zip(var_0, var_48)
    var_50 = lambda x, y, z: x + y + z
    var_51 = [var_1, var_2, var_3]
    var_52 = module_0.map_structure_zip(var_50, var_51)
    assert var_52 == 6
    var_53 = [var_1, var_2]
    var_54 = [var_14, var_15]
    var_55 = lambda x, y: x + y
    var_56 = 1
    var_57 = {var_56}
    var_58 = 2
    var_59 = {var_58}
    var_60 = [var_57, var_59]
    var_61 = module_0.map_structure_zip(var_0, var_60)
    var_62 = [var_56, var_57]
    var_63 = module_0.no_map_instance(var_62)
    var_64 = [var_14, var_15]
    var_65 = module_0.no_map_instance(var_64)
    var_66 = [var_63, var_65]
    var_67 = module_0.map_structure_zip(var_0, var_66)



# Parsed testcases at query #4
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
    var_12 = [var_1, var_2]
    var_13 = [var_4, var_5]
    var_14 = [var_7, var_8]
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = 0
    var_18 = lambda *args: args[var_17] * args[var_1]
    var_19 = 'a'
    var_20 = 'b'
    var_21 = {var_19: var_1, var_20: var_2}
    var_22 = 'bo'
    var_23 = {var_19: var_4, var_22: var_5}
    var_24 = [var_21, var_23]
    var_25 = {var_19: var_1, var_20: var_2}
    var_26 = {var_19: var_4, var_20: var_5}
    var_27 = [var_25, var_26]
    var_28 = {var_19: var_1, var_20: var_2}
    var_29 = {var_19: var_4, var_20: var_5}
    var_30 = [var_28, var_29]
    var_31 = module_0.map_structure_zip(var_18, var_30)
    var_32 = 'Point'
    var_33 = 'x'
    var_34 = 'y'
    var_35 = [var_33, var_34]



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10
    var_4 = 1
    var_5 = lambda x: x + var_4
    var_6 = 3
    var_7 = [var_4, var_0, var_6]
    var_8 = module_0.map_structure(var_5, var_7)
    var_9 = lambda x: x * var_0
    var_10 = [var_4, var_0]
    var_11 = 4
    var_12 = [var_6, var_11]
    var_13 = [var_10, var_12]
    var_14 = module_0.map_structure(var_9, var_13)
    var_15 = lambda x: x * var_0
    var_16 = (var_4, var_0, var_6)
    var_17 = module_0.map_structure(var_15, var_16)
    var_18 = 'Point'
    var_19 = 'x'
    var_20 = 'y'
    var_21 = [var_19, var_20]
    var_22 = lambda x: x + var_4
    var_23 = 'a'
    var_24 = 'b'
    var_25 = [var_0, var_6]
    var_26 = {var_23: var_4, var_24: var_25}
    var_27 = [var_6, var_11]
    var_28 = {var_23: var_0, var_24: var_27}
    var_29 = lambda x: x + var_4
    var_30 = module_0.map_structure(var_29, var_26)
    var_31 = {var_4, var_0, var_6}
    var_32 = 10
    var_33 = lambda x: x * var_32
    var_34 = module_0.map_structure(var_33, var_31)
    var_35 = lambda x: x.val + var_4
    var_36 = [var_4, var_0, var_6]
    var_37 = lambda x: len(x.items)
    var_38 = {var_23: var_6}
    var_39 = (var_0, var_38)
    var_40 = {var_2}
    var_41 = [var_11, var_40]
    var_42 = [var_4, var_39, var_41]
    var_43 = {var_23: var_11}
    var_44 = (var_6, var_43)
    var_45 = 6
    var_46 = {var_45}
    var_47 = [var_2, var_46]
    var_48 = [var_0, var_44, var_47]
    var_49 = lambda x: x + var_4
    var_50 = module_0.map_structure(var_49, var_42)
    var_51 = (var_23, var_4)
    var_52 = (var_24, var_0)
    var_53 = [var_51, var_52]
    var_54 = lambda x: x * var_0



# Parsed testcases at query #7
#--------------------------




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
    var_10 = [var_1]
    var_11 = [var_2]
    var_12 = [var_10, var_11]
    var_13 = [var_4]
    var_14 = [var_5]
    var_15 = [var_13, var_14]
    var_16 = [var_12, var_15]
    var_17 = module_0.map_structure_zip(var_9, var_16)
    var_18 = lambda x, y: x * y
    var_19 = (var_1, var_2)
    var_20 = (var_4, var_5)
    var_21 = [var_19, var_20]
    var_22 = module_0.map_structure_zip(var_18, var_21)
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_1, var_24: var_2}
    var_26 = 'tuple'
    var_27 = 10
    var_28 = 20
    var_29 = {var_23: var_27, var_26: var_28}
    var_30 = {var_23: var_27, var_24: var_28}
    var_31 = lambda x, y: x + y
    var_32 = [var_25, var_30]
    var_33 = module_0.map_structure_zip(var_31, var_32)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = [var_1, var_2]
    var_40 = module_0.no_map_instance(var_39)
    var_41 = [var_4, var_5]
    var_42 = lambda x, y: len(x) + len(y)
    var_43 = [var_40, var_41]
    var_44 = module_0.map_structure_zip(var_42, var_43)
    assert var_44 == 4
    var_45 = lambda x, y: x + y
    var_46 = [var_1]
    var_47 = [var_2]
    var_48 = [var_46, var_47]
    var_49 = module_0.map_structure_zip(var_45, var_48)
    var_50 = lambda x, y: x + y
    var_51 = 1
    var_52 = {var_51}
    var_53 = 2
    var_54 = {var_53}
    var_55 = [var_52, var_54]
    var_56 = module_0.map_structure_zip(var_50, var_55)
    var_57 = lambda x, y: x + y
    var_58 = [var_51, var_52]
    var_59 = module_0.map_structure_zip(var_57, var_58)
    assert var_59 == 3



# Parsed testcases at query #9
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
    var_15 = lambda x, y: x + y
    var_16 = module_0.map_structure_zip(var_15, var_14)
    var_17 = [var_1]
    var_18 = [var_2]
    var_19 = [var_17, var_18]
    var_20 = [var_11]
    var_21 = [var_12]
    var_22 = [var_20, var_21]
    var_23 = [var_19, var_22]
    var_24 = lambda x, y: x + y
    var_25 = module_0.map_structure_zip(var_24, var_23)
    var_26 = (var_1, var_2)
    var_27 = (var_11, var_12)
    var_28 = [var_26, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'Point'
    var_32 = 'x'
    var_33 = 'y'
    var_34 = [var_32, var_33]
    var_35 = lambda x, y: x + y
    var_36 = 0
    var_37 = 6
    var_38 = 8
    var_39 = 'a'
    var_40 = 'b'
    var_41 = {var_39: var_1, var_40: var_2}
    var_42 = {var_39: var_6, var_40: var_7}
    var_43 = [var_41, var_42]
    var_44 = lambda x, y: x + y
    var_45 = module_0.map_structure_zip(var_44, var_43)
    var_46 = {var_39: var_1}
    var_47 = (var_2, var_11)
    var_48 = [var_46, var_47]
    var_49 = {var_39: var_6}
    var_50 = 30
    var_51 = (var_7, var_50)
    var_52 = [var_49, var_51]
    var_53 = [var_48, var_52]
    var_54 = 11
    var_55 = {var_39: var_54}
    var_56 = 22
    var_57 = 33
    var_58 = (var_56, var_57)
    var_59 = [var_55, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_53)
    var_62 = lambda x, y: x.val + y.val
    var_63 = {var_1, var_2}
    var_64 = {var_11, var_12}
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_65)
    var_68 = [var_67, var_2]
    var_69 = [var_11, var_12]
    var_70 = lambda x, y: len(x) + len(y)



# Parsed testcases at query #10
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
    var_15 = lambda x: str(x)
    var_16 = (var_2, var_0)
    var_17 = module_0.map_structure(var_15, var_16)
    var_18 = 'Point'
    var_19 = 'x'
    var_20 = 'y'
    var_21 = [var_19, var_20]
    var_22 = 10
    var_23 = lambda x: x * var_22
    var_24 = 20
    var_25 = 'a'
    var_26 = 'b'
    var_27 = [var_0, var_3]
    var_28 = {var_25: var_2, var_26: var_27}
    var_29 = lambda x: x + var_2
    var_30 = module_0.map_structure(var_29, var_28)
    var_31 = (var_25, var_2)
    var_32 = (var_26, var_0)
    var_33 = [var_31, var_32]
    var_34 = lambda x: x * var_0
    var_35 = (var_25, var_0)
    var_36 = 4
    var_37 = (var_26, var_36)
    var_38 = [var_35, var_37]
    var_39 = {var_2, var_0, var_3}
    var_40 = lambda x: x + var_2
    var_41 = module_0.map_structure(var_40, var_39)
    var_42 = 5
    var_43 = lambda x: x + var_42
    var_44 = module_0.map_structure(var_43, var_22)
    assert var_44 == 15
    var_45 = {var_2, var_0}
    var_46 = lambda x: len(x)
    var_47 = module_0.map_structure(var_46, var_45)
    assert var_47 == 2
    var_48 = [var_2, var_0]
    var_49 = lambda x: x.val
    var_50 = 'found'
    var_51 = lambda x: var_50
    var_52 = (var_0, var_3)
    var_53 = {var_25: var_52}
    var_54 = {var_36, var_42}
    var_55 = [var_2, var_53, var_54]



# Parsed testcases at query #11
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
    var_21 = lambda x: x + var_0
    var_22 = 10
    var_23 = lambda x: x * var_22
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_0, var_25: var_3}
    var_27 = module_0.map_structure(var_23, var_26)
    var_28 = [var_0, var_3]
    var_29 = 'c'
    var_30 = {var_29: var_5}
    var_31 = {var_24: var_28, var_25: var_30}
    var_32 = [var_3, var_5]
    var_33 = 4
    var_34 = {var_29: var_33}
    var_35 = {var_24: var_32, var_25: var_34}
    var_36 = lambda x: x + var_0
    var_37 = module_0.map_structure(var_36, var_31)
    var_38 = lambda x: x * var_3
    var_39 = {var_0, var_3, var_5}
    var_40 = module_0.map_structure(var_38, var_39)
    var_41 = [var_0, var_3, var_5]
    var_42 = lambda x: len(x)
    var_43 = 5
    var_44 = lambda x: x.val + var_43
    var_45 = (var_0, var_3)
    var_46 = module_0.no_map_instance(var_45)
    var_47 = lambda x: len(x)
    var_48 = module_0.map_structure(var_47, var_46)
    assert var_48 == 2



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
    var_22 = lambda x: x * var_0
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_2, var_24: var_0}
    var_26 = module_0.map_structure(var_22, var_25)
    var_27 = (var_23, var_2)
    var_28 = (var_24, var_0)
    var_29 = [var_27, var_28]
    var_30 = lambda x: x + var_2
    var_31 = lambda x: x * var_0
    var_32 = {var_2, var_0}
    var_33 = module_0.map_structure(var_31, var_32)
    var_34 = 5
    var_35 = lambda x: x + var_34
    var_36 = module_0.map_structure(var_35, var_19)
    assert var_36 == 15
    var_37 = lambda x: len(x)
    var_38 = [var_2, var_0, var_3]
    var_39 = [var_2, var_0, var_3]
    var_40 = module_0.no_map_instance(var_39)
    var_41 = lambda x: len(x)
    var_42 = module_0.map_structure(var_41, var_40)
    assert var_42 == 3
    var_43 = (var_2, var_0)
    var_44 = {var_23: var_43}
    var_45 = 4
    var_46 = [var_3, var_45]
    var_47 = {var_24: var_46}
    var_48 = 6
    var_49 = (var_34, var_48)
    var_50 = [var_44, var_47, var_49]
    var_51 = (var_0, var_45)
    var_52 = {var_23: var_51}
    var_53 = 8
    var_54 = [var_48, var_53]
    var_55 = {var_24: var_54}
    var_56 = 12
    var_57 = (var_19, var_56)
    var_58 = [var_52, var_55, var_57]
    var_59 = lambda x: x * var_0
    var_60 = module_0.map_structure(var_59, var_50)
    var_61 = (var_2, var_0)
    var_62 = module_0.no_map_instance(var_61)
    var_63 = lambda x: len(x)
    var_64 = module_0.map_structure(var_63, var_62)
    assert var_64 == 2



# Parsed testcases at query #13
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = module_0.map_structure(var_1, var_0)
    assert var_2 == 2



# Parsed testcases at query #14
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.map_structure(var_1, var_4)



