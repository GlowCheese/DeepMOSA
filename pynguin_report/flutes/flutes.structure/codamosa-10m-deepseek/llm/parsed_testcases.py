####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
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
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x * var_0
    var_13 = (var_2, var_0, var_3)
    var_14 = module_0.map_structure(var_12, var_13)
    var_15 = lambda x: x * var_0
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = lambda x: x * var_0
    var_33 = [var_2, var_0, var_3]
    var_34 = module_0.map_structure(var_32, var_33)



# Parsed testcases at query #2
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = lambda x, y: x + y
    var_32 = [var_29, var_30]
    var_33 = module_0.map_structure_zip(var_31, var_32)
    var_34 = [var_0, var_1]
    var_35 = (var_2, var_4)
    var_36 = {var_27: var_34, var_28: var_35}
    var_37 = [var_5, var_6]
    var_38 = (var_15, var_16)
    var_39 = {var_27: var_37, var_28: var_38}
    var_40 = lambda x, y: x + y
    var_41 = [var_36, var_39]
    var_42 = module_0.map_structure_zip(var_40, var_41)
    var_43 = [var_6, var_16]
    var_44 = 10
    var_45 = 12
    var_46 = (var_44, var_45)
    var_47 = {var_27: var_43, var_28: var_46}
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x, y: x + y
    var_51 = [var_49, var_49]
    var_52 = module_0.map_structure_zip(var_50, var_51)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #3
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = {var_0, var_1}
    var_50 = {var_2, var_4}
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x.union(y)
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1]
    var_55 = [var_2, var_4]
    var_56 = lambda x, y: len(x) + len(y)
    var_57 = module_0.map_structure_zip(var_56, var_51)
    assert var_57 == 4
    var_58 = [var_0, var_1]
    var_59 = module_0.no_map_instance(var_58)
    var_60 = [var_2, var_4]
    var_61 = module_0.no_map_instance(var_60)
    var_62 = [var_59, var_61]
    var_63 = lambda x, y: len(x) + len(y)
    var_64 = module_0.map_structure_zip(var_63, var_62)
    assert var_64 == 4
    var_65 = 'All tests passed!'
    var_66 = print(var_65)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------



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
    var_10 = lambda x: x + var_0
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_1}
    var_18 = lambda x: x * var_1
    var_19 = module_0.map_structure(var_18, var_17)
    var_20 = {var_0, var_1, var_2}
    var_21 = lambda x: x * var_1
    var_22 = module_0.map_structure(var_21, var_20)
    var_23 = [var_0, var_1]
    var_24 = (var_2, var_7)
    var_25 = {var_15: var_23, var_16: var_24}
    var_26 = lambda x: x + var_0
    var_27 = module_0.map_structure(var_26, var_25)
    var_28 = 'mapped'
    var_29 = lambda x: var_28
    var_30 = module_0.map_structure(var_29, var_25)
    assert var_30 == 'mapped'
    var_31 = [var_0, var_1, var_2]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: var_28
    var_34 = module_0.map_structure(var_33, var_32)
    assert var_34 == 'mapped'
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #8
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_0, var_1}
    var_41 = {var_2, var_4}
    var_42 = [var_40, var_41]
    var_43 = module_0.map_structure_zip(var_38, var_42)
    var_44 = {var_33: var_0}
    var_45 = {var_34: var_1}
    var_46 = [var_44, var_45]
    var_47 = {var_33: var_2}
    var_48 = {var_34: var_4}
    var_49 = [var_47, var_48]
    var_50 = [var_46, var_49]
    var_51 = 0
    var_52 = lambda x, y: {k: v + y.get(k, var_51) for (k, v) in x.items()}
    var_53 = module_0.map_structure_zip(var_52, var_50)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_55, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #9
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_0, var_1}
    var_41 = {var_2, var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = {var_33: var_0}
    var_46 = {var_34: var_1}
    var_47 = [var_45, var_46]
    var_48 = {var_33: var_2}
    var_49 = {var_34: var_4}
    var_50 = [var_48, var_49]
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: {k: v + y[k] for (k, v) in x.items()}
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_55, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #10
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = lambda x: x * var_0
    var_33 = [var_2, var_0, var_3]
    var_34 = module_0.map_structure(var_32, var_33)



# Parsed testcases at query #11
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #12
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x + y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = [var_0, var_1]
    var_35 = (var_2, var_4)
    var_36 = {var_27: var_34, var_28: var_35}
    var_37 = [var_5, var_6]
    var_38 = (var_15, var_16)
    var_39 = {var_27: var_37, var_28: var_38}
    var_40 = [var_36, var_39]
    var_41 = lambda x, y: x + y
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = [var_6, var_16]
    var_44 = 10
    var_45 = 12
    var_46 = (var_44, var_45)
    var_47 = {var_27: var_43, var_28: var_46}
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = [var_49, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #13
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x + y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = {var_27: var_0}
    var_35 = {var_28: var_1}
    var_36 = [var_34, var_35]
    var_37 = {var_27: var_2}
    var_38 = {var_28: var_4}
    var_39 = [var_37, var_38]
    var_40 = [var_36, var_39]
    var_41 = lambda x, y: {}
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = [var_0, var_1, var_2]
    var_44 = module_0.no_map_instance(var_43)
    var_45 = [var_44, var_44]
    var_46 = lambda x, y: x + y
    var_47 = module_0.map_structure_zip(var_46, var_45)
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = 'Test 1 passed: Simple list'
    var_7 = print(var_6)
    var_8 = [var_0, var_1]
    var_9 = 4
    var_10 = [var_2, var_9]
    var_11 = [var_8, var_10]
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'Test 2 passed: Nested list'
    var_14 = print(var_13)
    var_15 = (var_0, var_1, var_2)
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = 'Test 3 passed: Tuple'
    var_18 = print(var_17)
    var_19 = 'Point'
    var_20 = 'x'
    var_21 = 'y'
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure(var_4, var_15)
    var_24 = 'Test 4 passed: Namedtuple'
    var_25 = print(var_24)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = {var_26: var_0, var_27: var_1}
    var_29 = module_0.map_structure(var_4, var_28)
    var_30 = 'Test 5 passed: Dictionary'
    var_31 = print(var_30)
    var_32 = {var_0, var_1, var_2}
    var_33 = module_0.map_structure(var_4, var_32)
    var_34 = 'Test 6 passed: Set'
    var_35 = print(var_34)
    var_36 = [var_0, var_1, var_2]
    var_37 = module_0.map_structure(var_4, var_32)
    var_38 = 'Test 7 passed: Registered non-mappable class'
    var_39 = print(var_38)
    var_40 = [var_0, var_1, var_2]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = module_0.map_structure(var_4, var_41)
    var_43 = 'Test 8 passed: Non-mappable instance'
    var_44 = print(var_43)
    var_45 = 'c'
    var_46 = [var_0, var_1]
    var_47 = (var_2, var_9)
    var_48 = 5
    var_49 = 6
    var_50 = {var_48, var_49}
    var_51 = {var_26: var_46, var_27: var_47, var_45: var_50}
    var_52 = module_0.map_structure(var_4, var_51)
    var_53 = [var_1, var_9]
    var_54 = 8
    var_55 = (var_49, var_54)
    var_56 = 10
    var_57 = 12
    var_58 = {var_56, var_57}
    var_59 = {var_26: var_53, var_27: var_55, var_45: var_58}
    var_60 = 'Test 9 passed: Complex nested structure'
    var_61 = print(var_60)
    var_62 = 'All tests passed!'
    var_63 = print(var_62)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = 'Test 1 passed: Simple list'
    var_7 = print(var_6)
    var_8 = [var_0, var_1]
    var_9 = 4
    var_10 = [var_2, var_9]
    var_11 = [var_8, var_10]
    var_12 = lambda x: x + var_0
    var_13 = module_0.map_structure(var_12, var_11)
    var_14 = 'Test 2 passed: Nested list'
    var_15 = print(var_14)
    var_16 = (var_0, var_1, var_2)
    var_17 = lambda x: x * var_2
    var_18 = module_0.map_structure(var_17, var_16)
    var_19 = 'Test 3 passed: Tuple'
    var_20 = print(var_19)
    var_21 = 'Point'
    var_22 = 'x'
    var_23 = 'y'
    var_24 = [var_22, var_23]
    var_25 = lambda x: x * var_1
    var_26 = module_0.map_structure(var_25, var_16)
    var_27 = 'Test 4 passed: Namedtuple'
    var_28 = print(var_27)
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_0, var_30: var_1}
    var_32 = lambda x: x * var_1
    var_33 = module_0.map_structure(var_32, var_31)
    var_34 = 'Test 5 passed: Dictionary'
    var_35 = print(var_34)
    var_36 = {var_0, var_1, var_2}
    var_37 = lambda x: x * var_1
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = 'Test 6 passed: Set'
    var_40 = print(var_39)
    var_41 = [var_0, var_1, var_2]
    var_42 = lambda x: x * var_1
    var_43 = module_0.map_structure(var_42, var_36)
    var_44 = lambda x: len(x)
    var_45 = module_0.map_structure(var_44, var_36)
    assert var_45 == 3
    var_46 = 'Test 7 passed: Registered non-mappable class'
    var_47 = print(var_46)
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x: sum(x)
    var_51 = module_0.map_structure(var_50, var_49)
    assert var_51 == 6
    var_52 = 'Test 8 passed: Non-mappable instance'
    var_53 = print(var_52)
    var_54 = [var_0, var_1, var_2]
    var_55 = 5
    var_56 = 6
    var_57 = (var_9, var_55, var_56)
    var_58 = {var_29: var_54, var_30: var_57}
    var_59 = lambda x: x * var_1
    var_60 = module_0.map_structure(var_59, var_58)
    var_61 = [var_1, var_9, var_56]
    var_62 = 8
    var_63 = 10
    var_64 = 12
    var_65 = (var_62, var_63, var_64)
    var_66 = {var_29: var_61, var_30: var_65}
    var_67 = 'Test 9 passed: Mixed nested structures'
    var_68 = print(var_67)
    var_69 = 'All tests passed!'
    var_70 = print(var_69)



# Parsed testcases at query #18
#--------------------------



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
    var_10 = lambda x: x + var_0
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_2
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_1}
    var_18 = lambda x: x * var_7
    var_19 = module_0.map_structure(var_18, var_17)
    var_20 = {var_0, var_1, var_2}
    var_21 = 5
    var_22 = lambda x: x * var_21
    var_23 = module_0.map_structure(var_22, var_20)
    var_24 = set(var_23)
    var_25 = 'Point'
    var_26 = 'x'
    var_27 = 'y'
    var_28 = [var_26, var_27]
    var_29 = 6
    var_30 = lambda x: x * var_29
    var_31 = 12
    var_32 = [var_0, var_1]
    var_33 = (var_2, var_7)
    var_34 = {var_15: var_32, var_16: var_33}
    var_35 = 7
    var_36 = lambda x: x * var_35
    var_37 = module_0.map_structure(var_36, var_34)
    var_38 = 14
    var_39 = [var_35, var_38]
    var_40 = 21
    var_41 = 28
    var_42 = (var_40, var_41)
    var_43 = {var_15: var_39, var_16: var_42}
    var_44 = [var_0, var_1, var_2]
    var_45 = 8
    var_46 = lambda x: x * var_45
    var_47 = [var_0, var_1, var_2]
    var_48 = [var_0, var_1, var_2]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = 9
    var_51 = lambda x: x * var_50
    var_52 = module_0.map_structure(var_51, var_49)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #19
#--------------------------



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
    var_10 = lambda x: x + var_0
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
    var_29 = [var_0, var_1]
    var_30 = (var_2, var_7)
    var_31 = {var_21: var_29, var_22: var_30}
    var_32 = lambda x: x + var_0
    var_33 = module_0.map_structure(var_32, var_31)
    var_34 = [var_0, var_1, var_2]
    var_35 = lambda x: x * var_1
    var_36 = module_0.map_structure(var_35, var_31)
    var_37 = [var_0, var_1, var_2]
    var_38 = module_0.no_map_instance(var_37)
    var_39 = lambda x: x * var_1
    var_40 = module_0.map_structure(var_39, var_38)
    var_41 = 'All tests passed!'
    var_42 = print(var_41)



# Parsed testcases at query #20
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = lambda x: x * var_0
    var_33 = [var_2, var_0, var_3]
    var_34 = module_0.map_structure(var_32, var_33)
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #21
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = 0
    var_48 = lambda x, y: {k: v + y.get(k, var_47) for (k, v) in x.items()}
    var_49 = module_0.map_structure_zip(var_48, var_46)
    var_50 = {var_33: var_4}
    var_51 = {var_34: var_6}
    var_52 = [var_50, var_51]
    var_53 = {var_0, var_1}
    var_54 = {var_2, var_4}
    var_55 = [var_53, var_54]
    var_56 = module_0.map_structure_zip(var_48, var_55)
    var_57 = [var_0, var_1, var_2]
    var_58 = module_0.no_map_instance(var_57)
    var_59 = [var_58, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = []
    var_63 = []
    var_64 = [var_62, var_63]
    var_65 = lambda x, y: x + y
    var_66 = module_0.map_structure_zip(var_65, var_64)
    var_67 = [var_0, var_1, var_2]
    var_68 = [var_67]
    var_69 = lambda x: x * var_1
    var_70 = module_0.map_structure_zip(var_69, var_68)
    var_71 = 'All tests passed!'
    var_72 = print(var_71)



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = [var_2, var_0, var_3]
    var_24 = module_0.no_map_instance(var_23)
    var_25 = lambda x: x * var_0
    var_26 = module_0.map_structure(var_25, var_24)
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.map_structure(var_27, var_28)



# Parsed testcases at query #24
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = module_0.map_structure(var_4, var_17)
    var_20 = [var_0, var_1, var_2]
    var_21 = module_0.no_map_instance(var_20)
    var_22 = module_0.map_structure(var_4, var_21)
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = 'x'
    var_35 = {var_34: var_0}
    var_36 = {var_34: var_1}
    var_37 = [var_35, var_36]
    var_38 = {var_34: var_2}
    var_39 = {var_34: var_4}
    var_40 = [var_38, var_39]
    var_41 = [var_37, var_40]
    var_42 = lambda a, b: {var_34: a[var_34] + b[var_34]}
    var_43 = module_0.map_structure_zip(var_42, var_41)
    var_44 = 1
    var_45 = 2
    var_46 = {var_44, var_45}
    var_47 = 3
    var_48 = 4
    var_49 = {var_47, var_48}
    var_50 = [var_46, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = [var_44, var_45, var_46]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = [var_54, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = lambda x, y: x
    var_59 = module_0.map_structure_zip(var_58, var_55)
    var_60 = 'All tests passed!'
    var_61 = print(var_60)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_33: var_0}
    var_41 = {var_34: var_1}
    var_42 = [var_40, var_41]
    var_43 = {var_33: var_2}
    var_44 = {var_34: var_4}
    var_45 = [var_43, var_44]
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: {}
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = 'hello'
    var_50 = 'world'
    var_51 = [var_49, var_50]
    var_52 = ' '
    var_53 = lambda x, y: x + var_52 + y
    var_54 = module_0.map_structure_zip(var_53, var_51)
    assert var_54 == 'hello world'
    var_55 = [var_0, var_1, var_2]
    var_56 = module_0.no_map_instance(var_55)
    var_57 = [var_4, var_5, var_6]
    var_58 = module_0.no_map_instance(var_57)
    var_59 = [var_56, var_58]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_59)
    var_62 = 1
    var_63 = 2
    var_64 = {var_62, var_63}
    var_65 = 3
    var_66 = 4
    var_67 = {var_65, var_66}
    var_68 = [var_64, var_67]
    var_69 = lambda x, y: x | y
    var_70 = module_0.map_structure_zip(var_69, var_68)
    var_71 = 'ERROR: Expected ValueError for set, but got result:'
    var_72 = print(var_71, var_70)
    var_73 = 'All tests passed!'
    var_74 = print(var_73)



# Parsed testcases at query #2
#--------------------------



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
    var_11 = 7
    var_12 = 9
    var_13 = [var_5, var_11, var_12]
    var_14 = [var_0, var_1]
    var_15 = [var_2, var_4]
    var_16 = [var_14, var_15]
    var_17 = [var_5, var_6]
    var_18 = 8
    var_19 = [var_11, var_18]
    var_20 = [var_17, var_19]
    var_21 = [var_16, var_20]
    var_22 = lambda x, y: x + y
    var_23 = module_0.map_structure_zip(var_22, var_21)
    var_24 = [var_6, var_18]
    var_25 = 10
    var_26 = 12
    var_27 = [var_25, var_26]
    var_28 = [var_24, var_27]
    var_29 = (var_0, var_1, var_2)
    var_30 = (var_4, var_5, var_6)
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x + y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = (var_5, var_11, var_12)
    var_35 = 'Point'
    var_36 = 'x'
    var_37 = 'y'
    var_38 = [var_36, var_37]
    var_39 = lambda x, y: x + y
    var_40 = module_0.map_structure_zip(var_39, var_31)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = {var_41: var_0, var_42: var_1}
    var_44 = {var_41: var_2, var_42: var_4}
    var_45 = [var_43, var_44]
    var_46 = lambda x, y: x + y
    var_47 = module_0.map_structure_zip(var_46, var_45)
    var_48 = {var_41: var_4, var_42: var_6}
    var_49 = 1
    var_50 = 2
    var_51 = {var_49, var_50}
    var_52 = 3
    var_53 = 4
    var_54 = {var_52, var_53}
    var_55 = [var_51, var_54]
    var_56 = lambda x, y: x + y
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = [var_49, var_50, var_51]
    var_59 = (var_53, var_54, var_6)
    var_60 = [var_58, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)
    var_63 = [var_54, var_11, var_12]
    var_64 = [var_49, var_50, var_51]
    var_65 = module_0.no_map_instance(var_64)
    var_66 = [var_65, var_65]
    var_67 = lambda x, y: x + y
    var_68 = module_0.map_structure_zip(var_67, var_66)
    var_69 = [var_50, var_53, var_6]
    var_70 = 'All tests passed!'
    var_71 = print(var_70)



# Parsed testcases at query #3
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = {var_0, var_1}
    var_41 = {var_2, var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = [var_0, var_1]
    var_46 = [var_2, var_4]
    var_47 = {var_33: var_45, var_34: var_46}
    var_48 = [var_5, var_6]
    var_49 = [var_15, var_16]
    var_50 = {var_33: var_48, var_34: var_49}
    var_51 = [var_47, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_55, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = []
    var_60 = []
    var_61 = [var_59, var_60]
    var_62 = lambda x, y: x + y
    var_63 = module_0.map_structure_zip(var_62, var_61)
    var_64 = [var_0, var_1, var_2]
    var_65 = [var_64]
    var_66 = lambda x: x * var_1
    var_67 = module_0.map_structure_zip(var_66, var_65)
    var_68 = 'All tests passed!'
    var_69 = print(var_68)



# Parsed testcases at query #4
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_31)
    var_40 = 1
    var_41 = 2
    var_42 = {var_40, var_41}
    var_43 = 3
    var_44 = 4
    var_45 = {var_43, var_44}
    var_46 = [var_42, var_45]
    var_47 = module_0.map_structure_zip(var_38, var_46)
    var_48 = [var_40, var_41, var_42]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = [var_49, var_49]
    var_51 = lambda x, y: x + y
    var_52 = module_0.map_structure_zip(var_51, var_50)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #5
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_31)
    var_40 = {var_0, var_1}
    var_41 = {var_2, var_4}
    var_42 = [var_40, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = [var_0, var_1]
    var_46 = (var_2, var_4)
    var_47 = [var_45, var_46]
    var_48 = lambda x, y: x + y
    var_49 = module_0.map_structure_zip(var_48, var_47)
    var_50 = 'All tests passed!'
    var_51 = print(var_50)



# Parsed testcases at query #6
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.map_structure(var_4, var_20)
    var_24 = 'All tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #7
#--------------------------



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
    var_18 = lambda x: x * var_0
    var_19 = {var_13: var_2, var_14: var_0}
    var_20 = module_0.map_structure(var_18, var_19)
    var_21 = lambda x: x * var_0
    var_22 = {var_2, var_0, var_3}
    var_23 = module_0.map_structure(var_21, var_22)
    var_24 = [var_2, var_0, var_3]
    var_25 = module_0.no_map_instance(var_24)
    var_26 = lambda x: x * var_0
    var_27 = module_0.map_structure(var_26, var_25)
    var_28 = 'All tests passed!'
    var_29 = print(var_28)



# Parsed testcases at query #8
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = 'Point'
    var_20 = 'x'
    var_21 = 'y'
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure(var_4, var_17)
    var_24 = [var_0, var_1, var_2]
    var_25 = module_0.no_map_instance(var_24)
    var_26 = module_0.map_structure(var_4, var_25)
    var_27 = 'All tests passed!'
    var_28 = print(var_27)



# Parsed testcases at query #9
#--------------------------



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
    var_10 = lambda x: x + var_0
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_1}
    var_18 = lambda x: x * var_1
    var_19 = module_0.map_structure(var_18, var_17)
    var_20 = {var_0, var_1, var_2}
    var_21 = lambda x: x * var_1
    var_22 = module_0.map_structure(var_21, var_20)
    var_23 = [var_0, var_1]
    var_24 = (var_2, var_7)
    var_25 = {var_15: var_23, var_16: var_24}
    var_26 = lambda x: x + var_0
    var_27 = module_0.map_structure(var_26, var_25)
    var_28 = [var_1, var_2]
    var_29 = 5
    var_30 = (var_7, var_29)
    var_31 = {var_15: var_28, var_16: var_30}
    var_32 = [var_0, var_1, var_2]
    var_33 = lambda x: x * var_1
    var_34 = module_0.map_structure(var_33, var_25)
    var_35 = [var_0, var_1, var_2]
    var_36 = module_0.no_map_instance(var_35)
    var_37 = lambda x: x * var_1
    var_38 = module_0.map_structure(var_37, var_36)
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #10
#--------------------------



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
    var_15 = 'Point'
    var_16 = 'x'
    var_17 = 'y'
    var_18 = [var_16, var_17]
    var_19 = lambda x: x * var_0
    var_20 = lambda x: x * var_0
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_2, var_22: var_0}
    var_24 = module_0.map_structure(var_20, var_23)
    var_25 = lambda x: x * var_0
    var_26 = {var_2, var_0, var_3}
    var_27 = module_0.map_structure(var_25, var_26)
    var_28 = lambda x: x * var_0
    var_29 = [var_2, var_0, var_3]
    var_30 = module_0.map_structure(var_28, var_29)
    var_31 = [var_2, var_0, var_3]
    var_32 = module_0.no_map_instance(var_31)
    var_33 = lambda x: x * var_0
    var_34 = module_0.map_structure(var_33, var_32)
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #11
#--------------------------



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
    var_25 = lambda x, y: x * y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = [var_0, var_1]
    var_35 = (var_2, var_4)
    var_36 = {var_27: var_34, var_28: var_35}
    var_37 = [var_5, var_6]
    var_38 = (var_15, var_16)
    var_39 = {var_27: var_37, var_28: var_38}
    var_40 = [var_36, var_39]
    var_41 = lambda x, y: x + y
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = [var_0, var_1, var_2]
    var_44 = [var_43]
    var_45 = lambda x: x * var_1
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = []
    var_48 = []
    var_49 = [var_47, var_48]
    var_50 = lambda x, y: x + y
    var_51 = module_0.map_structure_zip(var_50, var_49)
    var_52 = [var_0, var_1, var_2]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = [var_53, var_53]
    var_55 = lambda x, y: x + y
    var_56 = module_0.map_structure_zip(var_55, var_54)
    var_57 = 'All tests passed!'
    var_58 = print(var_57)



# Parsed testcases at query #12
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = 1
    var_41 = 2
    var_42 = {var_40, var_41}
    var_43 = 3
    var_44 = 4
    var_45 = {var_43, var_44}
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: x + y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = [var_40, var_41]
    var_50 = (var_42, var_44)
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x + y
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_40, var_41, var_42]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_55, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #13
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_31)
    var_40 = [var_0, var_1, var_2]
    var_41 = module_0.no_map_instance(var_40)
    var_42 = [var_41, var_41]
    var_43 = lambda x, y: x + y
    var_44 = module_0.map_structure_zip(var_43, var_42)
    var_45 = 'All tests passed!'
    var_46 = print(var_45)



# Parsed testcases at query #14
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = 'Point'
    var_20 = 'x'
    var_21 = 'y'
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure(var_4, var_17)
    var_24 = [var_0, var_1, var_2]
    var_25 = module_0.no_map_instance(var_24)
    var_26 = module_0.map_structure(var_4, var_25)
    var_27 = 'All tests passed!'
    var_28 = print(var_27)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x * var_1
    var_5 = module_0.map_structure(var_4, var_3)
    var_6 = [var_1, var_2]
    var_7 = 4
    var_8 = [var_0, var_6, var_7]
    var_9 = module_0.map_structure(var_4, var_8)
    var_10 = (var_0, var_1, var_2)
    var_11 = module_0.map_structure(var_4, var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = {var_12: var_0, var_13: var_1}
    var_15 = module_0.map_structure(var_4, var_14)
    var_16 = {var_0, var_1, var_2}
    var_17 = module_0.map_structure(var_4, var_16)
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.no_map_instance(var_18)
    var_20 = module_0.map_structure(var_4, var_19)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #16
#--------------------------



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
    var_16 = 'a'
    var_17 = 'b'
    var_18 = {var_16: var_2, var_17: var_0}
    var_19 = module_0.map_structure(var_15, var_18)
    var_20 = lambda x: x * var_0
    var_21 = {var_2, var_0, var_3}
    var_22 = module_0.map_structure(var_20, var_21)
    var_23 = 'Point'
    var_24 = 'x'
    var_25 = 'y'
    var_26 = [var_24, var_25]
    var_27 = lambda x: x * var_0
    var_28 = [var_2, var_0, var_3]
    var_29 = module_0.no_map_instance(var_28)
    var_30 = lambda x: x * var_0
    var_31 = module_0.map_structure(var_30, var_29)
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------



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
    var_38 = lambda x, y: x - y
    var_39 = module_0.map_structure_zip(var_38, var_37)
    var_40 = 1
    var_41 = 2
    var_42 = {var_40, var_41}
    var_43 = 3
    var_44 = 4
    var_45 = {var_43, var_44}
    var_46 = [var_42, var_45]
    var_47 = lambda x, y: x + y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = {var_33: var_40}
    var_50 = {var_34: var_41}
    var_51 = [var_49, var_50]
    var_52 = {var_33: var_42}
    var_53 = {var_34: var_44}
    var_54 = [var_52, var_53]
    var_55 = [var_51, var_54]
    var_56 = lambda x, y: {}
    var_57 = module_0.map_structure_zip(var_56, var_55)
    var_58 = [var_40, var_41, var_42]
    var_59 = module_0.no_map_instance(var_58)
    var_60 = [var_59, var_59]
    var_61 = lambda x, y: x + y
    var_62 = module_0.map_structure_zip(var_61, var_60)
    var_63 = 'All tests passed!'
    var_64 = print(var_63)



# Parsed testcases at query #19
#--------------------------



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
    var_33 = [var_2, var_0, var_3]
    var_34 = module_0.no_map_instance(var_33)
    var_35 = lambda x: x * var_0
    var_36 = module_0.map_structure(var_35, var_34)
    var_37 = lambda x: x * var_0
    var_38 = [var_2, var_0, var_3]
    var_39 = module_0.map_structure(var_37, var_38)
    var_40 = 'All tests passed!'
    var_41 = print(var_40)



# Parsed testcases at query #20
#--------------------------



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
    var_25 = lambda x, y: x * y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = 'Point'
    var_35 = 'x'
    var_36 = 'y'
    var_37 = [var_35, var_36]
    var_38 = lambda x, y: x + y
    var_39 = module_0.map_structure_zip(var_38, var_31)
    var_40 = {var_0, var_1}
    var_41 = {var_2, var_4}
    var_42 = [var_40, var_41]
    var_43 = module_0.map_structure_zip(var_38, var_42)
    var_44 = [var_0, var_1, var_2]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = [var_45, var_45]
    var_47 = lambda x, y: x + y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #21
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #22
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #23
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #24
#--------------------------



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
    var_33 = [var_2, var_0]
    var_34 = (var_3, var_8)
    var_35 = {var_21: var_33, var_22: var_34}
    var_36 = [var_0, var_8]
    var_37 = 6
    var_38 = 8
    var_39 = (var_37, var_38)
    var_40 = {var_21: var_36, var_22: var_39}
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_35)
    var_43 = [var_2, var_0, var_3]
    var_44 = module_0.no_map_instance(var_43)
    var_45 = lambda x: x * var_0
    var_46 = module_0.map_structure(var_45, var_44)
    var_47 = [var_2, var_0, var_3]
    var_48 = lambda x: x * var_0
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #25
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.no_map_instance(var_19)
    var_21 = module_0.map_structure(var_4, var_20)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #26
#--------------------------



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
    var_10 = lambda x: x + var_0
    var_11 = module_0.map_structure(var_10, var_9)
    var_12 = (var_0, var_1, var_2)
    var_13 = lambda x: x * var_1
    var_14 = module_0.map_structure(var_13, var_12)
    var_15 = 'a'
    var_16 = 'b'
    var_17 = {var_15: var_0, var_16: var_1}
    var_18 = lambda x: x * var_1
    var_19 = module_0.map_structure(var_18, var_17)
    var_20 = {var_0, var_1, var_2}
    var_21 = lambda x: x * var_1
    var_22 = module_0.map_structure(var_21, var_20)
    var_23 = [var_0, var_1, var_2]
    var_24 = module_0.no_map_instance(var_23)
    var_25 = lambda x: x * var_1
    var_26 = module_0.map_structure(var_25, var_24)
    var_27 = 'All tests passed!'
    var_28 = print(var_27)



# Parsed testcases at query #27
#--------------------------



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
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_0, var_28: var_1}
    var_30 = {var_27: var_2, var_28: var_4}
    var_31 = [var_29, var_30]
    var_32 = lambda x, y: x - y
    var_33 = module_0.map_structure_zip(var_32, var_31)
    var_34 = {var_27: var_0}
    var_35 = {var_28: var_1}
    var_36 = [var_34, var_35]
    var_37 = {var_27: var_2}
    var_38 = {var_28: var_4}
    var_39 = [var_37, var_38]
    var_40 = [var_36, var_39]
    var_41 = lambda x, y: {}
    var_42 = module_0.map_structure_zip(var_41, var_40)
    var_43 = {var_0, var_1}
    var_44 = {var_2, var_4}
    var_45 = [var_43, var_44]
    var_46 = module_0.map_structure_zip(var_41, var_45)
    var_47 = 'All tests passed!'
    var_48 = print(var_47)



# Parsed testcases at query #28
#--------------------------



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
    var_10 = module_0.map_structure(var_4, var_9)
    var_11 = (var_0, var_1, var_2)
    var_12 = module_0.map_structure(var_4, var_11)
    var_13 = 'a'
    var_14 = 'b'
    var_15 = {var_13: var_0, var_14: var_1}
    var_16 = module_0.map_structure(var_4, var_15)
    var_17 = {var_0, var_1, var_2}
    var_18 = module_0.map_structure(var_4, var_17)
    var_19 = 'Point'
    var_20 = 'x'
    var_21 = 'y'
    var_22 = [var_20, var_21]
    var_23 = module_0.map_structure(var_4, var_17)
    var_24 = [var_0, var_1, var_2]
    var_25 = module_0.no_map_instance(var_24)
    var_26 = module_0.map_structure(var_4, var_25)
    var_27 = 'All tests passed!'
    var_28 = print(var_27)



