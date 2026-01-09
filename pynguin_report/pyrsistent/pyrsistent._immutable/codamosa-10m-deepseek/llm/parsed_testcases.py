####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._immutable as module_0


def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = 'PointWithId'
    var_8 = module_0.immutable(var_6, var_7)
    var_9 = 17
    var_10 = 18
    var_11 = tuple()
    var_12 = -1
    var_13 = 2
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'Test 1 passed'
    var_7 = print(var_6)
    var_8 = 'x, y, id_'
    var_9 = module_0.immutable(var_8, var_1)
    var_10 = 17
    var_11 = 18
    var_12 = 'Test 2 passed'
    var_13 = print(var_12)
    var_14 = tuple()
    var_15 = -1
    var_16 = 2
    var_17 = 'Test 3 passed'
    var_18 = print(var_17)
    var_19 = module_0.immutable(var_15, var_16)
    var_20 = 3
    var_21 = 'Test 4 passed'
    var_22 = print(var_21)
    var_23 = ''
    var_24 = 'Empty'
    var_25 = module_0.immutable(var_23, var_24)
    var_26 = 'Test 5 passed'
    var_27 = print(var_26)
    var_28 = module_0.immutable(var_20, var_16)
    var_29 = 'Test 6 passed'
    var_30 = print(var_29)
    var_31 = 'a, b_, c, d_'
    var_32 = 'Mixed'
    var_33 = module_0.immutable(var_31, var_32)
    var_34 = 4
    var_35 = 10
    var_36 = 30
    var_37 = 20
    var_38 = 40
    var_39 = 'Test 7 passed'
    var_40 = print(var_39)
    var_41 = False
    var_42 = module_0.immutable(var_38, var_16, var_41)
    var_43 = 'Test 8 passed'
    var_44 = print(var_43)
    var_45 = module_0.immutable(var_38, var_16)
    var_46 = 'Test 9 passed'
    var_47 = print(var_46)
    var_48 = 'x y'
    var_49 = module_0.immutable(var_48, var_16)
    var_50 = 'Test 10 passed'
    var_51 = print(var_50)
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = 'PointWithId'
    var_8 = module_0.immutable(var_6, var_7)
    var_9 = 17
    var_10 = 18
    var_11 = tuple()
    var_12 = -1
    var_13 = 2
    var_14 = 10
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = module_0.immutable(var_6, var_1)
    var_8 = 17
    var_9 = 18
    var_10 = tuple()
    var_11 = -1
    var_12 = 2
    var_13 = module_0.immutable(var_11, var_12)
    var_14 = 3
    var_15 = ''
    var_16 = 'Empty'
    var_17 = module_0.immutable(var_15, var_16)
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'Test 1 passed: Basic functionality'
    var_7 = print(var_6)
    var_8 = 'x, y, id_'
    var_9 = module_0.immutable(var_8, var_1)
    var_10 = 17
    var_11 = 18
    var_12 = 'Test 2 passed: Frozen members'
    var_13 = print(var_12)
    var_14 = tuple()
    var_15 = -1
    var_16 = 2
    var_17 = 'Test 3 passed: Inheritance and validation'
    var_18 = print(var_17)
    var_19 = module_0.immutable(var_15, var_16)
    var_20 = 'Test 4 passed: String representation'
    var_21 = print(var_20)
    var_22 = 'Empty'
    var_23 = module_0.immutable(name=var_22)
    var_24 = 'Test 5 passed: No members'
    var_25 = print(var_24)
    var_26 = module_0.immutable(var_15, var_16)
    var_27 = 3
    var_28 = 'Test 6 passed: Invalid attribute set'
    var_29 = print(var_28)
    var_30 = 'name, age, id_, ssn_'
    var_31 = 'Person'
    var_32 = module_0.immutable(var_30, var_31)
    var_33 = 'Alice'
    var_34 = 30
    var_35 = 123
    var_36 = 456
    var_37 = 124
    var_38 = 457
    var_39 = 'Test 7 passed: Multiple frozen members'
    var_40 = print(var_39)
    var_41 = module_0.immutable(var_37, var_38)
    var_42 = 'Test 8 passed: Empty set call'
    var_43 = print(var_42)
    var_44 = module_0.immutable(var_37, var_38)
    var_45 = 3
    var_46 = 4
    var_47 = 'Test 9 passed: Mixed valid and invalid attribute set'
    var_48 = print(var_47)
    var_49 = 'x1, y2, z_3'
    var_50 = 'Complex'
    var_51 = module_0.immutable(var_49, var_50)
    var_52 = 4
    var_53 = 'Test 10 passed: Complex member names'
    var_54 = print(var_53)
    var_55 = 'All tests passed!'
    var_56 = print(var_55)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = 'PointWithId'
    var_8 = module_0.immutable(var_6, var_7)
    var_9 = 17
    var_10 = 18
    var_11 = tuple()
    var_12 = -1
    var_13 = 2
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'Test 1 passed: Basic functionality'
    var_7 = print(var_6)
    var_8 = 'x, y, id_'
    var_9 = module_0.immutable(var_8, var_1)
    var_10 = 17
    var_11 = 18
    var_12 = 'Test 2 passed: Frozen members'
    var_13 = print(var_12)
    var_14 = tuple()
    var_15 = -1
    var_16 = 2
    var_17 = 'Test 3 passed: Inheritance and validation'
    var_18 = print(var_17)
    var_19 = module_0.immutable(var_15, var_16)
    var_20 = 3
    var_21 = 'Test 4 passed: Non-existent attribute'
    var_22 = print(var_21)
    var_23 = module_0.immutable(var_20, var_16)
    var_24 = 'Test 5 passed: No mutation when no kwargs'
    var_25 = print(var_24)
    var_26 = module_0.immutable(var_20, var_16)
    var_27 = 'Test 6 passed: String representation'
    var_28 = print(var_27)
    var_29 = 'x, y, id_, version_'
    var_30 = module_0.immutable(var_29, var_16)
    var_31 = 18
    var_32 = 2
    var_33 = 'Test 7 passed: Multiple frozen members'
    var_34 = print(var_33)
    var_35 = module_0.immutable(var_8, var_32)
    var_36 = 4
    var_37 = 'Test 8 passed: Mixed updates with frozen members'
    var_38 = print(var_37)
    var_39 = ''
    var_40 = 'Empty'
    var_41 = module_0.immutable(var_39, var_40)
    var_42 = 'Test 9 passed: Empty members'
    var_43 = print(var_42)
    var_44 = False
    var_45 = module_0.immutable(var_31, var_32, var_44)
    var_46 = 'Test 10 passed: Verbose mode'
    var_47 = print(var_46)
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'Basic functionality test passed.'
    var_7 = print(var_6)
    var_8 = 'x, y, id_'
    var_9 = 'PointWithId'
    var_10 = module_0.immutable(var_8, var_9)
    var_11 = 17
    var_12 = 18
    var_13 = 'ERROR: Should have raised AttributeError for frozen member'
    var_14 = print(var_13)
    var_15 = 5
    var_16 = 'ERROR: Should have raised AttributeError for invalid member'
    var_17 = print(var_16)
    var_18 = tuple()
    var_19 = -3
    var_20 = 'ERROR: Should have raised Exception for negative coordinate'
    var_21 = print(var_20)
    var_22 = 'All tests passed.'
    var_23 = print(var_22)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = 'PointWithId'
    var_8 = module_0.immutable(var_6, var_7)
    var_9 = 17
    var_10 = 18
    var_11 = tuple()
    var_12 = -1
    var_13 = 2
    var_14 = 3
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x, y, id_'
    var_7 = 'PointWithId'
    var_8 = module_0.immutable(var_6, var_7)
    var_9 = 17
    var_10 = 18
    var_11 = tuple()
    var_12 = -1
    var_13 = 2
    var_14 = 3
    var_15 = 'All tests passed.'
    var_16 = print(var_15)



