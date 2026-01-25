####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = var_0.generate_string_by_mask()
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6:]
    var_8 = 'A###B##C#'
    var_9 = var_0.generate_string_by_mask(var_8)
    var_10 = len(var_9)
    assert var_10 == 9
    var_11 = var_9[var_4]
    var_12 = 4
    var_13 = var_9[var_6:var_12]
    var_14 = var_9[var_12]
    var_15 = 5
    var_16 = 7
    var_17 = var_9[var_15:var_16]
    var_18 = var_9[var_16]
    var_19 = 8
    var_20 = var_9[var_19]
    var_21 = 'X###Y##Z#'
    var_22 = 'X'
    var_23 = 'Y'
    var_24 = var_0.generate_string_by_mask(var_21, var_22, var_23)
    var_25 = len(var_24)
    assert var_25 == 9
    var_26 = var_24[var_4]
    var_27 = var_24[var_6:var_12]
    var_28 = var_24[var_12]
    var_29 = var_24[var_15:var_16]
    var_30 = var_24[var_16]
    var_31 = var_24[var_19]
    var_32 = '@###'
    var_33 = '@'
    var_34 = var_0.generate_string_by_mask(var_32, var_33, var_33)
    var_35 = 'A#B#C#'
    var_36 = 'A'
    var_37 = '#'
    var_38 = var_0.generate_string_by_mask(var_35, var_36, var_37)
    var_39 = len(var_38)
    assert var_39 == 6
    var_40 = var_38[var_4]
    var_41 = var_38[var_6]
    var_42 = 2
    var_43 = var_38[var_42]
    var_44 = 3
    var_45 = var_38[var_44]
    var_46 = var_38[var_12]
    var_47 = var_38[var_15]
    var_48 = 'A#-B#-C#'
    var_49 = var_0.generate_string_by_mask(var_48, var_36, var_37)
    var_50 = len(var_49)
    assert var_50 == 8
    var_51 = var_49[var_4]
    var_52 = var_49[var_6]
    var_53 = var_49[var_44]
    var_54 = var_49[var_12]
    var_55 = 6
    var_56 = var_49[var_55]
    var_57 = var_49[var_16]



# Parsed testcases at query #2
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 7
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = var_8[var_14:]
    var_16 = 'X##Y##'
    var_17 = 'X'
    var_18 = 'Y'
    var_19 = var_0.generate_string_by_mask(var_16, var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 6
    var_21 = var_19[var_3]
    var_22 = 3
    var_23 = var_19[var_5:var_22]
    var_24 = var_19[var_22]
    var_25 = var_19[var_11:]
    var_26 = '@##'
    var_27 = '@'
    var_28 = var_0.generate_string_by_mask(var_26, var_27, var_27)
    var_29 = 'A#B#C#'
    var_30 = 'A'
    var_31 = 'B'
    var_32 = var_0.generate_string_by_mask(var_29, var_30, var_31)
    var_33 = len(var_32)
    assert var_33 == 6
    var_34 = var_32[var_27]
    var_35 = var_32[var_5]
    var_36 = 2
    var_37 = var_32[var_36]
    var_38 = var_32[var_22]
    var_39 = var_32[var_11]
    var_40 = 'X#Y#Z#'
    var_41 = var_0.generate_string_by_mask(var_40, var_17, var_18)
    var_42 = len(var_41)
    assert var_42 == 6
    var_43 = var_41[var_27]
    var_44 = var_41[var_5]
    var_45 = var_41[var_36]
    var_46 = var_41[var_22]



# Parsed testcases at query #3
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@@##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 6
    var_10 = 2
    var_11 = var_8[:var_10]
    var_12 = 4
    var_13 = var_8[var_10:var_12]
    var_14 = var_8[var_12:]
    var_15 = 'A!B!C!'
    var_16 = 'A'
    var_17 = '!'
    var_18 = var_0.generate_string_by_mask(var_15, var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 6
    var_20 = var_18[var_3]
    var_21 = var_18[var_5]
    var_22 = var_18[var_10]
    var_23 = 3
    var_24 = var_18[var_23]
    var_25 = var_18[var_12]
    var_26 = 5
    var_27 = var_18[var_26]
    var_28 = '@##'
    var_29 = '@'
    var_30 = var_0.generate_string_by_mask(var_28, var_29, var_29)
    var_31 = 'X#Y#Z#'
    var_32 = 'X'
    var_33 = '#'
    var_34 = var_0.generate_string_by_mask(var_31, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 6
    var_36 = var_34[var_5]
    var_37 = var_34[var_23]
    var_38 = var_34[var_26]



# Parsed testcases at query #4
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##C'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 8
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = 7
    var_16 = var_8[var_14:var_15]
    var_17 = var_8[var_15]
    var_18 = 'X@X#X'
    var_19 = 'X'
    var_20 = 'Y'
    var_21 = var_0.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 5
    var_23 = var_21[var_3]
    var_24 = var_21[var_5]
    var_25 = 2
    var_26 = var_21[var_25]
    var_27 = 3
    var_28 = var_21[var_27]
    var_29 = var_21[var_11]
    var_30 = '#'
    var_31 = var_0.generate_string_by_mask(char=var_30, digit=var_30)
    var_32 = 42
    var_33 = '@###'
    var_34 = var_0.generate_string_by_mask(var_33)
    var_35 = var_0.generate_string_by_mask(var_33)



# Parsed testcases at query #5
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##C'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 8
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = 7
    var_16 = var_8[var_14:var_15]
    var_17 = var_8[var_15]
    var_18 = 'X###Y##Z'
    var_19 = 'X'
    var_20 = 'Y'
    var_21 = var_0.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 8
    var_23 = var_21[var_3]
    var_24 = var_21[var_5:var_11]
    var_25 = var_21[var_11]
    var_26 = var_21[var_14:var_15]
    var_27 = var_21[var_15]
    var_28 = '@###'
    var_29 = '@'
    var_30 = var_0.generate_string_by_mask(var_28, var_29, var_29)
    var_31 = ''
    var_32 = var_0.generate_string_by_mask(var_31)
    assert var_32 == ''
    var_33 = 'ABC123'
    var_34 = var_0.generate_string_by_mask(var_33)
    assert var_34 == 'ABC123'



# Parsed testcases at query #6
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42
    var_5 = set()
    var_6 = len(var_5)



# Parsed testcases at query #7
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 2
    var_11 = var_8[var_3:var_10]
    var_12 = var_8[var_10]
    var_13 = 3
    var_14 = var_8[var_13:]
    var_15 = 'A1A1'
    var_16 = 'A'
    var_17 = '1'
    var_18 = var_0.generate_string_by_mask(var_15, var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = var_18[var_3]
    var_21 = var_18[var_5]
    var_22 = var_18[var_10]
    var_23 = var_18[var_13]
    var_24 = 'X#X#'
    var_25 = 'X'
    var_26 = '#'
    var_27 = var_0.generate_string_by_mask(var_24, var_25, var_26)
    var_28 = len(var_27)
    assert var_28 == 4
    var_29 = var_27[var_3]
    var_30 = var_27[var_5]
    var_31 = var_27[var_10]
    var_32 = var_27[var_13]
    var_33 = '@@@'
    var_34 = '@'
    var_35 = var_0.generate_string_by_mask(var_33, var_34, var_34)
    var_36 = ''
    var_37 = var_0.generate_string_by_mask(var_36)
    assert var_37 == ''
    var_38 = 'A#-B#'
    var_39 = var_0.generate_string_by_mask(var_38)
    var_40 = len(var_39)
    assert var_40 == 5
    var_41 = var_39[var_34]
    var_42 = var_39[var_5]
    var_43 = var_39[var_13]
    var_44 = 4
    var_45 = var_39[var_44]



# Parsed testcases at query #8
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = -5.0
    var_5 = -1.0
    var_6 = var_0.uniform(var_4, var_5)
    var_7 = 0.0
    var_8 = var_0.uniform(var_7, var_1, var_2)
    var_9 = str(var_8)
    var_10 = '.'
    var_11 = var_5.split(var_10)[var_1]
    var_12 = len(var_11)
    var_13 = var_0.uniform(var_7, var_7)
    var_14 = 10.0
    var_15 = var_0.uniform(var_14, var_14)
    var_16 = 10000000000.0
    var_17 = 100000000000.0
    var_18 = var_0.uniform(var_16, var_17)



# Parsed testcases at query #9
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = len(var_1)
    var_4 = 10
    var_5 = var_0.randbytes(var_4)
    var_6 = var_0.randbytes(var_4)



# Parsed testcases at query #10
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@@'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = 2
    var_11 = var_8[:var_10]
    var_12 = var_8[var_10:]
    var_13 = 'A#B#'
    var_14 = 'A'
    var_15 = 'B'
    var_16 = var_0.generate_string_by_mask(var_13, var_14, var_15)
    var_17 = len(var_16)
    assert var_17 == 4
    var_18 = var_16[var_3]
    var_19 = var_16[var_5]
    var_20 = var_16[var_10]
    var_21 = 3
    var_22 = var_16[var_21]
    var_23 = '@##'
    var_24 = '@'
    var_25 = var_0.generate_string_by_mask(var_23, var_24, var_24)
    var_26 = 'X#Y#'
    var_27 = 'X'
    var_28 = 'Y'
    var_29 = var_0.generate_string_by_mask(var_26, var_27, var_28)
    var_30 = len(var_29)
    assert var_30 == 4
    var_31 = var_29[var_24]
    var_32 = var_29[var_5]
    var_33 = var_29[var_10]
    var_34 = var_29[var_21]



# Parsed testcases at query #11
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #12
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand2.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random(var_5)
    var_7 = 999
    var_8 = module_0.Random()



# Parsed testcases at query #13
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 100
    var_4 = module_0.Random(var_3)
    var_5 = 200
    var_6 = module_0.Random(var_5)
    var_7 = 300
    var_8 = module_0.Random(var_7)
    var_9 = module_0.Random(var_7)



# Parsed testcases at query #14
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A#B#C#'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 6
    var_10 = var_8[var_3]
    var_11 = var_8[var_5]
    var_12 = 2
    var_13 = var_8[var_12]
    var_14 = 3
    var_15 = var_8[var_14]
    var_16 = 4
    var_17 = var_8[var_16]
    var_18 = 5
    var_19 = var_8[var_18]
    var_20 = 'X@Y@Z@'
    var_21 = '@'
    var_22 = 'X'
    var_23 = var_0.generate_string_by_mask(var_20, var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 6
    var_25 = var_23[var_3]
    var_26 = var_23[var_5]
    var_27 = var_23[var_12]
    var_28 = var_23[var_14]
    var_29 = var_23[var_16]
    var_30 = var_23[var_18]
    var_31 = '@##'
    var_32 = '@'
    var_33 = var_0.generate_string_by_mask(var_31, var_32, var_32)
    var_34 = 42
    var_35 = module_0.Random(var_34)
    var_36 = '@###'
    var_37 = var_35.generate_string_by_mask(var_36)
    var_38 = module_0.Random(var_34)
    var_39 = var_38.generate_string_by_mask(var_36)



# Parsed testcases at query #15
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42
    var_5 = 100
    var_6 = range(var_5)
    var_7 = 1



# Parsed testcases at query #16
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '###@'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = 3
    var_11 = var_8[:var_10]
    var_12 = var_8[var_10]
    var_13 = 'AAA111'
    var_14 = 'A'
    var_15 = '1'
    var_16 = var_0.generate_string_by_mask(var_13, var_14, var_15)
    var_17 = len(var_16)
    assert var_17 == 6
    var_18 = var_16[:var_10]
    var_19 = var_16[var_10:]
    var_20 = '@#'
    var_21 = '@'
    var_22 = var_0.generate_string_by_mask(var_20, var_21, var_21)
    var_23 = 'X#X#'
    var_24 = 'X'
    var_25 = '#'
    var_26 = var_0.generate_string_by_mask(var_23, var_24, var_25)
    var_27 = len(var_26)
    assert var_27 == 4
    var_28 = var_26[var_21]
    var_29 = var_26[var_5]
    var_30 = 2
    var_31 = var_26[var_30]
    var_32 = var_26[var_10]



# Parsed testcases at query #17
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 100
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = var_0.randints()
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = -10
    var_9 = 10
    var_10 = var_0.randints(var_1, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = -10
    var_13 = 0
    var_14 = 50
    var_15 = var_0.randints(var_1, var_13, var_14)
    var_16 = len(var_15)
    assert var_16 == 5
    var_17 = 0
    var_18 = var_0.randints(var_17)
    var_19 = -1
    var_20 = var_0.randints(var_19)



# Parsed testcases at query #18
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = r2.getstate()[var_3][:var_3]
    var_5 = b'12345'
    var_6 = module_0.Random(var_5)
    var_7 = 100
    var_8 = module_0.Random(var_7)
    var_9 = module_0.Random(var_7)
    var_10 = 101
    var_11 = module_0.Random(var_10)



# Parsed testcases at query #19
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 100
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = var_0.randints()
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = -100
    var_9 = -1
    var_10 = var_0.randints(var_1, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = -100
    var_13 = -1
    var_14 = 0
    var_15 = var_0.randints(var_1, var_14, var_14)
    var_16 = len(var_15)
    assert var_16 == 5
    var_17 = 0
    var_18 = var_0.randints(var_17)
    var_19 = -1
    var_20 = var_0.randints(var_19)



# Parsed testcases at query #20
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = 2.0
    var_4 = var_0.uniform(var_2, var_3)
    var_5 = var_0.uniform(var_2, var_3, var_3)
    var_6 = str(var_5)
    var_7 = '.'
    var_8 = var_5.split(var_7)[var_2]
    var_9 = len(var_8)
    var_10 = 0.0
    var_11 = var_0.uniform(var_10, var_10)
    var_12 = -1.0
    var_13 = var_0.uniform(var_12, var_2)
    var_14 = 10000000000.0
    var_15 = 100000000000.0
    var_16 = var_0.uniform(var_14, var_15)
    var_17 = -100.0
    var_18 = -1.0
    var_19 = var_0.uniform(var_17, var_18)



# Parsed testcases at query #21
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 32
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 10
    var_7 = var_0.randbytes(var_6)
    var_8 = 0
    var_9 = var_0.randbytes(var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #22
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 0.5
    var_5 = 0.3
    var_6 = 0.2
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = var_0.weighted_choice(var_7)
    var_9 = {}
    var_10 = var_0.weighted_choice(var_9)
    var_11 = 'only'
    var_12 = 1.0
    var_13 = {var_11: var_12}
    var_14 = var_0.weighted_choice(var_13)
    assert var_14 == 'only'
    var_15 = 'x'
    var_16 = 'y'
    var_17 = 'z'
    var_18 = {var_15: var_12, var_16: var_12, var_17: var_12}
    var_19 = 1000
    var_20 = range(var_19)
    var_21 = [random.weighted_choice(var_18) for _ in var_20]



# Parsed testcases at query #23
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42



# Parsed testcases at query #24
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 8
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 8
    var_6 = 4
    var_7 = var_0.randbytes(var_6)
    var_8 = var_0.randbytes(var_6)
    var_9 = 1
    var_10 = var_0.randbytes(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 32
    var_13 = var_0.randbytes(var_12)
    var_14 = len(var_13)
    assert var_14 == 32



# Parsed testcases at query #25
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 8
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 8
    var_6 = 10
    var_7 = var_0.randbytes(var_6)
    var_8 = 0



# Parsed testcases at query #26
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_seeded.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random()
    var_7 = rand_global.getstate()[var_3][:var_3]



# Parsed testcases at query #27
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_with_seed.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random(var_5)
    var_7 = module_0.Random(var_5)
    var_8 = 10
    var_9 = 200
    var_10 = module_0.Random(var_9)



# Parsed testcases at query #28
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 0.5
    var_4 = 0.3
    var_5 = 0.2
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.Random()
    var_8 = var_7.weighted_choice(var_6)
    var_9 = 'only'
    var_10 = 1.0
    var_11 = {var_9: var_10}
    var_12 = var_7.weighted_choice(var_11)
    assert var_12 == 'only'
    var_13 = 'x'
    var_14 = 'y'
    var_15 = {var_13: var_3, var_14: var_3}
    var_16 = 100
    var_17 = range(var_16)
    var_18 = [random.weighted_choice(var_15) for _ in var_17]
    var_19 = {}
    var_20 = var_7.weighted_choice(var_19)



# Parsed testcases at query #29
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = 2.0
    var_4 = var_0.uniform(var_2, var_3)
    var_5 = -5.0
    var_6 = 5.0
    var_7 = var_0.uniform(var_5, var_6)
    var_8 = var_0.uniform(var_2, var_3, var_3)
    var_9 = str(var_8)
    var_10 = '.'
    var_11 = var_7.split(var_10)[var_2]
    var_12 = len(var_11)
    var_13 = 0.0
    var_14 = var_0.uniform(var_13, var_2)
    var_15 = 100.0
    var_16 = 101.0
    var_17 = var_0.uniform(var_15, var_16)
    var_18 = -10.0
    var_19 = -5.0
    var_20 = var_0.uniform(var_18, var_19)



# Parsed testcases at query #30
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42
    var_5 = set()
    var_6 = 1



# Parsed testcases at query #31
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 7
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = var_8[var_14:]
    var_16 = 'X###Y##'
    var_17 = 'X'
    var_18 = 'Y'
    var_19 = var_0.generate_string_by_mask(var_16, var_17, var_18)
    var_20 = len(var_19)
    assert var_20 == 7
    var_21 = var_19[var_3]
    var_22 = var_19[var_5:var_11]
    var_23 = var_19[var_11]
    var_24 = var_19[var_14:]
    var_25 = '@###'
    var_26 = '@'
    var_27 = var_0.generate_string_by_mask(var_25, var_26, var_26)
    var_28 = 'C###D##'
    var_29 = 'C'
    var_30 = 'D'
    var_31 = var_0.generate_string_by_mask(var_28, var_29, var_30)
    var_32 = len(var_31)
    assert var_32 == 7
    var_33 = var_31[var_26]
    var_34 = var_31[var_5:var_11]
    var_35 = var_31[var_11]
    var_36 = var_31[var_14:]



# Parsed testcases at query #32
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 100
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = var_0.randints()
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = -10
    var_9 = 10
    var_10 = var_0.randints(var_1, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = -10
    var_13 = 0
    var_14 = var_0.randints(var_13)
    var_15 = -1
    var_16 = var_0.randints(var_15)



# Parsed testcases at query #33
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 100
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = var_0.randints()
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = 3
    var_9 = -10
    var_10 = 10
    var_11 = var_0.randints(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = -10
    var_14 = 4
    var_15 = 0
    var_16 = 50
    var_17 = var_0.randints(var_14, var_15, var_16)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = 2
    var_20 = 1000
    var_21 = 9999
    var_22 = var_0.randints(var_19, var_20, var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 0
    var_25 = var_0.randints(var_24)
    var_26 = -1
    var_27 = var_0.randints(var_26)



# Parsed testcases at query #34
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 0.1
    var_5 = 0.2
    var_6 = 0.7
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = var_0.weighted_choice(var_7)
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 0.5
    var_12 = {var_9: var_11, var_10: var_11}
    var_13 = var_0.weighted_choice(var_12)
    var_14 = 'only'
    var_15 = 1.0
    var_16 = {var_14: var_15}
    var_17 = var_0.weighted_choice(var_16)
    assert var_17 == 'only'
    var_18 = {}
    var_19 = var_0.weighted_choice(var_18)
    var_20 = 0.0
    var_21 = {var_19: var_20, var_2: var_15}
    var_22 = var_0.weighted_choice(var_21)
    var_23 = 0.0001
    var_24 = 0.9999
    var_25 = {var_19: var_23, var_2: var_24}
    var_26 = var_0.weighted_choice(var_25)



# Parsed testcases at query #35
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 5
    var_3 = 1
    var_4 = 100
    var_5 = var_0.randints(var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = 3
    var_8 = 10
    var_9 = 20
    var_10 = var_0.randints(var_7, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_0.randints()
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = 50
    var_15 = 60
    var_16 = var_0.randints(var_3, var_14, var_15)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 0
    var_19 = var_0.randints(var_18)
    var_20 = -5
    var_21 = var_0.randints(var_20)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@#'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = 2
    var_11 = var_8[var_3:var_10]
    var_12 = var_8[var_10]
    var_13 = 3
    var_14 = var_8[var_13]
    var_15 = '????'
    var_16 = '?'
    var_17 = var_0.generate_string_by_mask(var_15, var_16, var_16)
    var_18 = '????'
    var_19 = '?'
    var_20 = var_0.generate_string_by_mask(var_18, var_19, var_19)
    var_21 = '@###@###@###'
    var_22 = var_0.generate_string_by_mask(var_21)
    var_23 = len(var_22)
    assert var_23 == 10
    var_24 = var_22[var_19]
    var_25 = 4
    var_26 = var_22[var_5:var_25]
    var_27 = var_22[var_25]
    var_28 = 5
    var_29 = 8
    var_30 = var_22[var_28:var_29]
    var_31 = var_22[var_29]
    var_32 = 9
    var_33 = var_22[var_32:]
    var_34 = 'X111X111'
    var_35 = 'X'
    var_36 = '1'
    var_37 = var_0.generate_string_by_mask(var_34, var_35, var_36)
    var_38 = len(var_37)
    assert var_38 == 8
    var_39 = var_37[var_19]
    var_40 = var_37[var_5:var_25]
    var_41 = var_37[var_25]
    var_42 = var_37[var_28:var_29]
    var_43 = 'A-###-B'
    var_44 = var_0.generate_string_by_mask(var_43)
    var_45 = len(var_44)
    assert var_45 == 7
    var_46 = var_44[var_19]
    var_47 = var_44[var_10:var_28]
    var_48 = 6
    var_49 = var_44[var_48]



# Parsed testcases at query #2
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##C#'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 9
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = 7
    var_16 = var_8[var_14:var_15]
    var_17 = var_8[var_15]
    var_18 = 8
    var_19 = var_8[var_18]
    var_20 = 'X1Y2Z3'
    var_21 = 'X'
    var_22 = '1'
    var_23 = var_0.generate_string_by_mask(var_20, var_21, var_22)
    var_24 = len(var_23)
    assert var_24 == 6
    var_25 = var_23[var_3]
    var_26 = var_23[var_5]
    var_27 = 2
    var_28 = var_23[var_27]
    var_29 = 3
    var_30 = var_23[var_29]
    var_31 = var_23[var_11]
    var_32 = var_23[var_14]
    var_33 = '#'
    var_34 = var_0.generate_string_by_mask(char=var_33, digit=var_33)
    var_35 = 'A#B#C#'
    var_36 = 'A'
    var_37 = 'B'
    var_38 = var_0.generate_string_by_mask(var_35, var_36, var_37)
    var_39 = len(var_38)
    assert var_39 == 6
    var_40 = var_38[var_34]
    var_41 = var_38[var_5]
    var_42 = var_38[var_27]
    var_43 = var_38[var_29]
    var_44 = var_38[var_11]
    var_45 = var_38[var_14]
    var_46 = 'A-#B-#C-#'
    var_47 = var_0.generate_string_by_mask(var_46, var_36, var_37)
    var_48 = len(var_47)
    assert var_48 == 9
    var_49 = var_47[var_34]
    var_50 = var_47[var_27]
    var_51 = var_47[var_11]
    var_52 = 6
    var_53 = var_47[var_52]



# Parsed testcases at query #3
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@@##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 6
    var_10 = 2
    var_11 = var_8[:var_10]
    var_12 = 4
    var_13 = var_8[var_10:var_12]
    var_14 = var_8[var_12:]
    var_15 = 'A1B2C3'
    var_16 = 'A'
    var_17 = '1'
    var_18 = var_0.generate_string_by_mask(var_15, var_16, var_17)
    var_19 = len(var_18)
    assert var_19 == 6
    var_20 = var_18[var_3]
    var_21 = var_18[var_5]
    var_22 = var_18[var_10]
    var_23 = 3
    var_24 = var_18[var_23]
    var_25 = var_18[var_12]
    var_26 = 5
    var_27 = var_18[var_26]
    var_28 = '@#@#'
    var_29 = '@'
    var_30 = var_0.generate_string_by_mask(var_28, var_29, var_29)
    var_31 = 'A#-B#'
    var_32 = var_0.generate_string_by_mask(var_31)
    var_33 = len(var_32)
    assert var_33 == 5
    var_34 = var_32[var_29]
    var_35 = var_32[var_5]
    var_36 = var_32[var_23]
    var_37 = var_32[var_12]



# Parsed testcases at query #4
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = 'A###B##C'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 8
    var_10 = var_8[var_3]
    var_11 = 4
    var_12 = var_8[var_5:var_11]
    var_13 = var_8[var_11]
    var_14 = 5
    var_15 = 7
    var_16 = var_8[var_14:var_15]
    var_17 = var_8[var_15]
    var_18 = 'X11Y22'
    var_19 = 'X'
    var_20 = '1'
    var_21 = var_0.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 6
    var_23 = var_21[var_3]
    var_24 = 3
    var_25 = var_21[var_5:var_24]
    var_26 = var_21[var_24]
    var_27 = 6
    var_28 = var_21[var_11:var_27]
    var_29 = '@##'
    var_30 = '@'
    var_31 = var_0.generate_string_by_mask(var_29, var_30, var_30)
    var_32 = '!@#'
    var_33 = '!'
    var_34 = '#'
    var_35 = var_0.generate_string_by_mask(var_32, var_33, var_34)
    var_36 = len(var_35)
    assert var_36 == 3
    var_37 = var_35[var_30]
    var_38 = var_35[var_5]
    var_39 = 2
    var_40 = var_35[var_39]



# Parsed testcases at query #5
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6:]
    var_8 = 'A1B2'
    var_9 = 'A'
    var_10 = '1'
    var_11 = var_0.generate_string_by_mask(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = var_11[var_4]
    var_14 = var_11[var_6]
    var_15 = 2
    var_16 = var_11[var_15]
    var_17 = 3
    var_18 = var_11[var_17]
    var_19 = '@@@@'
    var_20 = var_0.generate_string_by_mask(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = '####'
    var_23 = var_0.generate_string_by_mask(var_22)
    var_24 = len(var_23)
    assert var_24 == 4
    var_25 = 'A#B#C#'
    var_26 = '#'
    var_27 = var_0.generate_string_by_mask(var_25, var_9, var_26)
    var_28 = len(var_27)
    assert var_28 == 6
    var_29 = var_27[var_4]
    var_30 = var_27[var_6]
    var_31 = var_27[var_15]
    var_32 = var_27[var_17]
    var_33 = 4
    var_34 = var_27[var_33]
    var_35 = 5
    var_36 = var_27[var_35]
    var_37 = 'A#-B#'
    var_38 = var_0.generate_string_by_mask(var_37, var_9, var_26)
    var_39 = len(var_38)
    assert var_39 == 5
    var_40 = var_38[var_4]
    var_41 = var_38[var_6]
    var_42 = var_38[var_17]
    var_43 = var_38[var_33]
    var_44 = '@###'
    var_45 = '@'
    var_46 = var_0.generate_string_by_mask(var_44, var_45, var_45)
    var_47 = 42
    var_48 = module_0.Random(var_47)
    var_49 = var_48.generate_string_by_mask(var_44)
    var_50 = module_0.Random(var_47)
    var_51 = var_50.generate_string_by_mask(var_44)



# Parsed testcases at query #6
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6:]
    var_8 = 'A1B2'
    var_9 = 'A'
    var_10 = '1'
    var_11 = var_0.generate_string_by_mask(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = var_11[var_4]
    var_14 = var_11[var_6]
    var_15 = 2
    var_16 = var_11[var_15]
    var_17 = 3
    var_18 = var_11[var_17]
    var_19 = '@@@@'
    var_20 = var_0.generate_string_by_mask(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = '####'
    var_23 = var_0.generate_string_by_mask(var_22)
    var_24 = len(var_23)
    assert var_24 == 4
    var_25 = 'A@1#B'
    var_26 = var_0.generate_string_by_mask(var_25)
    var_27 = len(var_26)
    assert var_27 == 5
    var_28 = var_26[var_6]
    var_29 = var_26[var_17]
    var_30 = '@###'
    var_31 = '@'
    var_32 = var_0.generate_string_by_mask(var_30, var_31, var_31)
    var_33 = '@###'
    var_34 = '#'
    var_35 = var_0.generate_string_by_mask(var_33, var_34, var_34)



# Parsed testcases at query #7
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #8
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 10.0
    var_3 = -10.0
    var_4 = -1.0
    var_5 = 2.0
    var_6 = '.'
    var_7 = var_7.split(var_6)[var_1]
    var_8 = len(var_7)
    var_9 = 5.0



# Parsed testcases at query #9
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = 2.0
    var_4 = var_0.uniform(var_2, var_3)
    var_5 = var_0.uniform(var_2, var_3, var_3)
    var_6 = str(var_5)
    var_7 = '.'
    var_8 = var_5.split(var_7)[var_2]
    var_9 = len(var_8)
    var_10 = -2.0
    var_11 = -1.0
    var_12 = var_0.uniform(var_10, var_11)
    var_13 = 0.0
    var_14 = var_0.uniform(var_13, var_2)
    var_15 = 5.0
    var_16 = var_0.uniform(var_15, var_15)
    var_17 = 10000000000.0
    var_18 = 100000000000.0
    var_19 = var_0.uniform(var_17, var_18)
    var_20 = 1e-10
    var_21 = 1e-09
    var_22 = var_0.uniform(var_20, var_21)



# Parsed testcases at query #10
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 32
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 0
    var_7 = var_0.randbytes(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = 10
    var_10 = var_0.randbytes(var_9)
    var_11 = var_0.randbytes(var_9)



# Parsed testcases at query #11
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_seeded.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random(var_5)
    var_7 = module_0.Random(var_5)
    var_8 = 200
    var_9 = module_0.Random(var_8)
    var_10 = 5
    var_11 = var_0.randints(var_10)
    var_12 = var_0.randints(var_10)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = 'abc'
    var_15 = '@###'
    var_16 = var_0.generate_string_by_mask(var_15)
    var_17 = 2.0
    var_18 = var_0.uniform(var_3, var_17)
    var_19 = 10
    var_20 = var_0.randbytes(var_19)



# Parsed testcases at query #12
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 32
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 0
    var_7 = var_0.randbytes(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = 10
    var_10 = var_0.randbytes(var_9)
    var_11 = var_0.randbytes(var_9)



# Parsed testcases at query #13
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = var_0.randints()
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = 1
    var_5 = 100
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_0.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 0
    var_12 = var_0.randints(var_4, var_11, var_4)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 0
    var_15 = var_0.randints(var_14)
    var_16 = -1
    var_17 = var_0.randints(var_16)



# Parsed testcases at query #14
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '@#@#'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = var_8[var_3]
    var_11 = var_8[var_5]
    var_12 = 2
    var_13 = var_8[var_12]
    var_14 = 3
    var_15 = var_8[var_14]
    var_16 = 'A'
    var_17 = '9'
    var_18 = var_0.generate_string_by_mask(char=var_16, digit=var_17)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = var_18[var_3]
    var_21 = var_18[var_5:]
    var_22 = '@###@###'
    var_23 = var_0.generate_string_by_mask(var_22)
    var_24 = len(var_23)
    assert var_24 == 8
    var_25 = var_23[var_3]
    var_26 = 4
    var_27 = var_23[var_5:var_26]
    var_28 = var_23[var_26]
    var_29 = 5
    var_30 = var_23[var_29:]
    var_31 = '#'
    var_32 = var_0.generate_string_by_mask(char=var_31, digit=var_31)
    var_33 = 'X1X1'
    var_34 = 'X'
    var_35 = '1'
    var_36 = var_0.generate_string_by_mask(var_33, var_34, var_35)
    var_37 = len(var_36)
    assert var_37 == 4
    var_38 = var_36[var_32]
    var_39 = var_36[var_5]
    var_40 = var_36[var_12]
    var_41 = var_36[var_14]



# Parsed testcases at query #15
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42
    var_5 = 100
    var_6 = range(var_5)
    var_7 = 'only'



# Parsed testcases at query #16
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_seeded.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random(var_5)
    var_7 = 200
    var_8 = module_0.Random(var_7)
    var_9 = 0
    var_10 = var_0.randints()
    var_11 = 5
    var_12 = var_0.randints(var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = 'abc'
    var_15 = var_0.generate_string_by_mask()
    var_16 = 2.0
    var_17 = var_0.uniform(var_3, var_16)
    var_18 = var_0.randbytes()
    var_19 = 0.5
    var_20 = {var_3: var_19, var_16: var_19}
    var_21 = var_0.weighted_choice(var_20)



# Parsed testcases at query #17
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_seed.getstate()[var_3][:var_3]
    var_5 = None
    var_6 = module_0.Random(var_5)
    var_7 = 100
    var_8 = module_0.Random(var_7)
    var_9 = 200
    var_10 = module_0.Random(var_9)
    var_11 = 300
    var_12 = module_0.Random(var_11)
    var_13 = module_0.Random(var_11)



# Parsed testcases at query #18
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 8
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 8
    var_6 = 10
    var_7 = var_0.randbytes(var_6)
    var_8 = 0
    var_9 = var_0.randbytes(var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #19
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = rand_seeded.getstate()[var_3][:var_3]
    var_5 = 4
    var_6 = 'little'
    var_7 = seed.to_bytes(var_5, var_6)[:var_3]
    var_8 = 123
    var_9 = module_0.Random(var_8)
    var_10 = rand_another.getstate()[var_3][:var_3]
    var_11 = rand_seeded.getstate()[var_3][:var_3]



# Parsed testcases at query #20
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = module_0.Random(var_1)
    var_4 = module_0.Random(var_1)
    var_5 = 1
    var_6 = 100
    var_7 = var_1 + var_5
    var_8 = module_0.Random(var_7)
    var_9 = 123
    var_10 = module_0.Random()
    var_11 = 123
    var_12 = module_0.Random(var_11)



# Parsed testcases at query #21
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = -5.0
    var_5 = -1.0
    var_6 = var_0.uniform(var_4, var_5)
    var_7 = 0.0
    var_8 = var_0.uniform(var_7, var_1, var_2)
    var_9 = str(var_8)
    var_10 = '.'
    var_11 = var_5.split(var_10)[var_1]
    var_12 = len(var_11)
    var_13 = 5.0
    var_14 = var_0.uniform(var_13, var_13)
    var_15 = 10000000000.0
    var_16 = 100000000000.0
    var_17 = var_0.uniform(var_15, var_16)
    var_18 = 1e-10
    var_19 = 1e-09
    var_20 = var_0.uniform(var_18, var_19)



# Parsed testcases at query #22
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = 5
    var_6 = 10
    var_7 = 20
    var_8 = var_0.randints(var_5, var_6, var_7)
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 0
    var_11 = var_0.randints(var_3, var_10, var_3)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_0.randints(var_13)
    var_15 = -1
    var_16 = var_0.randints(var_15)



# Parsed testcases at query #23
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = module_0.Random(var_3)
    var_5 = 2
    var_6 = module_0.Random(var_5)
    var_7 = module_0.Random(var_3)
    var_8 = module_0.Random(var_3)
    var_9 = 'random'
    var_10 = hasattr(var_0, var_9)
    var_11 = 'randint'
    var_12 = hasattr(var_0, var_11)
    var_13 = 'choice'
    var_14 = hasattr(var_0, var_13)
    var_15 = 'shuffle'
    var_16 = hasattr(var_0, var_15)



# Parsed testcases at query #24
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = r_seeded.getstate()[var_3][:var_3]
    var_5 = None
    var_6 = module_0.Random(var_5)
    var_7 = r_none.getstate()[var_3][:var_3]
    var_8 = 100
    var_9 = module_0.Random(var_8)
    var_10 = module_0.Random(var_8)
    var_11 = 0
    var_12 = 200
    var_13 = module_0.Random(var_12)



# Parsed testcases at query #25
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = module_0.Random(var_1)
    var_4 = module_0.Random(var_1)
    var_5 = var_3.randints()
    var_6 = var_4.randints()
    var_7 = 1
    var_8 = var_1 + var_7
    var_9 = module_0.Random(var_8)
    var_10 = var_3.randints()
    var_11 = var_9.randints()



# Parsed testcases at query #26
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = 2.0
    var_4 = var_0.uniform(var_2, var_3)
    var_5 = -5.0
    var_6 = -1.0
    var_7 = var_0.uniform(var_5, var_6)
    var_8 = 0.0
    var_9 = var_0.uniform(var_8, var_2, var_3)
    var_10 = str(var_9)
    var_11 = '.'
    var_12 = var_8.split(var_11)[var_2]
    var_13 = len(var_12)
    var_14 = var_0.uniform(var_8, var_8)
    var_15 = 10.0
    var_16 = var_0.uniform(var_15, var_15)
    var_17 = 0.0
    var_18 = 100.0
    var_19 = var_0.uniform(var_17, var_18)



# Parsed testcases at query #27
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 32
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 10
    var_7 = var_0.randbytes(var_6)
    var_8 = var_0.randbytes(var_6)



# Parsed testcases at query #28
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 1.0
    var_3 = 2.0
    var_4 = var_0.uniform(var_2, var_3)
    var_5 = -5.0
    var_6 = 5.0
    var_7 = var_0.uniform(var_5, var_6)
    var_8 = var_0.uniform(var_2, var_3, var_3)
    var_9 = str(var_8)
    var_10 = '.'
    var_11 = var_7.split(var_10)[var_2]
    var_12 = len(var_11)
    var_13 = 0.0
    var_14 = var_0.uniform(var_13, var_13)
    var_15 = 10.0
    var_16 = var_0.uniform(var_15, var_15)
    var_17 = -10.0
    var_18 = -5.0
    var_19 = var_0.uniform(var_17, var_18)



# Parsed testcases at query #29
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = r2.getstate()[var_3][:var_3]
    var_5 = module_0.Random(var_1)
    var_6 = 100
    var_7 = var_1 + var_3
    var_8 = module_0.Random(var_7)
    var_9 = 100
    var_10 = module_0.Random()
    var_11 = module_0.Random()



# Parsed testcases at query #30
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 8
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 8
    var_6 = 0
    var_7 = var_0.randbytes(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = 4
    var_10 = var_0.randbytes(var_9)
    var_11 = var_0.randbytes(var_9)



# Parsed testcases at query #31
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = '##@##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 2
    var_11 = var_8[var_3:var_10]
    var_12 = var_8[var_10]
    var_13 = 3
    var_14 = var_8[var_13:]
    var_15 = '????'
    var_16 = '?'
    var_17 = var_0.generate_string_by_mask(var_15, var_16, var_16)
    var_18 = '????'
    var_19 = '?'
    var_20 = var_0.generate_string_by_mask(var_18, var_19, var_19)
    var_21 = '@#@#'
    var_22 = var_0.generate_string_by_mask(var_21)
    var_23 = len(var_22)
    assert var_23 == 4
    var_24 = var_22[var_19]
    var_25 = var_22[var_5]
    var_26 = var_22[var_10]
    var_27 = var_22[var_13]



# Parsed testcases at query #32
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 1
    var_4 = r2.getstate()[var_3][:var_3]
    var_5 = 100
    var_6 = module_0.Random(var_5)
    var_7 = 200
    var_8 = module_0.Random(var_7)
    var_9 = 123
    var_10 = module_0.Random()
    var_11 = r5.getstate()[var_3][:var_3]



# Parsed testcases at query #33
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 42
    var_5 = 100
    var_6 = range(var_5)
    var_7 = 'only'



# Parsed testcases at query #34
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #35
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 32
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = 0
    var_7 = var_0.randbytes(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = 10
    var_10 = var_0.randbytes(var_9)
    var_11 = var_0.randbytes(var_9)



