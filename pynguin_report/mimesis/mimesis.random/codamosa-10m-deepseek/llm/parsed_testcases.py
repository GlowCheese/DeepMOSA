####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8:]
    var_10 = '@@##'
    var_11 = var_0.generate_string_by_mask(var_10, var_2, var_3)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = 2
    var_14 = var_11[:var_13]
    var_15 = var_11[var_13:]
    var_16 = '####'
    var_17 = var_0.generate_string_by_mask(var_16, var_2, var_3)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = '@@@@'
    var_20 = var_0.generate_string_by_mask(var_19, var_2, var_3)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = '@@##'
    var_23 = '@'
    var_24 = var_0.generate_string_by_mask(var_22, var_23, var_23)
    var_25 = 'All tests passed.'
    var_26 = print(var_25)



# Parsed testcases at query #2
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 1
    var_10 = var_4[var_9:]



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
    var_7 = '@@##'
    var_8 = '@'
    var_9 = '#'
    var_10 = var_0.generate_string_by_mask(var_7, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 2
    var_13 = var_10[:var_12]
    var_14 = var_10[var_12:]
    var_15 = '@@##@@'
    var_16 = var_0.generate_string_by_mask(var_15, var_8, var_9)
    var_17 = len(var_16)
    assert var_17 == 6
    var_18 = var_16[:var_12]
    var_19 = 4
    var_20 = var_16[var_12:var_19]
    var_21 = var_16[var_19:]
    var_22 = '@@@@'
    var_23 = '@'
    var_24 = var_0.generate_string_by_mask(var_22, var_23, var_23)



# Parsed testcases at query #4
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    var_4 = len(var_1)
    var_5 = 0
    var_6 = var_2[var_5]
    var_7 = 1
    var_8 = var_2[var_7:]



# Parsed testcases at query #5
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Unit test for method generate_string_by_mask of class Random.'
    var_1 = module_0.Random()
    var_2 = '@###'
    var_3 = '@'
    var_4 = '#'
    var_5 = var_1.generate_string_by_mask(var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = len(var_2)
    var_8 = 0
    var_9 = var_5[var_8]
    var_10 = 1
    var_11 = var_5[var_10:]



# Parsed testcases at query #6
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 0.1
    var_5 = 0.3
    var_6 = 0.6
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = var_0.weighted_choice(var_7)



# Parsed testcases at query #7
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
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
    var_11 = var_0.randints(var_10)
    var_12 = -1
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #9
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = var_0.randints(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = 3
    var_5 = var_0.randints(var_4)
    var_6 = 10
    var_7 = 1
    var_8 = 100
    var_9 = var_0.randints(var_6, var_7, var_8)
    var_10 = 0
    var_11 = var_0.randints(var_10)



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
    var_7 = '@@##'
    var_8 = '@'
    var_9 = '#'
    var_10 = var_0.generate_string_by_mask(var_7, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 2
    var_13 = var_10[:var_12]
    var_14 = var_10[var_12:]
    var_15 = '####'
    var_16 = var_0.generate_string_by_mask(var_15, var_8, var_9)
    var_17 = len(var_16)
    assert var_17 == 4
    var_18 = '@@@@'
    var_19 = var_0.generate_string_by_mask(var_18, var_8, var_9)
    var_20 = len(var_19)
    assert var_20 == 4
    var_21 = '@@##'
    var_22 = '@'
    var_23 = var_0.generate_string_by_mask(var_21, var_22, var_22)
    var_24 = ''
    var_25 = '@'
    var_26 = '#'
    var_27 = var_0.generate_string_by_mask(var_24, var_25, var_26)



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
    var_1 = 16
    var_2 = var_0.randbytes(var_1)
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 32
    var_5 = var_0.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 32
    var_7 = 64
    var_8 = var_0.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 64



# Parsed testcases at query #13
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = 5
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = 1
    var_6 = str(var_4)
    var_7 = '.'
    var_8 = var_1.split(var_7)[var_5]
    var_9 = len(var_8)



# Parsed testcases at query #14
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
    var_11 = var_0.randints(var_10)



# Parsed testcases at query #15
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #16
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = '@@###'
    var_4 = var_0.generate_string_by_mask(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = '@@@###'
    var_7 = var_0.generate_string_by_mask(var_6)
    var_8 = len(var_7)
    assert var_8 == 6
    var_9 = '@@@###@@@'
    var_10 = var_0.generate_string_by_mask(var_9)
    var_11 = len(var_10)
    assert var_11 == 9
    var_12 = '@@@###@@@###'
    var_13 = var_0.generate_string_by_mask(var_12)
    var_14 = len(var_13)
    assert var_14 == 12
    var_15 = '@@@###@@@###@@@'
    var_16 = var_0.generate_string_by_mask(var_15)
    var_17 = len(var_16)
    assert var_17 == 15
    var_18 = '@@@###@@@###@@@###'
    var_19 = var_0.generate_string_by_mask(var_18)
    var_20 = len(var_19)
    assert var_20 == 18
    var_21 = '@@@###@@@###@@@###@@@'
    var_22 = var_0.generate_string_by_mask(var_21)
    var_23 = len(var_22)
    assert var_23 == 21
    var_24 = '@@@###@@@###@@@###@@@###'
    var_25 = var_0.generate_string_by_mask(var_24)
    var_26 = len(var_25)
    assert var_26 == 24
    var_27 = '@@@###@@@###@@@###@@@###@@@'
    var_28 = var_0.generate_string_by_mask(var_27)
    var_29 = len(var_28)
    assert var_29 == 27
    var_30 = '@@@###@@@###@@@###@@@###@@@###'
    var_31 = var_0.generate_string_by_mask(var_30)
    var_32 = len(var_31)
    assert var_32 == 30
    var_33 = '@@@###@@@###@@@###@@@###@@@###@@@'
    var_34 = var_0.generate_string_by_mask(var_33)
    var_35 = len(var_34)
    assert var_35 == 33
    var_36 = '@@@###@@@###@@@###@@@###@@@###@@@###'
    var_37 = var_0.generate_string_by_mask(var_36)
    var_38 = len(var_37)
    assert var_38 == 36
    var_39 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_40 = var_0.generate_string_by_mask(var_39)
    var_41 = len(var_40)
    assert var_41 == 39
    var_42 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_43 = var_0.generate_string_by_mask(var_42)
    var_44 = len(var_43)
    assert var_44 == 42
    var_45 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_46 = var_0.generate_string_by_mask(var_45)
    var_47 = len(var_46)
    assert var_47 == 45
    var_48 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_49 = var_0.generate_string_by_mask(var_48)
    var_50 = len(var_49)
    assert var_50 == 48
    var_51 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_52 = var_0.generate_string_by_mask(var_51)
    var_53 = len(var_52)
    assert var_53 == 51
    var_54 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_55 = var_0.generate_string_by_mask(var_54)
    var_56 = len(var_55)
    assert var_56 == 54
    var_57 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_58 = var_0.generate_string_by_mask(var_57)
    var_59 = len(var_58)
    assert var_59 == 57
    var_60 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_61 = var_0.generate_string_by_mask(var_60)
    var_62 = len(var_61)
    assert var_62 == 60
    var_63 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_64 = var_0.generate_string_by_mask(var_63)
    var_65 = len(var_64)
    assert var_65 == 63
    var_66 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_67 = var_0.generate_string_by_mask(var_66)
    var_68 = len(var_67)
    assert var_68 == 66
    var_69 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_70 = var_0.generate_string_by_mask(var_69)
    var_71 = len(var_70)
    assert var_71 == 69
    var_72 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_73 = var_0.generate_string_by_mask(var_72)
    var_74 = len(var_73)
    assert var_74 == 72
    var_75 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_76 = var_0.generate_string_by_mask(var_75)
    var_77 = len(var_76)
    assert var_77 == 75
    var_78 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_79 = var_0.generate_string_by_mask(var_78)
    var_80 = len(var_79)
    assert var_80 == 78
    var_81 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_82 = var_0.generate_string_by_mask(var_81)
    var_83 = len(var_82)
    assert var_83 == 81
    var_84 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_85 = var_0.generate_string_by_mask(var_84)
    var_86 = len(var_85)
    assert var_86 == 84
    var_87 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_88 = var_0.generate_string_by_mask(var_87)
    var_89 = len(var_88)
    assert var_89 == 87
    var_90 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_91 = var_0.generate_string_by_mask(var_90)
    var_92 = len(var_91)
    assert var_92 == 90
    var_93 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_94 = var_0.generate_string_by_mask(var_93)
    var_95 = len(var_94)
    assert var_95 == 93
    var_96 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_97 = var_0.generate_string_by_mask(var_96)
    var_98 = len(var_97)
    assert var_98 == 96
    var_99 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_100 = var_0.generate_string_by_mask(var_99)
    var_101 = len(var_100)
    assert var_101 == 99
    var_102 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_103 = var_0.generate_string_by_mask(var_102)
    var_104 = len(var_103)
    assert var_104 == 102
    var_105 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_106 = var_0.generate_string_by_mask(var_105)
    var_107 = len(var_106)
    assert var_107 == 105
    var_108 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_109 = var_0.generate_string_by_mask(var_108)
    var_110 = len(var_109)
    assert var_110 == 108
    var_111 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_112 = var_0.generate_string_by_mask(var_111)
    var_113 = len(var_112)
    assert var_113 == 111
    var_114 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_115 = var_0.generate_string_by_mask(var_114)
    var_116 = len(var_115)
    assert var_116 == 114
    var_117 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_118 = var_0.generate_string_by_mask(var_117)
    var_119 = len(var_118)
    assert var_119 == 117
    var_120 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###'
    var_121 = var_0.generate_string_by_mask(var_120)
    var_122 = len(var_121)
    assert var_122 == 120
    var_123 = '@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@###@@@'
    var_124 = var_0.generate_string_by_mask(var_123)
    var_125 = len(var_124)
    assert var_125 == 123



# Parsed testcases at query #17
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test randints method of Random class.'
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = 10
    var_4 = 20
    var_5 = var_1.randints(var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = 0
    var_8 = var_1.randints(var_7)



# Parsed testcases at query #18
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = var_0.uniform(var_1, var_2)



# Parsed testcases at query #19
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = module_0.Random(var_1)
    var_4 = 43
    var_5 = module_0.Random(var_4)



# Parsed testcases at query #20
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #21
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
    var_7 = '@@##'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = 2
    var_11 = var_8[:var_10]
    var_12 = var_8[var_10:]
    var_13 = 'A'
    var_14 = var_0.generate_string_by_mask(var_7, var_13)
    var_15 = len(var_14)
    assert var_15 == 4
    var_16 = var_14[:var_10]
    var_17 = var_14[var_10:]
    var_18 = 'D'
    var_19 = var_0.generate_string_by_mask(var_7, digit=var_18)
    var_20 = len(var_19)
    assert var_20 == 4
    var_21 = var_19[:var_10]
    var_22 = var_19[var_10:]
    var_23 = '@@##'
    var_24 = '#'
    var_25 = var_0.generate_string_by_mask(var_23, var_24, var_24)



# Parsed testcases at query #22
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 'cherry'
    var_5 = 0.5
    var_6 = 0.3
    var_7 = 0.2
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.weighted_choice(var_8)
    assert var_9 == 'apple'
    var_10 = {}
    var_11 = var_1.weighted_choice(var_10)
    var_12 = 0.1
    var_13 = 0.8
    var_14 = {var_10: var_12, var_11: var_12, var_4: var_13}
    var_15 = var_1.weighted_choice(var_14)
    assert var_15 == 'cherry'
    var_16 = 1.0
    var_17 = {var_10: var_16}
    var_18 = var_1.weighted_choice(var_17)
    assert var_18 == 'apple'



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)



# Parsed testcases at query #25
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 16
    var_2 = var_0.randbytes(var_1)
    var_3 = len(var_2)



# Parsed testcases at query #26
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #27
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #28
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Unit test for constructor of class Random.'
    var_1 = module_0.Random()



# Parsed testcases at query #29
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Unit test for method randbytes of class Random.'
    var_1 = module_0.Random()
    var_2 = var_1.randbytes()
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 32
    var_5 = var_1.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 32
    var_7 = var_1.randbytes()



# Parsed testcases at query #30
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 5
    var_4 = 10
    var_5 = 20
    var_6 = var_0.randints(var_3, var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = 0
    var_9 = var_0.randints(var_8)
    var_10 = -1
    var_11 = var_0.randints(var_10)



# Parsed testcases at query #31
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 10.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = 2
    var_5 = var_0.uniform(var_1, var_2, var_4)
    var_6 = str(var_5)
    var_7 = '.'
    var_8 = var_3.split(var_7)[var_1]
    var_9 = len(var_8)



# Parsed testcases at query #32
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 5
    var_4 = 10
    var_5 = 20
    var_6 = var_0.randints(var_3, var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = 0
    var_9 = var_0.randints(var_8)
    var_10 = -1
    var_11 = var_0.randints(var_10)



# Parsed testcases at query #33
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10
    var_3 = 20
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)



# Parsed testcases at query #34
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Unit test for the Random class.'
    var_1 = module_0.Random()



# Parsed testcases at query #35
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = module_0.Random()
    var_4 = module_0.Random()
    var_5 = 42
    var_6 = module_0.Random(var_5)



# Parsed testcases at query #36
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10
    var_3 = 20
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5



# Parsed testcases at query #37
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #38
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'apple'
    var_2 = 'banana'
    var_3 = 'cherry'
    var_4 = 0.2
    var_5 = 0.3
    var_6 = 0.5
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = var_0.weighted_choice(var_7)
    var_9 = {}
    var_10 = var_0.weighted_choice(var_9)
    var_11 = 0
    var_12 = {var_9: var_11, var_10: var_11, var_3: var_11}
    var_13 = var_0.weighted_choice(var_12)
    var_14 = {var_13: var_4, var_10: var_5, var_3: var_6}
    var_15 = var_0.weighted_choice(var_14)
    var_16 = 1.0
    var_17 = {var_13: var_16}
    var_18 = var_0.weighted_choice(var_17)
    assert var_18 == 'apple'



# Parsed testcases at query #39
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #40
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 10.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = 2
    var_5 = var_0.uniform(var_1, var_2, var_4)
    var_6 = str(var_5)
    var_7 = '.'
    var_8 = var_3.split(var_7)[var_1]
    var_9 = len(var_8)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test the generate_string_by_mask method of the Random class.'
    var_1 = module_0.Random()
    var_2 = '@###'
    var_3 = var_1.generate_string_by_mask(var_2)
    var_4 = len(var_3)
    var_5 = len(var_2)
    var_6 = 0
    var_7 = var_3[var_6]
    var_8 = 1
    var_9 = var_3[var_8:]
    var_10 = 'A##B'
    var_11 = 'A'
    var_12 = '#'
    var_13 = var_1.generate_string_by_mask(var_10, var_11, var_12)
    var_14 = len(var_13)
    var_15 = len(var_10)
    var_16 = var_13[var_6]
    var_17 = 3
    var_18 = var_13[var_8:var_17]
    var_19 = var_13[var_17]
    var_20 = '###'
    var_21 = '@'
    var_22 = var_1.generate_string_by_mask(var_20, var_21, var_12)
    var_23 = len(var_22)
    var_24 = len(var_20)
    var_25 = '@@@'
    var_26 = var_1.generate_string_by_mask(var_25, var_21, var_12)
    var_27 = len(var_26)
    var_28 = len(var_25)
    var_29 = '@'
    var_30 = var_1.generate_string_by_mask(var_25, var_29, var_29)



# Parsed testcases at query #2
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 1
    var_10 = var_4[var_9:]
    var_11 = '###@'
    var_12 = var_0.generate_string_by_mask(var_11, var_2, var_3)
    var_13 = len(var_12)
    var_14 = len(var_11)
    var_15 = 3
    var_16 = var_12[:var_15]
    var_17 = var_12[var_15]
    var_18 = '@@@@'
    var_19 = var_0.generate_string_by_mask(var_18, var_2, var_3)
    var_20 = len(var_19)
    var_21 = len(var_18)
    var_22 = '####'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    var_25 = len(var_22)
    var_26 = '#@#@'
    var_27 = var_0.generate_string_by_mask(var_26, var_2, var_3)
    var_28 = len(var_27)
    var_29 = len(var_26)
    var_30 = var_27[var_7]
    var_31 = var_27[var_9]
    var_32 = 2
    var_33 = var_27[var_32]
    var_34 = var_27[var_15]
    var_35 = '@#@#'
    var_36 = var_0.generate_string_by_mask(var_35, var_2, var_3)
    var_37 = len(var_36)
    var_38 = len(var_35)
    var_39 = var_36[var_7]
    var_40 = var_36[var_9]
    var_41 = var_36[var_32]
    var_42 = var_36[var_15]



# Parsed testcases at query #3
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = 0
    var_8 = var_4[var_7]
    var_9 = 1
    var_10 = var_4[var_9:]



# Parsed testcases at query #4
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]
    var_12 = 3
    var_13 = var_4[var_12]
    var_14 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_15 = len(var_14)
    assert var_15 == 4
    var_16 = var_14[var_6]
    var_17 = var_14[var_8]
    var_18 = var_14[var_10]
    var_19 = var_14[var_12]
    var_20 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = var_20[var_6]
    var_23 = var_20[var_8]
    var_24 = var_20[var_10]
    var_25 = var_20[var_12]
    var_26 = '@@##'
    var_27 = '@'
    var_28 = var_0.generate_string_by_mask(var_26, var_27, var_27)



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
    var_7 = '@@##'
    var_8 = '@'
    var_9 = '#'
    var_10 = var_0.generate_string_by_mask(var_7, var_8, var_9)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 2
    var_13 = var_10[:var_12]
    var_14 = var_10[var_12:]
    var_15 = 'A###'
    var_16 = 'A'
    var_17 = var_0.generate_string_by_mask(var_15, var_16, var_9)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = var_17[var_5:]
    var_20 = '@@@@'
    var_21 = '@'
    var_22 = var_0.generate_string_by_mask(var_20, var_21, var_21)



# Parsed testcases at query #6
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 10
    var_2 = var_0.randbytes(var_1)
    var_3 = len(var_2)
    assert var_3 == 10
    var_4 = 5
    var_5 = var_0.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = 0
    var_8 = var_0.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #7
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
    var_11 = 1.0
    var_12 = {var_10: var_11}
    var_13 = var_0.weighted_choice(var_12)
    assert var_13 == 'a'
    var_14 = {var_10: var_4, var_2: var_4}
    var_15 = var_0.weighted_choice(var_14)



# Parsed testcases at query #8
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test method generate_string_by_mask of class Random.'
    var_1 = module_0.Random()
    var_2 = '@###'
    var_3 = var_1.generate_string_by_mask(var_2)
    var_4 = len(var_3)
    assert var_4 == 4
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 1
    var_8 = var_3[var_7:]



# Parsed testcases at query #9
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #10
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
    var_11 = var_0.randints(var_10)
    var_12 = -1
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #11
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
    var_11 = var_0.randints(var_10)
    var_12 = -1
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #12
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10
    var_3 = 20
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 0
    var_7 = 10
    var_8 = 20
    var_9 = var_0.randints(var_6, var_7, var_8)
    var_10 = -1
    var_11 = 10
    var_12 = 20
    var_13 = var_0.randints(var_10, var_11, var_12)



# Parsed testcases at query #13
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #14
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
    var_11 = var_0.randints(var_10)
    var_12 = -1
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #15
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
    var_6 = var_0.randbytes()



# Parsed testcases at query #16
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test for method randints of class Random.'
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = 1
    var_4 = 10
    var_5 = var_1.randints(var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = var_1.randints(var_3, var_3, var_4)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_1.randints(var_2, var_3, var_3)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 0
    var_12 = 1
    var_13 = 10
    var_14 = var_1.randints(var_11, var_12, var_13)
    var_15 = -1
    var_16 = 1
    var_17 = 10
    var_18 = var_1.randints(var_15, var_16, var_17)



# Parsed testcases at query #17
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test Random class.'
    var_1 = module_0.Random()



# Parsed testcases at query #18
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()
    var_4 = 1
    var_5 = module_0.Random()
    var_6 = module_0.Random()



# Parsed testcases at query #21
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 10
    var_3 = 20
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5



# Parsed testcases at query #22
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()
    var_2 = len(var_1)
    assert var_2 == 16
    var_3 = 10
    var_4 = var_0.randbytes(var_3)
    var_5 = len(var_4)
    assert var_5 == 10
    var_6 = 0
    var_7 = var_0.randbytes(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = -5
    var_10 = var_0.randbytes(var_9)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 1000
    var_13 = var_0.randbytes(var_12)
    var_14 = len(var_13)
    assert var_14 == 1000
    var_15 = 1
    var_16 = var_0.randbytes(var_15)
    var_17 = len(var_16)
    assert var_17 == 1



# Parsed testcases at query #23
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #24
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test the randints method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randints()
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = 1
    var_5 = 100
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_1.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 0
    var_12 = var_1.randints(var_11)
    var_13 = -1
    var_14 = var_1.randints(var_13)



# Parsed testcases at query #25
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test method choice_enum_item of class Random.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = module_0.Random()



# Parsed testcases at query #26
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test the Random class.'
    var_1 = module_0.Random()



# Parsed testcases at query #27
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_0.Random()
    var_3 = 42.42
    var_4 = module_0.Random()
    var_5 = 'test'
    var_6 = module_0.Random()
    var_7 = None
    var_8 = module_0.Random()



# Parsed testcases at query #28
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 0
    var_7 = 1
    var_8 = 10
    var_9 = var_0.randints(var_6, var_7, var_8)
    var_10 = -5
    var_11 = 1
    var_12 = 10
    var_13 = var_0.randints(var_10, var_11, var_12)



# Parsed testcases at query #29
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)



# Parsed testcases at query #30
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #31
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
    var_11 = var_0.randints(var_10)
    var_12 = -1
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #32
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #33
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 5
    var_4 = 10
    var_5 = 20
    var_6 = var_0.randints(var_3, var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = 0
    var_9 = var_0.randints(var_8)
    var_10 = -1
    var_11 = var_0.randints(var_10)



# Parsed testcases at query #34
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = var_0.randints(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = var_0.randints(var_1)
    var_5 = 1
    var_6 = 100
    var_7 = 10
    var_8 = 20
    var_9 = var_0.randints(var_1, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = var_0.randints(var_1, var_7, var_8)
    var_12 = 0
    var_13 = var_0.randints(var_12)
    var_14 = -1
    var_15 = var_0.randints(var_14)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test method generate_string_by_mask of class Random.'
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = len(var_1)
    var_5 = 0
    var_6 = 1
    var_7 = 'A***'
    var_8 = 'A'
    var_9 = '*'
    var_10 = len(var_7)
    var_11 = '@@@@'
    var_12 = '@'
    var_13 = '@'
    var_14 = ''
    var_15 = '@'
    var_16 = '#'
    var_17 = 'ABC123'
    var_18 = '@'
    var_19 = '#'



# Parsed testcases at query #36
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Unit test for method randints of class Random.'
    var_1 = 42
    var_2 = module_0.Random(var_1)
    var_3 = 3
    var_4 = 1
    var_5 = 100
    var_6 = var_2.randints(var_3, var_4, var_5)
    var_7 = 5
    var_8 = 10
    var_9 = 20
    var_10 = var_2.randints(var_7, var_8, var_9)
    var_11 = 200
    var_12 = var_2.randints(var_4, var_5, var_11)
    var_13 = 0
    var_14 = 1
    var_15 = 100
    var_16 = var_2.randints(var_13, var_14, var_15)
    var_17 = -1
    var_18 = 1
    var_19 = 100
    var_20 = var_2.randints(var_17, var_18, var_19)



# Parsed testcases at query #37
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test the randints method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randints()
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = 1
    var_5 = 100
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_1.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 0
    var_12 = var_1.randints(var_11)
    var_13 = -1
    var_14 = var_1.randints(var_13)



# Parsed testcases at query #38
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test the weighted_choice method of the Random class.'
    var_1 = module_0.Random()
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 0.5
    var_6 = 0.3
    var_7 = 0.2
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = []
    var_10 = var_1.weighted_choice(var_8)
    var_11 = len(var_9)
    assert var_11 == 1000
    var_12 = set(var_9)



# Parsed testcases at query #39
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = var_0.uniform(var_1, var_2, var_2)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_2.split(var_6)[var_1]
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_0.uniform(var_1, var_1)



# Parsed testcases at query #40
#--------------------------


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = var_0.uniform(var_1, var_2)
    var_4 = 5
    var_5 = var_0.uniform(var_1, var_2, var_4)
    var_6 = str(var_5)
    var_7 = '.'
    var_8 = var_3.split(var_7)[var_1]
    var_9 = len(var_8)
    var_10 = -2.0
    var_11 = -1.0
    var_12 = var_0.uniform(var_10, var_11)
    var_13 = var_0.uniform(var_1, var_1)



