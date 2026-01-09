####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]
    var_12 = 3
    var_13 = var_4[var_12]
    var_14 = 4
    var_15 = var_4[var_14]
    var_16 = '@@@@'
    var_17 = var_0.generate_string_by_mask(var_16, var_2, var_3)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = '####'
    var_20 = var_0.generate_string_by_mask(var_19, var_2, var_3)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 'A@B#C'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    assert var_24 == 5
    var_25 = var_23[var_8]
    var_26 = var_23[var_12]
    var_27 = '@@##'
    var_28 = '@'
    var_29 = var_0.generate_string_by_mask(var_27, var_28, var_28)
    var_30 = ''
    var_31 = var_0.generate_string_by_mask(var_30, var_28, var_29)
    assert var_31 == ''
    var_32 = 'Hello'
    var_33 = var_0.generate_string_by_mask(var_32, var_28, var_29)
    assert var_33 == 'Hello'
    var_34 = '@@@'
    var_35 = var_0.generate_string_by_mask(var_34, var_28, var_29)
    var_36 = len(var_35)
    assert var_36 == 3
    var_37 = '###'
    var_38 = var_0.generate_string_by_mask(var_37, var_28, var_29)
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = '!@#$%'
    var_41 = var_0.generate_string_by_mask(var_40, var_28, var_29)
    var_42 = len(var_41)
    assert var_42 == 5
    var_43 = var_41[var_8]
    var_44 = var_41[var_10]
    var_45 = 'α@β#γ'
    var_46 = var_0.generate_string_by_mask(var_45, var_28, var_29)
    var_47 = len(var_46)
    assert var_47 == 5
    var_48 = var_46[var_8]
    var_49 = var_46[var_12]
    var_50 = ' @ # '
    var_51 = var_0.generate_string_by_mask(var_50, var_28, var_29)
    var_52 = len(var_51)
    assert var_52 == 5
    var_53 = var_51[var_8]
    var_54 = var_51[var_12]
    var_55 = '\n@\n#\n'
    var_56 = var_0.generate_string_by_mask(var_55, var_28, var_29)
    var_57 = len(var_56)
    assert var_57 == 5
    var_58 = var_56[var_8]
    var_59 = var_56[var_12]
    var_60 = '\t@\t#\t'
    var_61 = var_0.generate_string_by_mask(var_60, var_28, var_29)
    var_62 = len(var_61)
    assert var_62 == 5
    var_63 = var_61[var_8]
    var_64 = var_61[var_12]
    var_65 = '\r@\r#\r'
    var_66 = var_0.generate_string_by_mask(var_65, var_28, var_29)
    var_67 = len(var_66)
    assert var_67 == 5
    var_68 = var_66[var_8]
    var_69 = var_66[var_12]
    var_70 = '\\@\\#\\'
    var_71 = var_0.generate_string_by_mask(var_70, var_28, var_29)
    var_72 = len(var_71)
    assert var_72 == 5
    var_73 = var_71[var_8]
    var_74 = var_71[var_12]
    var_75 = '"@"#"'
    var_76 = var_0.generate_string_by_mask(var_75, var_28, var_29)
    var_77 = len(var_76)
    assert var_77 == 5
    var_78 = var_76[var_8]
    var_79 = var_76[var_12]
    var_80 = "'@'#'"
    var_81 = var_0.generate_string_by_mask(var_80, var_28, var_29)
    var_82 = len(var_81)
    assert var_82 == 5
    var_83 = var_81[var_8]
    var_84 = var_81[var_12]
    var_85 = '\x00@\x00#\x00'
    var_86 = var_0.generate_string_by_mask(var_85, var_28, var_29)
    var_87 = len(var_86)
    assert var_87 == 5
    var_88 = var_86[var_8]
    var_89 = var_86[var_12]
    var_90 = 'αβγ'
    var_91 = 'α'
    var_92 = 'β'
    var_93 = var_0.generate_string_by_mask(var_90, var_91, var_92)
    var_94 = len(var_93)
    assert var_94 == 3
    var_95 = var_93[var_6]
    var_96 = var_93[var_8]
    var_97 = 'All test cases passed!'
    var_98 = print(var_97)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@#'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = '@@###'
    var_12 = var_0.generate_string_by_mask(var_11, var_2, var_3)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_12[var_6]
    var_15 = var_12[var_6]
    var_16 = var_12[var_9]
    var_17 = var_12[var_9]
    var_18 = 2
    var_19 = var_12[var_18]
    var_20 = 3
    var_21 = var_12[var_20]
    var_22 = 4
    var_23 = var_12[var_22]
    var_24 = 'ABC@#'
    var_25 = var_0.generate_string_by_mask(var_24, var_2, var_3)
    var_26 = len(var_25)
    assert var_26 == 5
    var_27 = var_25[var_20]
    var_28 = var_25[var_20]
    var_29 = var_25[var_22]
    var_30 = '@@@'
    var_31 = var_0.generate_string_by_mask(var_30, var_2, var_3)
    var_32 = len(var_31)
    assert var_32 == 3
    var_33 = '###'
    var_34 = var_0.generate_string_by_mask(var_33, var_2, var_3)
    var_35 = len(var_34)
    assert var_35 == 3
    var_36 = ''
    var_37 = var_0.generate_string_by_mask(var_36, var_2, var_3)
    assert var_37 == ''
    var_38 = '@#'
    var_39 = '@'
    var_40 = var_0.generate_string_by_mask(var_38, var_39, var_39)
    var_41 = 'αβγ'
    var_42 = 'α'
    var_43 = 'β'
    var_44 = var_0.generate_string_by_mask(var_41, var_42, var_43)
    var_45 = len(var_44)
    assert var_45 == 3
    var_46 = 'All tests passed!'
    var_47 = print(var_46)



# Parsed testcases at query #3
#--------------------------



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
    var_14 = '@@@@'
    var_15 = var_0.generate_string_by_mask(var_14, var_2, var_3)
    var_16 = len(var_15)
    assert var_16 == 4
    var_17 = '####'
    var_18 = var_0.generate_string_by_mask(var_17, var_2, var_3)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = 'A@B#C'
    var_21 = var_0.generate_string_by_mask(var_20, var_2, var_3)
    var_22 = len(var_21)
    assert var_22 == 5
    var_23 = var_21[var_8]
    var_24 = var_21[var_12]
    var_25 = '@@##'
    var_26 = '@'
    var_27 = var_0.generate_string_by_mask(var_25, var_26, var_26)
    var_28 = ''
    var_29 = var_0.generate_string_by_mask(var_28, var_26, var_27)
    assert var_29 == ''
    var_30 = 'Hello'
    var_31 = var_0.generate_string_by_mask(var_30, var_26, var_27)
    assert var_31 == 'Hello'
    var_32 = 'AA11'
    var_33 = 'A'
    var_34 = '1'
    var_35 = var_0.generate_string_by_mask(var_32, var_33, var_34)
    var_36 = len(var_35)
    assert var_36 == 4
    var_37 = var_35[var_6]
    var_38 = var_35[var_8]
    var_39 = var_35[var_10]
    var_40 = var_35[var_12]
    var_41 = 'αβ12'
    var_42 = 'α'
    var_43 = var_0.generate_string_by_mask(var_41, var_42, var_34)
    var_44 = len(var_43)
    assert var_44 == 4
    var_45 = 'All tests passed!'
    var_46 = print(var_45)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = module_0.Random()
    var_8 = 'AA##'
    var_9 = 'A'
    var_10 = '#'
    var_11 = var_7.generate_string_by_mask(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = 2
    var_14 = var_11[var_3:var_13]
    var_15 = var_11[var_13:]
    var_16 = module_0.Random()
    var_17 = '@@@'
    var_18 = '@'
    var_19 = var_16.generate_string_by_mask(var_17, var_18, var_10)
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = module_0.Random()
    var_22 = '###'
    var_23 = var_21.generate_string_by_mask(var_22, var_18, var_10)
    var_24 = len(var_23)
    assert var_24 == 3
    var_25 = module_0.Random()
    var_26 = 'ABC@##XYZ'
    var_27 = var_25.generate_string_by_mask(var_26, var_18, var_10)
    var_28 = len(var_27)
    assert var_28 == 9
    var_29 = 3
    var_30 = var_27[var_29]
    var_31 = 4
    var_32 = 6
    var_33 = var_27[var_31:var_32]
    var_34 = module_0.Random()
    var_35 = '@@##'
    var_36 = '@'
    var_37 = var_34.generate_string_by_mask(var_35, var_36, var_36)
    var_38 = module_0.Random()
    var_39 = ''
    var_40 = var_38.generate_string_by_mask(var_39, var_18, var_10)
    assert var_40 == ''
    var_41 = module_0.Random()
    var_42 = 'FIXEDTEXT'
    var_43 = var_41.generate_string_by_mask(var_42, var_18, var_10)
    assert var_43 == 'FIXEDTEXT'
    var_44 = module_0.Random()
    var_45 = '🎉@##🎊'
    var_46 = var_44.generate_string_by_mask(var_45, var_18, var_10)
    var_47 = len(var_46)
    assert var_47 == 6
    var_48 = var_46[var_5]
    var_49 = var_46[var_13:var_31]
    var_50 = module_0.Random()
    var_51 = '@@@###'
    var_52 = var_50.generate_string_by_mask(var_51, var_18, var_10)
    var_53 = len(var_52)
    assert var_53 == 6
    var_54 = var_52[var_36:var_29]
    var_55 = var_52[var_29:]
    var_56 = 'All tests passed!'
    var_57 = print(var_56)



# Parsed testcases at query #5
#--------------------------



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
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9:]
    var_11 = '@@##'
    var_12 = var_0.generate_string_by_mask(var_11, var_2, var_3)
    var_13 = len(var_12)
    assert var_13 == 4
    var_14 = 2
    var_15 = var_12[:var_14]
    var_16 = var_12[var_14:]
    var_17 = 'A@B#C'
    var_18 = var_0.generate_string_by_mask(var_17, var_2, var_3)
    var_19 = len(var_18)
    assert var_19 == 5
    var_20 = var_18[var_9]
    var_21 = var_18[var_9]
    var_22 = 3
    var_23 = var_18[var_22]
    var_24 = '@###'
    var_25 = '@'
    var_26 = var_0.generate_string_by_mask(var_24, var_25, var_25)
    var_27 = ''
    var_28 = var_0.generate_string_by_mask(var_27, var_25, var_26)
    assert var_28 == ''
    var_29 = 'ABCD'
    var_30 = var_0.generate_string_by_mask(var_29, var_25, var_26)
    assert var_30 == 'ABCD'
    var_31 = 'a*b?'
    var_32 = 'a'
    var_33 = '?'
    var_34 = var_0.generate_string_by_mask(var_31, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 4
    var_36 = var_34[var_6]
    var_37 = var_34[var_6]
    var_38 = var_34[var_22]
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]
    var_12 = 3
    var_13 = var_4[var_12]
    var_14 = 4
    var_15 = var_4[var_14]
    var_16 = '@@@@'
    var_17 = var_0.generate_string_by_mask(var_16, var_2, var_3)
    var_18 = len(var_17)
    assert var_18 == 4
    var_19 = '####'
    var_20 = var_0.generate_string_by_mask(var_19, var_2, var_3)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 'A@B#C'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    assert var_24 == 5
    var_25 = var_23[var_8]
    var_26 = var_23[var_12]
    var_27 = '@@##'
    var_28 = '@'
    var_29 = var_0.generate_string_by_mask(var_27, var_28, var_28)
    var_30 = ''
    var_31 = var_0.generate_string_by_mask(var_30, var_28, var_29)
    assert var_31 == ''
    var_32 = 'Hello'
    var_33 = var_0.generate_string_by_mask(var_32, var_28, var_29)
    assert var_33 == 'Hello'
    var_34 = var_0.generate_string_by_mask(var_28, var_28, var_29)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = var_34[var_6]
    var_37 = var_0.generate_string_by_mask(var_29, var_28, var_29)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = var_37[var_6]
    var_40 = '@@@###'
    var_41 = var_0.generate_string_by_mask(var_40, var_28, var_29)
    var_42 = len(var_41)
    assert var_42 == 6
    var_43 = var_41[:var_12]
    var_44 = var_41[var_12:]
    var_45 = 'All test cases passed!'
    var_46 = print(var_45)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = module_0.Random()
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_5.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = module_0.Random()
    var_12 = 0
    var_13 = var_11.randints(var_12)
    var_14 = module_0.Random()
    var_15 = 'abc'
    var_16 = len(var_9)
    assert var_16 == 5
    var_17 = module_0.Random()
    var_18 = '@###'
    var_19 = '@'
    var_20 = '#'
    var_21 = var_17.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 4
    var_23 = 0
    var_24 = var_21[var_23]
    var_25 = var_21[var_3:]
    var_26 = module_0.Random()
    var_27 = '@@@'
    var_28 = '@'
    var_29 = var_26.generate_string_by_mask(var_27, var_28, var_28)
    var_30 = module_0.Random()
    var_31 = 2.0
    var_32 = var_30.uniform(var_3, var_31, var_31)
    var_33 = str(var_32)
    var_34 = '.'
    var_35 = var_29.split(var_34)[var_3]
    var_36 = len(var_35)
    var_37 = module_0.Random()
    var_38 = 8
    var_39 = var_37.randbytes(var_38)
    var_40 = len(var_39)
    assert var_40 == 8
    var_41 = module_0.Random()
    var_42 = 'a'
    var_43 = 'b'
    var_44 = 'c'
    var_45 = 0.5
    var_46 = 0.3
    var_47 = 0.2
    var_48 = {var_42: var_45, var_43: var_46, var_44: var_47}
    var_49 = var_41.weighted_choice(var_48)
    var_50 = module_0.Random()
    var_51 = {}
    var_52 = var_50.weighted_choice(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = module_0.Random()
    var_57 = module_0.Random()
    var_58 = 42
    var_59 = module_0.Random()
    var_60 = module_0.Random()
    var_61 = 123
    var_62 = module_0.Random()
    var_63 = 123
    var_64 = module_0.Random()
    var_65 = module_0.Random()
    var_66 = module_0.Random()
    var_67 = 'All test cases passed!'
    var_68 = print(var_67)



# Parsed testcases at query #9
#--------------------------



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
    var_14 = '@@@@'
    var_15 = var_0.generate_string_by_mask(var_14, var_2, var_3)
    var_16 = len(var_15)
    assert var_16 == 4
    var_17 = '####'
    var_18 = var_0.generate_string_by_mask(var_17, var_2, var_3)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = 'AB@12#'
    var_21 = var_0.generate_string_by_mask(var_20, var_2, var_3)
    var_22 = len(var_21)
    assert var_22 == 6
    var_23 = var_21[var_10]
    var_24 = 5
    var_25 = var_21[var_24]
    var_26 = '@@##'
    var_27 = '@'
    var_28 = var_0.generate_string_by_mask(var_26, var_27, var_27)
    var_29 = ''
    var_30 = var_0.generate_string_by_mask(var_29, var_27, var_28)
    assert var_30 == ''
    var_31 = 'Hello'
    var_32 = var_0.generate_string_by_mask(var_31, var_27, var_28)
    assert var_32 == 'Hello'
    var_33 = 'Привет@#'
    var_34 = var_0.generate_string_by_mask(var_33, var_27, var_28)
    var_35 = len(var_34)
    assert var_35 == 8
    var_36 = 'Привет'
    var_37 = 6
    var_38 = var_34[var_37]
    var_39 = 7
    var_40 = var_34[var_39]
    var_41 = 'All tests passed!'
    var_42 = print(var_41)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_4.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 2
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_12.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 5.5
    var_19 = 6.5
    var_20 = var_17.uniform(var_18, var_19, var_1)
    var_21 = str(var_20)
    var_22 = var_18.split(var_6)[var_2]
    var_23 = len(var_22)
    var_24 = module_0.Random()
    var_25 = var_24.uniform(var_1, var_1, var_11)
    var_26 = str(var_25)
    var_27 = var_22.split(var_6)[var_2]
    var_28 = len(var_27)
    var_29 = module_0.Random()
    var_30 = -100
    var_31 = -50
    var_32 = 5
    var_33 = var_29.uniform(var_30, var_31, var_32)
    var_34 = str(var_33)
    var_35 = var_29.split(var_6)[var_2]
    var_36 = len(var_35)
    var_37 = module_0.Random()
    var_38 = 1.23456789
    var_39 = 9.87654321
    var_40 = 8
    var_41 = var_37.uniform(var_38, var_39, var_40)
    var_42 = str(var_41)
    var_43 = var_36.split(var_6)[var_2]
    var_44 = len(var_43)
    var_45 = module_0.Random()
    var_46 = 0.0001
    var_47 = 20
    var_48 = var_45.uniform(var_1, var_46, var_47)
    var_49 = str(var_48)
    var_50 = var_42.split(var_6)[var_2]
    var_51 = len(var_50)
    var_52 = module_0.Random()
    var_53 = -1.5
    var_54 = 1.5
    var_55 = var_52.uniform(var_53, var_54, var_2)
    var_56 = str(var_55)
    var_57 = var_48.split(var_6)[var_2]
    var_58 = len(var_57)
    var_59 = module_0.Random()
    var_60 = 100
    var_61 = 200
    var_62 = var_59.uniform(var_60, var_61, var_1)
    var_63 = str(var_62)
    var_64 = var_54.split(var_6)[var_2]
    var_65 = len(var_64)
    var_66 = module_0.Random()
    var_67 = -0.001
    var_68 = 0.001
    var_69 = var_66.uniform(var_67, var_68, var_11)
    var_70 = str(var_69)
    var_71 = var_60.split(var_6)[var_2]
    var_72 = len(var_71)
    var_73 = module_0.Random()
    var_74 = var_73.uniform(var_1, var_1, var_1)
    var_75 = str(var_74)
    var_76 = var_64.split(var_6)[var_2]
    var_77 = len(var_76)
    var_78 = module_0.Random()
    var_79 = -10
    var_80 = -5
    var_81 = 3
    var_82 = var_78.uniform(var_79, var_80, var_81)
    var_83 = str(var_82)
    var_84 = var_71.split(var_6)[var_2]
    var_85 = len(var_84)
    var_86 = module_0.Random()
    var_87 = 0.123456789
    var_88 = 0.987654321
    var_89 = 6
    var_90 = var_86.uniform(var_87, var_88, var_89)
    var_91 = str(var_90)
    var_92 = var_78.split(var_6)[var_2]
    var_93 = len(var_92)
    var_94 = module_0.Random()
    var_95 = -1000
    var_96 = 1000
    var_97 = var_94.uniform(var_95, var_96, var_12)
    var_98 = str(var_97)
    var_99 = var_84.split(var_6)[var_2]
    var_100 = len(var_99)
    var_101 = module_0.Random()
    var_102 = 1e-06
    var_103 = 2e-06
    var_104 = var_101.uniform(var_102, var_103, var_11)
    var_105 = str(var_104)
    var_106 = var_90.split(var_6)[var_2]
    var_107 = len(var_106)
    var_108 = module_0.Random()
    var_109 = -1
    var_110 = var_108.uniform(var_109, var_2, var_1)
    var_111 = str(var_110)
    var_112 = var_95.split(var_6)[var_2]
    var_113 = len(var_112)
    var_114 = module_0.Random()
    var_115 = var_114.uniform(var_1, var_11, var_2)
    var_116 = str(var_115)
    var_117 = var_99.split(var_6)[var_2]
    var_118 = len(var_117)
    var_119 = module_0.Random()
    var_120 = -0.5
    var_121 = 0.5
    var_122 = var_119.uniform(var_120, var_121, var_32)
    var_123 = str(var_122)
    var_124 = var_105.split(var_6)[var_2]
    var_125 = len(var_124)
    var_126 = module_0.Random()
    var_127 = 1000000
    var_128 = 2000000
    var_129 = var_126.uniform(var_127, var_128, var_1)
    var_130 = str(var_129)
    var_131 = var_111.split(var_6)[var_2]
    var_132 = len(var_131)
    var_133 = module_0.Random()
    var_134 = -1e-07
    var_135 = 1e-07
    var_136 = var_133.uniform(var_134, var_135, var_3)
    var_137 = str(var_136)
    var_138 = var_117.split(var_6)[var_2]
    var_139 = len(var_138)
    var_140 = 'All test cases pass'
    var_141 = print(var_140)



# Parsed testcases at query #11
#--------------------------



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



# Parsed testcases at query #12
#--------------------------



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
    var_9 = {}
    var_10 = var_7.weighted_choice(var_9)
    var_11 = 'x'
    var_12 = 1.0
    var_13 = {var_11: var_12}
    var_14 = var_7.weighted_choice(var_13)
    assert var_14 == 'x'
    var_15 = 0.0
    var_16 = {var_10: var_15, var_1: var_12}
    var_17 = var_7.weighted_choice(var_16)
    assert var_17 == 'b'
    var_18 = -1.0
    var_19 = 2.0
    var_20 = {var_10: var_18, var_1: var_19}
    var_21 = var_7.weighted_choice(var_20)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@#'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = '@@###'
    var_12 = var_0.generate_string_by_mask(var_11, var_2, var_3)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_12[var_6]
    var_15 = var_12[var_6]
    var_16 = var_12[var_9]
    var_17 = var_12[var_9]
    var_18 = 2
    var_19 = var_12[var_18]
    var_20 = 3
    var_21 = var_12[var_20]
    var_22 = 4
    var_23 = var_12[var_22]
    var_24 = 'ABC@#'
    var_25 = var_0.generate_string_by_mask(var_24, var_2, var_3)
    var_26 = len(var_25)
    assert var_26 == 5
    var_27 = var_25[var_20]
    var_28 = var_25[var_20]
    var_29 = var_25[var_22]
    var_30 = '@@@'
    var_31 = var_0.generate_string_by_mask(var_30, var_2, var_3)
    var_32 = len(var_31)
    assert var_32 == 3
    var_33 = '###'
    var_34 = var_0.generate_string_by_mask(var_33, var_2, var_3)
    var_35 = len(var_34)
    assert var_35 == 3
    var_36 = ''
    var_37 = var_0.generate_string_by_mask(var_36, var_2, var_3)
    assert var_37 == ''
    var_38 = '@#'
    var_39 = '@'
    var_40 = var_0.generate_string_by_mask(var_38, var_39, var_39)
    var_41 = 'a1b2'
    var_42 = 'a'
    var_43 = '1'
    var_44 = var_0.generate_string_by_mask(var_41, var_42, var_43)
    var_45 = len(var_44)
    assert var_45 == 4
    var_46 = var_44[var_6]
    var_47 = var_44[var_6]
    var_48 = var_44[var_9]
    var_49 = 'All tests passed!'
    var_50 = print(var_49)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'Test the randbytes method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randbytes()
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 8
    var_5 = var_1.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 8
    var_7 = 0
    var_8 = var_1.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = 1
    var_11 = var_1.randbytes(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 32
    var_14 = var_1.randbytes(var_13)
    var_15 = len(var_14)
    assert var_15 == 32
    var_16 = 16
    var_17 = var_1.randbytes(var_16)
    var_18 = 'All tests passed for Random.randbytes()'
    var_19 = print(var_18)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'Test the randbytes method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randbytes()
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 0
    var_5 = var_1.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = 1
    var_8 = var_1.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 10
    var_11 = var_1.randbytes(var_10)
    var_12 = len(var_11)
    assert var_12 == 10
    var_13 = 100
    var_14 = var_1.randbytes(var_13)
    var_15 = len(var_14)
    assert var_15 == 100
    var_16 = [var_4]
    var_17 = var_16 * var_13
    var_18 = bytes(var_17)
    var_19 = var_1.randbytes(var_13)
    var_20 = 42
    var_21 = module_0.Random(var_20)
    var_22 = module_0.Random(var_20)
    var_23 = 20
    var_24 = var_21.randbytes(var_23)
    var_25 = var_22.randbytes(var_23)
    var_26 = module_0.Random(var_20)
    var_27 = 43
    var_28 = module_0.Random(var_27)
    var_29 = var_26.randbytes(var_23)
    var_30 = var_28.randbytes(var_23)
    var_31 = 5
    var_32 = len(var_19)
    assert var_32 == 5
    var_33 = 'All tests passed!'
    var_34 = print(var_33)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_3.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_10.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 100
    var_19 = 200
    var_20 = var_17.uniform(var_18, var_19, var_1)
    var_21 = module_0.Random()
    var_22 = 0.5
    var_23 = 0.6
    var_24 = var_21.uniform(var_22, var_23, var_11)
    var_25 = str(var_24)
    var_26 = var_18.split(var_6)[var_2]
    var_27 = len(var_26)
    var_28 = module_0.Random()
    var_29 = -1
    var_30 = 2
    var_31 = var_28.uniform(var_29, var_2, var_30)
    var_32 = str(var_31)
    var_33 = var_23.split(var_6)[var_2]
    var_34 = len(var_33)
    var_35 = module_0.Random()
    var_36 = var_35.uniform(var_1, var_1, var_3)
    assert var_36 == 0
    var_37 = module_0.Random()
    var_38 = var_37.uniform(var_11, var_11, var_12)
    assert var_38 == 10
    var_39 = module_0.Random()
    var_40 = -5
    var_41 = -5
    var_42 = var_39.uniform(var_40, var_41, var_1)
    assert var_42 == -5
    var_43 = module_0.Random()
    var_44 = var_43.uniform(var_1, var_2, var_1)
    var_45 = module_0.Random()
    var_46 = var_45.uniform(var_1, var_2, var_2)
    var_47 = str(var_46)
    var_48 = var_28.split(var_6)[var_2]
    var_49 = len(var_48)
    var_50 = module_0.Random()
    var_51 = var_50.uniform(var_1, var_2, var_30)
    var_52 = str(var_51)
    var_53 = var_31.split(var_6)[var_2]
    var_54 = len(var_53)
    var_55 = module_0.Random()
    var_56 = 3
    var_57 = var_55.uniform(var_1, var_2, var_56)
    var_58 = str(var_57)
    var_59 = var_35.split(var_6)[var_2]
    var_60 = len(var_59)
    var_61 = module_0.Random()
    var_62 = 4
    var_63 = var_61.uniform(var_1, var_2, var_62)
    var_64 = str(var_63)
    var_65 = var_39.split(var_6)[var_2]
    var_66 = len(var_65)
    var_67 = module_0.Random()
    var_68 = var_67.uniform(var_1, var_2, var_12)
    var_69 = str(var_68)
    var_70 = var_42.split(var_6)[var_2]
    var_71 = len(var_70)
    var_72 = module_0.Random()
    var_73 = 6
    var_74 = var_72.uniform(var_1, var_2, var_73)
    var_75 = str(var_74)
    var_76 = var_46.split(var_6)[var_2]
    var_77 = len(var_76)
    var_78 = module_0.Random()
    var_79 = 7
    var_80 = var_78.uniform(var_1, var_2, var_79)
    var_81 = str(var_80)
    var_82 = var_50.split(var_6)[var_2]
    var_83 = len(var_82)
    var_84 = module_0.Random()
    var_85 = 8
    var_86 = var_84.uniform(var_1, var_2, var_85)
    var_87 = str(var_86)
    var_88 = var_54.split(var_6)[var_2]
    var_89 = len(var_88)
    var_90 = module_0.Random()
    var_91 = 9
    var_92 = var_90.uniform(var_1, var_2, var_91)
    var_93 = str(var_92)
    var_94 = var_58.split(var_6)[var_2]
    var_95 = len(var_94)
    var_96 = module_0.Random()
    var_97 = var_96.uniform(var_1, var_2, var_11)
    var_98 = str(var_97)
    var_99 = var_61.split(var_6)[var_2]
    var_100 = len(var_99)
    var_101 = module_0.Random()
    var_102 = 11
    var_103 = var_101.uniform(var_1, var_2, var_102)
    var_104 = str(var_103)
    var_105 = var_65.split(var_6)[var_2]
    var_106 = len(var_105)
    var_107 = module_0.Random()
    var_108 = 12
    var_109 = var_107.uniform(var_1, var_2, var_108)
    var_110 = str(var_109)
    var_111 = var_69.split(var_6)[var_2]
    var_112 = len(var_111)
    var_113 = module_0.Random()
    var_114 = 13
    var_115 = var_113.uniform(var_1, var_2, var_114)
    var_116 = str(var_115)
    var_117 = var_73.split(var_6)[var_2]
    var_118 = len(var_117)
    var_119 = module_0.Random()
    var_120 = 14
    var_121 = var_119.uniform(var_1, var_2, var_120)
    var_122 = str(var_121)
    var_123 = var_77.split(var_6)[var_2]
    var_124 = len(var_123)
    var_125 = module_0.Random()
    var_126 = var_125.uniform(var_1, var_2, var_3)
    var_127 = str(var_126)
    var_128 = var_80.split(var_6)[var_2]
    var_129 = len(var_128)
    var_130 = module_0.Random()
    var_131 = 16
    var_132 = var_130.uniform(var_1, var_2, var_131)
    var_133 = str(var_132)
    var_134 = var_84.split(var_6)[var_2]
    var_135 = len(var_134)
    var_136 = module_0.Random()
    var_137 = 17
    var_138 = var_136.uniform(var_1, var_2, var_137)
    var_139 = str(var_138)
    var_140 = var_88.split(var_6)[var_2]
    var_141 = len(var_140)
    var_142 = module_0.Random()
    var_143 = 18
    var_144 = var_142.uniform(var_1, var_2, var_143)
    var_145 = str(var_144)
    var_146 = var_92.split(var_6)[var_2]
    var_147 = len(var_146)
    var_148 = module_0.Random()
    var_149 = 19
    var_150 = var_148.uniform(var_1, var_2, var_149)
    var_151 = str(var_150)
    var_152 = var_96.split(var_6)[var_2]
    var_153 = len(var_152)
    var_154 = module_0.Random()
    var_155 = 20
    var_156 = var_154.uniform(var_1, var_2, var_155)
    var_157 = str(var_156)
    var_158 = var_100.split(var_6)[var_2]
    var_159 = len(var_158)
    var_160 = module_0.Random()
    var_161 = 21
    var_162 = var_160.uniform(var_1, var_2, var_161)
    var_163 = str(var_162)
    var_164 = var_104.split(var_6)[var_2]
    var_165 = len(var_164)
    var_166 = module_0.Random()



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = module_0.Random()
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_5.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = module_0.Random()
    var_12 = 0
    var_13 = var_11.randints(var_12)
    var_14 = module_0.Random()
    var_15 = -1
    var_16 = var_14.randints(var_15)
    var_17 = module_0.Random()
    var_18 = 'abc'
    var_19 = module_0.Random()
    var_20 = '123'
    var_21 = 8
    var_22 = module_0.Random()
    var_23 = var_22.generate_string_by_mask()
    var_24 = len(var_23)
    assert var_24 == 4
    var_25 = 0
    var_26 = var_23[var_25]
    var_27 = var_23[var_3:]
    var_28 = module_0.Random()
    var_29 = '@@##'
    var_30 = '@'
    var_31 = '#'
    var_32 = var_28.generate_string_by_mask(var_29, var_30, var_31)
    var_33 = len(var_32)
    assert var_33 == 4
    var_34 = 2
    var_35 = var_32[:var_34]
    var_36 = var_32[var_34:]
    var_37 = module_0.Random()
    var_38 = '@@##'
    var_39 = '@'
    var_40 = var_37.generate_string_by_mask(var_38, var_39, var_39)
    var_41 = module_0.Random()
    var_42 = var_41.uniform(var_25, var_3)
    var_43 = module_0.Random()
    var_44 = 1.5
    var_45 = 2.5
    var_46 = var_43.uniform(var_44, var_45, var_34)
    var_47 = str(var_46)
    var_48 = '.'
    var_49 = var_41.split(var_48)[var_3]
    var_50 = len(var_49)
    var_51 = module_0.Random()
    var_52 = var_51.randbytes()
    var_53 = len(var_52)
    assert var_53 == 16
    var_54 = module_0.Random()
    var_55 = var_54.randbytes(var_21)
    var_56 = len(var_55)
    assert var_56 == 8
    var_57 = module_0.Random()
    var_58 = 'a'
    var_59 = 'b'
    var_60 = 'c'
    var_61 = 0.5
    var_62 = 0.3
    var_63 = 0.2
    var_64 = {var_58: var_61, var_59: var_62, var_60: var_63}
    var_65 = var_57.weighted_choice(var_64)
    var_66 = module_0.Random()
    var_67 = {}
    var_68 = var_66.weighted_choice(var_67)
    var_69 = 1
    var_70 = 2
    var_71 = 3
    var_72 = module_0.Random()
    var_73 = module_0.Random()
    var_74 = module_0.Random()
    var_75 = 42
    var_76 = module_0.Random()
    var_77 = module_0.Random()
    var_78 = module_0.Random()
    var_79 = 'All test cases passed!'
    var_80 = print(var_79)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_3.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_10.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 100
    var_19 = 200
    var_20 = var_17.uniform(var_18, var_19, var_1)
    var_21 = module_0.Random()
    var_22 = 0.5
    var_23 = 0.6
    var_24 = var_21.uniform(var_22, var_23, var_11)
    var_25 = str(var_24)
    var_26 = var_18.split(var_6)[var_2]
    var_27 = len(var_26)
    var_28 = module_0.Random()
    var_29 = -1
    var_30 = 20
    var_31 = var_28.uniform(var_29, var_2, var_30)
    var_32 = str(var_31)
    var_33 = var_23.split(var_6)[var_2]
    var_34 = len(var_33)
    var_35 = module_0.Random()
    var_36 = var_35.uniform(var_1, var_1, var_3)
    assert var_36 == 0
    var_37 = module_0.Random()
    var_38 = var_37.uniform(var_11, var_11, var_12)
    assert var_38 == 10
    var_39 = module_0.Random()
    var_40 = -5
    var_41 = -5
    var_42 = var_39.uniform(var_40, var_41, var_1)
    assert var_42 == -5
    var_43 = module_0.Random()
    var_44 = var_43.uniform(var_1, var_2, var_1)
    var_45 = module_0.Random()
    var_46 = var_45.uniform(var_1, var_2, var_2)
    var_47 = str(var_46)
    var_48 = var_29.split(var_6)[var_2]
    var_49 = len(var_48)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Random()



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = {}
    var_2 = var_0.weighted_choice(var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.Random()
    var_7 = var_6.weighted_choice(var_5)
    assert var_7 == 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = 2
    var_11 = 3
    var_12 = {var_3: var_4, var_8: var_10, var_9: var_11}
    var_13 = module_0.Random()
    var_14 = var_13.weighted_choice(var_12)
    var_15 = 0
    var_16 = {var_3: var_15, var_8: var_4}
    var_17 = module_0.Random()
    var_18 = var_17.weighted_choice(var_16)
    assert var_18 == 'b'
    var_19 = -1
    var_20 = {var_3: var_19, var_8: var_4}
    var_21 = module_0.Random()
    var_22 = var_21.weighted_choice(var_20)
    assert var_22 == 'b'
    var_23 = 0.5
    var_24 = {var_3: var_23, var_8: var_23}
    var_25 = module_0.Random()
    var_26 = var_25.weighted_choice(var_24)
    var_27 = 1000000
    var_28 = {var_3: var_27, var_8: var_4}
    var_29 = module_0.Random()
    var_30 = var_29.weighted_choice(var_28)
    var_31 = {var_3: var_4, var_8: var_4, var_9: var_4}
    var_32 = module_0.Random()
    var_33 = var_32.weighted_choice(var_31)
    var_34 = {var_4: var_4, var_10: var_10, var_11: var_11}
    var_35 = module_0.Random()
    var_36 = var_35.weighted_choice(var_34)
    var_37 = module_0.Random()
    var_38 = var_37.weighted_choice(var_34)
    var_39 = 9
    var_40 = {var_3: var_4, var_8: var_39}
    var_41 = {var_3: var_15, var_8: var_15}
    var_42 = module_0.Random()
    var_43 = var_42.weighted_choice(var_40)
    var_44 = var_41[var_43]
    var_45 = 1
    var_46 = var_44 + var_45
    var_47 = var_41[var_8]
    var_48 = var_41[var_42]
    var_49 = max(var_48, var_46)
    var_50 = var_47 / var_49
    var_51 = 'All tests passed!'
    var_52 = print(var_51)



# Parsed testcases at query #22
#--------------------------



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
    var_11 = -1
    var_12 = var_1.randints(var_11)
    var_13 = 0
    var_14 = var_1.randints(var_13)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = '###'
    var_7 = var_0.generate_string_by_mask(var_6, var_2, var_3)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = '@#@#'
    var_10 = var_0.generate_string_by_mask(var_9, var_2, var_3)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 0
    var_13 = var_10[var_12]
    var_14 = var_10[var_12]
    var_15 = 1
    var_16 = var_10[var_15]
    var_17 = 2
    var_18 = var_10[var_17]
    var_19 = var_10[var_17]
    var_20 = 3
    var_21 = var_10[var_20]
    var_22 = 'A@B#C'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    assert var_24 == 5
    var_25 = var_23[var_15]
    var_26 = var_23[var_15]
    var_27 = var_23[var_20]
    var_28 = '@#@#'
    var_29 = '@'
    var_30 = var_0.generate_string_by_mask(var_28, var_29, var_29)
    var_31 = ''
    var_32 = var_0.generate_string_by_mask(var_31, var_29, var_30)
    assert var_32 == ''
    var_33 = 'ABC'
    var_34 = var_0.generate_string_by_mask(var_33, var_29, var_30)
    assert var_34 == 'ABC'
    var_35 = '@@@###'
    var_36 = var_0.generate_string_by_mask(var_35, var_29, var_30)
    var_37 = len(var_36)
    assert var_37 == 6
    var_38 = var_36[:var_20]
    var_39 = var_36[:var_20]
    var_40 = var_36[var_20:]
    var_41 = '###@@@'
    var_42 = var_0.generate_string_by_mask(var_41, var_29, var_30)
    var_43 = len(var_42)
    assert var_43 == 6
    var_44 = var_42[:var_20]
    var_45 = var_42[var_20:]
    var_46 = var_42[var_20:]
    var_47 = '@#!@#'
    var_48 = var_0.generate_string_by_mask(var_47, var_29, var_30)
    var_49 = len(var_48)
    assert var_49 == 5
    var_50 = var_48[var_12]
    var_51 = var_48[var_12]
    var_52 = var_48[var_15]
    var_53 = var_48[var_20]
    var_54 = var_48[var_20]
    var_55 = 4
    var_56 = var_48[var_55]
    var_57 = 'All test cases passed!'
    var_58 = print(var_57)



# Parsed testcases at query #24
#--------------------------



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



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_4.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_12.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 100
    var_19 = 200
    var_20 = var_17.uniform(var_18, var_19, var_1)
    var_21 = str(var_20)
    var_22 = var_18.split(var_6)[var_2]
    var_23 = len(var_22)
    var_24 = module_0.Random()
    var_25 = 0.5
    var_26 = var_24.uniform(var_25, var_25, var_11)
    var_27 = str(var_26)
    var_28 = var_23.split(var_6)[var_2]
    var_29 = len(var_28)
    var_30 = module_0.Random()
    var_31 = -1000
    var_32 = -500
    var_33 = 20
    var_34 = var_30.uniform(var_31, var_32, var_33)
    var_35 = str(var_34)
    var_36 = var_30.split(var_6)[var_2]
    var_37 = len(var_36)
    var_38 = module_0.Random()
    var_39 = var_38.uniform(var_1, var_1, var_12)
    assert var_39 == 0
    var_40 = str(var_39)
    var_41 = var_34.split(var_6)[var_2]
    var_42 = len(var_41)
    var_43 = module_0.Random()
    var_44 = 1.23
    var_45 = 4.56
    var_46 = 2
    var_47 = var_43.uniform(var_44, var_45, var_46)
    var_48 = str(var_47)
    var_49 = var_41.split(var_6)[var_2]
    var_50 = len(var_49)
    var_51 = module_0.Random()
    var_52 = -0.001
    var_53 = 0.001
    var_54 = 8
    var_55 = var_51.uniform(var_52, var_53, var_54)
    var_56 = str(var_55)
    var_57 = var_48.split(var_6)[var_2]
    var_58 = len(var_57)
    var_59 = module_0.Random()
    var_60 = 1000000
    var_61 = 2000000
    var_62 = 12
    var_63 = var_59.uniform(var_60, var_61, var_62)
    var_64 = str(var_63)
    var_65 = var_55.split(var_6)[var_2]
    var_66 = len(var_65)
    var_67 = module_0.Random()
    var_68 = -3.14
    var_69 = 3.14
    var_70 = 3
    var_71 = var_67.uniform(var_68, var_69, var_70)
    var_72 = str(var_71)
    var_73 = var_62.split(var_6)[var_2]
    var_74 = len(var_73)
    var_75 = module_0.Random()
    var_76 = 0.0001
    var_77 = 6
    var_78 = var_75.uniform(var_1, var_76, var_77)
    var_79 = str(var_78)
    var_80 = var_68.split(var_6)[var_2]
    var_81 = len(var_80)
    var_82 = module_0.Random()
    var_83 = -100
    var_84 = var_82.uniform(var_83, var_18, var_2)
    var_85 = str(var_84)
    var_86 = var_73.split(var_6)[var_2]
    var_87 = len(var_86)
    var_88 = module_0.Random()
    var_89 = var_88.uniform(var_2, var_2, var_1)
    var_90 = str(var_89)
    var_91 = var_77.split(var_6)[var_2]
    var_92 = len(var_91)
    var_93 = module_0.Random()
    var_94 = -999.999
    var_95 = 999.999
    var_96 = var_93.uniform(var_94, var_95, var_11)
    var_97 = str(var_96)
    var_98 = var_83.split(var_6)[var_2]
    var_99 = len(var_98)
    var_100 = module_0.Random()
    var_101 = 0.123456789
    var_102 = 0.987654321
    var_103 = 9
    var_104 = var_100.uniform(var_101, var_102, var_103)
    var_105 = str(var_104)
    var_106 = var_90.split(var_6)[var_2]
    var_107 = len(var_106)
    var_108 = module_0.Random()
    var_109 = -1000.0
    var_110 = 1000.0
    var_111 = 4
    var_112 = var_108.uniform(var_109, var_110, var_111)
    var_113 = str(var_112)
    var_114 = var_97.split(var_6)[var_2]
    var_115 = len(var_114)
    var_116 = module_0.Random()
    var_117 = var_116.uniform(var_1, var_1, var_46)
    var_118 = str(var_117)
    var_119 = var_101.split(var_6)[var_2]
    var_120 = len(var_119)
    var_121 = module_0.Random()
    var_122 = -1.5
    var_123 = 1.5
    var_124 = 7
    var_125 = var_121.uniform(var_122, var_123, var_124)
    var_126 = str(var_125)
    var_127 = var_108.split(var_6)[var_2]
    var_128 = len(var_127)
    var_129 = module_0.Random()
    var_130 = var_129.uniform(var_18, var_19, var_70)
    var_131 = str(var_130)
    var_132 = var_112.split(var_6)[var_2]
    var_133 = len(var_132)
    var_134 = module_0.Random()
    var_135 = -0.5
    var_136 = var_134.uniform(var_135, var_25, var_2)
    var_137 = str(var_136)
    var_138 = var_117.split(var_6)[var_2]
    var_139 = len(var_138)
    var_140 = module_0.Random()
    var_141 = var_140.uniform(var_1, var_2, var_1)
    var_142 = str(var_141)
    var_143 = var_121.split(var_6)[var_2]
    var_144 = len(var_143)
    var_145 = module_0.Random()
    var_146 = -10.0
    var_147 = var_145.uniform(var_146, var_11, var_77)
    var_148 = str(var_147)
    var_149 = var_126.split(var_6)[var_2]
    var_150 = len(var_149)
    var_151 = module_0.Random()
    var_152 = var_151.uniform(var_1, var_1, var_1)
    var_153 = str(var_152)
    var_154 = var_130.split(var_6)[var_2]
    var_155 = len(var_154)
    var_156 = module_0.Random()
    var_157 = var_156.uniform(var_2, var_46, var_11)
    var_158 = str(var_157)
    var_159 = var_134.split(var_6)[var_2]
    var_160 = len(var_159)
    var_161 = module_0.Random()
    var_162 = -100.0
    var_163 = var_161.uniform(var_162, var_18, var_12)



# Parsed testcases at query #26
#--------------------------



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
    var_9 = -1
    var_10 = var_0.randbytes(var_9)
    var_11 = 1000
    var_12 = var_0.randbytes(var_11)
    var_13 = len(var_12)
    assert var_13 == 1000
    var_14 = 16
    var_15 = var_0.randbytes(var_14)
    var_16 = var_0.randbytes(var_14)
    var_17 = var_0.randbytes(var_14)
    var_18 = 42
    var_19 = module_0.Random(var_18)
    var_20 = module_0.Random(var_18)
    var_21 = var_19.randbytes(var_14)
    var_22 = var_20.randbytes(var_14)
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = var_0.randbytes(var_1)
    assert var_2 == b''
    var_3 = module_0.Random()
    var_4 = 1
    var_5 = var_3.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = module_0.Random()
    var_8 = 10
    var_9 = var_7.randbytes(var_8)
    var_10 = len(var_9)
    assert var_10 == 10
    var_11 = module_0.Random()
    var_12 = 100
    var_13 = var_11.randbytes(var_12)
    var_14 = len(var_13)
    assert var_14 == 100
    var_15 = module_0.Random()
    var_16 = 1000
    var_17 = var_15.randbytes(var_16)
    var_18 = len(var_17)
    assert var_18 == 1000



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 12345
    var_3 = module_0.Random(var_2)
    var_4 = module_0.Random(var_2)
    var_5 = None
    var_6 = module_0.Random(var_5)
    var_7 = module_0.Random(var_5)
    var_8 = 'test_seed'
    var_9 = module_0.Random(var_8)
    var_10 = module_0.Random(var_8)
    var_11 = b'test_seed'
    var_12 = module_0.Random(var_11)
    var_13 = module_0.Random(var_11)
    var_14 = 3.14
    var_15 = module_0.Random(var_14)
    var_16 = module_0.Random(var_14)
    var_17 = 42
    var_18 = module_0.Random(var_17)
    var_19 = module_0.Random(var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #29
#--------------------------



def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = 0.5
    var_4 = 0.3
    var_5 = 0.2
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.Random()
    var_8 = var_7.weighted_choice(var_6)
    var_9 = 'Test case 1 passed: Normal case with valid choices and weights'
    var_10 = print(var_9)
    var_11 = {}
    var_12 = var_7.weighted_choice(var_11)
    var_13 = 'Test case 2 failed: Expected ValueError for empty choices'
    var_14 = print(var_13)
    var_15 = 1.0
    var_16 = {var_12: var_15}
    var_17 = var_7.weighted_choice(var_16)
    assert var_17 == 'A'
    var_18 = 'Test case 3 passed: Edge case with single choice'
    var_19 = print(var_18)
    var_20 = 0.0
    var_21 = {var_12: var_20, var_13: var_15}
    var_22 = var_7.weighted_choice(var_21)
    var_23 = 'Test case 4 passed: Edge case with zero weight'
    var_24 = print(var_23)
    var_25 = -0.5
    var_26 = 1.5
    var_27 = {var_12: var_25, var_13: var_26}
    var_28 = var_7.weighted_choice(var_27)
    var_29 = 'Test case 5 passed: Edge case with negative weight'
    var_30 = print(var_29)
    var_31 = 1000
    var_32 = range(var_31)
    var_33 = {str(i): i for i in var_32}
    var_34 = var_7.weighted_choice(var_33)
    var_35 = 'Test case 6 passed: Large number of choices'
    var_36 = print(var_35)
    var_37 = -1.0
    var_38 = {var_12: var_15, var_13: var_37}
    var_39 = var_7.weighted_choice(var_38)
    var_40 = 'Test case 7 passed: Weights that sum to zero'
    var_41 = print(var_40)
    var_42 = 0.1
    var_43 = 0.7
    var_44 = {var_12: var_42, var_13: var_5, var_14: var_43}
    var_45 = var_7.weighted_choice(var_44)
    var_46 = 'Test case 8 passed: Floating point weights'
    var_47 = print(var_46)
    var_48 = 5
    var_49 = 3
    var_50 = 2
    var_51 = {var_12: var_48, var_13: var_49, var_14: var_50}
    var_52 = var_7.weighted_choice(var_51)
    var_53 = 'Test case 9 passed: Integer weights'
    var_54 = print(var_53)
    var_55 = 3.5
    var_56 = {var_12: var_48, var_13: var_55, var_14: var_26}
    var_57 = var_7.weighted_choice(var_56)
    var_58 = 'Test case 10 passed: Mixed types of weights'
    var_59 = print(var_58)
    var_60 = 'All test cases passed!'
    var_61 = print(var_60)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = '###'
    var_7 = var_0.generate_string_by_mask(var_6, var_2, var_3)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = '@#@#'
    var_10 = var_0.generate_string_by_mask(var_9, var_2, var_3)
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 0
    var_13 = var_10[var_12]
    var_14 = var_10[var_12]
    var_15 = 1
    var_16 = var_10[var_15]
    var_17 = 2
    var_18 = var_10[var_17]
    var_19 = var_10[var_17]
    var_20 = 3
    var_21 = var_10[var_20]
    var_22 = 'A@B#C'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    assert var_24 == 5
    var_25 = var_23[var_15]
    var_26 = var_23[var_15]
    var_27 = var_23[var_20]
    var_28 = '@@@'
    var_29 = '@'
    var_30 = var_0.generate_string_by_mask(var_28, var_29, var_29)
    var_31 = 'A$B%'
    var_32 = '$'
    var_33 = '%'
    var_34 = var_0.generate_string_by_mask(var_31, var_32, var_33)
    var_35 = len(var_34)
    assert var_35 == 4
    var_36 = var_34[var_15]
    var_37 = var_34[var_15]
    var_38 = var_34[var_20]
    var_39 = 'All test cases passed!'
    var_40 = print(var_39)



# Parsed testcases at query #2
#--------------------------



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
    var_15 = '@@##'
    var_16 = '@'
    var_17 = var_0.generate_string_by_mask(var_15, var_16, var_16)
    var_18 = ''
    var_19 = var_0.generate_string_by_mask(var_18, var_8, var_9)
    assert var_19 == ''
    var_20 = '@@@'
    var_21 = var_0.generate_string_by_mask(var_20, var_8, var_9)
    var_22 = len(var_21)
    assert var_22 == 3
    var_23 = '###'
    var_24 = var_0.generate_string_by_mask(var_23, var_8, var_9)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = '@#!@#'
    var_27 = var_0.generate_string_by_mask(var_26, var_8, var_9)
    var_28 = len(var_27)
    assert var_28 == 5
    var_29 = var_27[var_16]
    var_30 = var_27[var_5]
    var_31 = 3
    var_32 = var_27[var_31]
    var_33 = 4
    var_34 = var_27[var_33]
    var_35 = 'AA00'
    var_36 = 'A'
    var_37 = '0'
    var_38 = var_0.generate_string_by_mask(var_35, var_36, var_37)
    var_39 = len(var_38)
    assert var_39 == 4
    var_40 = var_38[:var_12]
    var_41 = var_38[var_12:]
    var_42 = var_0.generate_string_by_mask(var_8, var_8, var_9)
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = 10
    var_45 = var_8 * var_44
    var_46 = var_0.generate_string_by_mask(var_45, var_8, var_9)
    var_47 = len(var_46)
    assert var_47 == 10
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)



# Parsed testcases at query #4
#--------------------------



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
    var_14 = '@@@@'
    var_15 = var_0.generate_string_by_mask(var_14, var_2, var_3)
    var_16 = len(var_15)
    assert var_16 == 4
    var_17 = '####'
    var_18 = var_0.generate_string_by_mask(var_17, var_2, var_3)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = 'AB@12#CD'
    var_21 = var_0.generate_string_by_mask(var_20, var_2, var_3)
    var_22 = len(var_21)
    assert var_22 == 8
    var_23 = var_21[var_10]
    var_24 = 5
    var_25 = var_21[var_24]
    var_26 = '@@##'
    var_27 = '@'
    var_28 = var_0.generate_string_by_mask(var_26, var_27, var_27)
    var_29 = ''
    var_30 = var_0.generate_string_by_mask(var_29, var_27, var_28)
    assert var_30 == ''
    var_31 = 'FIXED'
    var_32 = var_0.generate_string_by_mask(var_31, var_27, var_28)
    assert var_32 == 'FIXED'
    var_33 = 'LLDD'
    var_34 = 'L'
    var_35 = 'D'
    var_36 = var_0.generate_string_by_mask(var_33, var_34, var_35)
    var_37 = len(var_36)
    assert var_37 == 4
    var_38 = var_36[var_6]
    var_39 = var_36[var_8]
    var_40 = var_36[var_10]
    var_41 = var_36[var_12]
    var_42 = 100
    var_43 = var_27 * var_42
    var_44 = var_0.generate_string_by_mask(var_43, var_27, var_28)
    var_45 = len(var_44)
    assert var_45 == 100
    var_46 = '@-#@'
    var_47 = var_0.generate_string_by_mask(var_46, var_27, var_28)
    var_48 = len(var_47)
    assert var_48 == 4
    var_49 = var_47[var_6]
    var_50 = var_47[var_10]
    var_51 = var_47[var_12]
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@#'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = '@@##'
    var_12 = var_0.generate_string_by_mask(var_11, var_2, var_3)
    var_13 = len(var_12)
    assert var_13 == 4
    var_14 = var_12[var_6]
    var_15 = var_12[var_6]
    var_16 = var_12[var_9]
    var_17 = var_12[var_9]
    var_18 = 2
    var_19 = var_12[var_18]
    var_20 = 3
    var_21 = var_12[var_20]
    var_22 = 'AB@#CD'
    var_23 = var_0.generate_string_by_mask(var_22, var_2, var_3)
    var_24 = len(var_23)
    assert var_24 == 6
    var_25 = var_23[var_18]
    var_26 = var_23[var_18]
    var_27 = var_23[var_20]
    var_28 = '@@@'
    var_29 = var_0.generate_string_by_mask(var_28, var_2, var_3)
    var_30 = len(var_29)
    assert var_30 == 3
    var_31 = '###'
    var_32 = var_0.generate_string_by_mask(var_31, var_2, var_3)
    var_33 = len(var_32)
    assert var_33 == 3
    var_34 = ''
    var_35 = var_0.generate_string_by_mask(var_34, var_2, var_3)
    assert var_35 == ''
    var_36 = '@#'
    var_37 = '@'
    var_38 = var_0.generate_string_by_mask(var_36, var_37, var_37)
    var_39 = 'a1b2'
    var_40 = 'a'
    var_41 = '1'
    var_42 = var_0.generate_string_by_mask(var_39, var_40, var_41)
    var_43 = len(var_42)
    assert var_43 == 4
    var_44 = var_42[var_6]
    var_45 = var_42[var_6]
    var_46 = var_42[var_9]
    var_47 = '@#!@#'
    var_48 = var_0.generate_string_by_mask(var_47, var_37, var_38)
    var_49 = len(var_48)
    assert var_49 == 5
    var_50 = var_48[var_6]
    var_51 = var_48[var_6]
    var_52 = var_48[var_9]
    var_53 = var_48[var_20]
    var_54 = var_48[var_20]
    var_55 = 4
    var_56 = var_48[var_55]
    var_57 = 'All tests passed!'
    var_58 = print(var_57)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_3.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 2
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_10.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 5
    var_19 = var_17.uniform(var_18, var_18, var_1)
    assert var_19 == 5
    var_20 = module_0.Random()
    var_21 = 0.1
    var_22 = 0.2
    var_23 = var_20.uniform(var_21, var_22, var_11)
    var_24 = str(var_23)
    var_25 = var_16.split(var_6)[var_2]
    var_26 = len(var_25)
    var_27 = module_0.Random()
    var_28 = -100
    var_29 = 100
    var_30 = var_27.uniform(var_28, var_29, var_18)
    var_31 = str(var_30)
    var_32 = var_21.split(var_6)[var_2]
    var_33 = len(var_32)
    var_34 = module_0.Random()
    var_35 = var_34.uniform(var_1, var_1, var_3)
    assert var_35 == 0
    var_36 = module_0.Random()
    var_37 = -1
    var_38 = var_36.uniform(var_37, var_2, var_1)
    var_39 = module_0.Random()
    var_40 = 0.0001
    var_41 = 0.0002
    var_42 = 20
    var_43 = var_39.uniform(var_40, var_41, var_42)
    var_44 = str(var_43)
    var_45 = var_29.split(var_6)[var_2]
    var_46 = len(var_45)
    var_47 = module_0.Random()
    var_48 = 200
    var_49 = var_47.uniform(var_29, var_48, var_2)
    var_50 = str(var_49)
    var_51 = var_33.split(var_6)[var_2]
    var_52 = len(var_51)
    var_53 = module_0.Random()
    var_54 = -0.5
    var_55 = 0.5
    var_56 = 3
    var_57 = var_53.uniform(var_54, var_55, var_56)
    var_58 = str(var_57)
    var_59 = var_39.split(var_6)[var_2]
    var_60 = len(var_59)
    var_61 = 'All test cases passed!'
    var_62 = print(var_61)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_3.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_10.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = var_17.uniform(var_1, var_1, var_3)
    assert var_18 == 0
    var_19 = module_0.Random()
    var_20 = var_19.uniform(var_1, var_2, var_1)
    var_21 = str(var_20)
    var_22 = var_13.split(var_6)[var_2]
    var_23 = len(var_22)
    var_24 = module_0.Random()
    var_25 = var_24.uniform(var_1, var_2, var_2)
    var_26 = str(var_25)
    var_27 = var_16.split(var_6)[var_2]
    var_28 = len(var_27)
    var_29 = module_0.Random()
    var_30 = 2
    var_31 = var_29.uniform(var_1, var_2, var_30)
    var_32 = str(var_31)
    var_33 = var_20.split(var_6)[var_2]
    var_34 = len(var_33)
    var_35 = module_0.Random()
    var_36 = 3
    var_37 = var_35.uniform(var_1, var_2, var_36)
    var_38 = str(var_37)
    var_39 = var_24.split(var_6)[var_2]
    var_40 = len(var_39)
    var_41 = module_0.Random()
    var_42 = 4
    var_43 = var_41.uniform(var_1, var_2, var_42)
    var_44 = str(var_43)
    var_45 = var_28.split(var_6)[var_2]
    var_46 = len(var_45)
    var_47 = module_0.Random()
    var_48 = var_47.uniform(var_1, var_2, var_12)
    var_49 = str(var_48)
    var_50 = var_31.split(var_6)[var_2]
    var_51 = len(var_50)
    var_52 = module_0.Random()
    var_53 = 6
    var_54 = var_52.uniform(var_1, var_2, var_53)
    var_55 = str(var_54)
    var_56 = var_35.split(var_6)[var_2]
    var_57 = len(var_56)
    var_58 = module_0.Random()
    var_59 = 7
    var_60 = var_58.uniform(var_1, var_2, var_59)
    var_61 = str(var_60)
    var_62 = var_39.split(var_6)[var_2]
    var_63 = len(var_62)
    var_64 = module_0.Random()
    var_65 = 8
    var_66 = var_64.uniform(var_1, var_2, var_65)
    var_67 = str(var_66)
    var_68 = var_43.split(var_6)[var_2]
    var_69 = len(var_68)
    var_70 = module_0.Random()
    var_71 = 9
    var_72 = var_70.uniform(var_1, var_2, var_71)
    var_73 = str(var_72)
    var_74 = var_47.split(var_6)[var_2]
    var_75 = len(var_74)
    var_76 = module_0.Random()
    var_77 = var_76.uniform(var_1, var_2, var_11)
    var_78 = str(var_77)
    var_79 = var_50.split(var_6)[var_2]
    var_80 = len(var_79)
    var_81 = module_0.Random()
    var_82 = 11
    var_83 = var_81.uniform(var_1, var_2, var_82)
    var_84 = str(var_83)
    var_85 = var_54.split(var_6)[var_2]
    var_86 = len(var_85)
    var_87 = module_0.Random()
    var_88 = 12
    var_89 = var_87.uniform(var_1, var_2, var_88)
    var_90 = str(var_89)
    var_91 = var_58.split(var_6)[var_2]
    var_92 = len(var_91)
    var_93 = module_0.Random()
    var_94 = 13
    var_95 = var_93.uniform(var_1, var_2, var_94)
    var_96 = str(var_95)
    var_97 = var_62.split(var_6)[var_2]
    var_98 = len(var_97)
    var_99 = module_0.Random()
    var_100 = 14
    var_101 = var_99.uniform(var_1, var_2, var_100)
    var_102 = str(var_101)
    var_103 = var_66.split(var_6)[var_2]
    var_104 = len(var_103)
    var_105 = module_0.Random()
    var_106 = var_105.uniform(var_1, var_2, var_3)
    var_107 = str(var_106)
    var_108 = var_69.split(var_6)[var_2]
    var_109 = len(var_108)
    var_110 = module_0.Random()
    var_111 = 16
    var_112 = var_110.uniform(var_1, var_2, var_111)
    var_113 = str(var_112)
    var_114 = var_73.split(var_6)[var_2]
    var_115 = len(var_114)
    var_116 = module_0.Random()
    var_117 = 17
    var_118 = var_116.uniform(var_1, var_2, var_117)
    var_119 = str(var_118)
    var_120 = var_77.split(var_6)[var_2]
    var_121 = len(var_120)
    var_122 = module_0.Random()
    var_123 = 18
    var_124 = var_122.uniform(var_1, var_2, var_123)
    var_125 = str(var_124)
    var_126 = var_81.split(var_6)[var_2]
    var_127 = len(var_126)
    var_128 = module_0.Random()
    var_129 = 19
    var_130 = var_128.uniform(var_1, var_2, var_129)
    var_131 = str(var_130)
    var_132 = var_85.split(var_6)[var_2]
    var_133 = len(var_132)
    var_134 = module_0.Random()
    var_135 = 20
    var_136 = var_134.uniform(var_1, var_2, var_135)
    var_137 = str(var_136)
    var_138 = var_89.split(var_6)[var_2]
    var_139 = len(var_138)
    var_140 = module_0.Random()
    var_141 = 21
    var_142 = var_140.uniform(var_1, var_2, var_141)
    var_143 = str(var_142)
    var_144 = var_93.split(var_6)[var_2]
    var_145 = len(var_144)
    var_146 = module_0.Random()
    var_147 = 22
    var_148 = var_146.uniform(var_1, var_2, var_147)
    var_149 = str(var_148)
    var_150 = var_97.split(var_6)[var_2]
    var_151 = len(var_150)
    var_152 = module_0.Random()
    var_153 = 23
    var_154 = var_152.uniform(var_1, var_2, var_153)
    var_155 = str(var_154)
    var_156 = var_101.split(var_6)[var_2]
    var_157 = len(var_156)
    var_158 = module_0.Random()
    var_159 = 24
    var_160 = var_158.uniform(var_1, var_2, var_159)
    var_161 = str(var_160)
    var_162 = var_105.split(var_6)[var_2]
    var_163 = len(var_162)
    var_164 = module_0.Random()
    var_165 = 25
    var_166 = var_164.uniform(var_1, var_2, var_165)
    var_167 = str(var_166)
    var_168 = var_109.split(var_6)[var_2]
    var_169 = len(var_168)
    var_170 = module_0.Random()
    var_171 = 26
    var_172 = var_170.uniform(var_1, var_2, var_171)



# Parsed testcases at query #8
#--------------------------



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
    var_9 = {}
    var_10 = var_7.weighted_choice(var_9)
    var_11 = 'x'
    var_12 = 1.0
    var_13 = {var_11: var_12}
    var_14 = var_7.weighted_choice(var_13)
    assert var_14 == 'x'
    var_15 = 0.0
    var_16 = {var_10: var_15, var_1: var_12}
    var_17 = var_7.weighted_choice(var_16)
    assert var_17 == 'b'
    var_18 = -1.0
    var_19 = 2.0
    var_20 = {var_10: var_18, var_1: var_19}
    var_21 = var_7.weighted_choice(var_20)
    var_22 = 100
    var_23 = range(var_22)
    var_24 = {i: i for i in var_23}
    var_25 = var_7.weighted_choice(var_24)
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #9
#--------------------------



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
    var_9 = {}
    var_10 = var_7.weighted_choice(var_9)
    var_11 = 1.0
    var_12 = {var_9: var_11}
    var_13 = var_7.weighted_choice(var_12)
    assert var_13 == 'a'
    var_14 = 0.0
    var_15 = {var_9: var_14, var_10: var_11}
    var_16 = var_7.weighted_choice(var_15)
    assert var_16 == 'b'
    var_17 = -1.0
    var_18 = 2.0
    var_19 = {var_9: var_17, var_10: var_18}
    var_20 = var_7.weighted_choice(var_19)
    assert var_20 == 'b'



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = module_0.Random()
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_5.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = module_0.Random()
    var_12 = 0
    var_13 = var_11.randints(var_12)
    var_14 = module_0.Random()
    var_15 = 'abc'
    var_16 = len(var_9)
    assert var_16 == 5
    var_17 = module_0.Random()
    var_18 = '@###'
    var_19 = '@'
    var_20 = '#'
    var_21 = var_17.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 4
    var_23 = 0
    var_24 = var_21[var_23]
    var_25 = var_21[var_3:]
    var_26 = module_0.Random()
    var_27 = '@@@'
    var_28 = '@'
    var_29 = var_26.generate_string_by_mask(var_27, var_28, var_28)
    var_30 = module_0.Random()
    var_31 = 2.0
    var_32 = var_30.uniform(var_3, var_31, var_31)
    var_33 = str(var_32)
    var_34 = '.'
    var_35 = var_29.split(var_34)[var_3]
    var_36 = len(var_35)
    var_37 = module_0.Random()
    var_38 = 8
    var_39 = var_37.randbytes(var_38)
    var_40 = len(var_39)
    assert var_40 == 8
    var_41 = module_0.Random()
    var_42 = 'a'
    var_43 = 'b'
    var_44 = 'c'
    var_45 = 0.5
    var_46 = 0.3
    var_47 = 0.2
    var_48 = {var_42: var_45, var_43: var_46, var_44: var_47}
    var_49 = var_41.weighted_choice(var_48)
    var_50 = module_0.Random()
    var_51 = {}
    var_52 = var_50.weighted_choice(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = module_0.Random()
    var_57 = module_0.Random()
    var_58 = 123
    var_59 = module_0.Random()
    var_60 = module_0.Random()
    var_61 = 456
    var_62 = module_0.Random()
    var_63 = 456
    var_64 = module_0.Random()
    var_65 = module_0.Random()
    var_66 = 'All test cases passed!'
    var_67 = print(var_66)



# Parsed testcases at query #11
#--------------------------



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
    var_14 = '@@@@'
    var_15 = var_0.generate_string_by_mask(var_14, var_2, var_3)
    var_16 = len(var_15)
    assert var_16 == 4
    var_17 = '####'
    var_18 = var_0.generate_string_by_mask(var_17, var_2, var_3)
    var_19 = len(var_18)
    assert var_19 == 4
    var_20 = 'AB@12#'
    var_21 = var_0.generate_string_by_mask(var_20, var_2, var_3)
    var_22 = len(var_21)
    assert var_22 == 6
    var_23 = var_21[var_10]
    var_24 = 5
    var_25 = var_21[var_24]
    var_26 = '@@##'
    var_27 = '@'
    var_28 = var_0.generate_string_by_mask(var_26, var_27, var_27)
    var_29 = ''
    var_30 = var_0.generate_string_by_mask(var_29, var_27, var_28)
    assert var_30 == ''
    var_31 = 'FIXED'
    var_32 = var_0.generate_string_by_mask(var_31, var_27, var_28)
    assert var_32 == 'FIXED'
    var_33 = '@#@#'
    var_34 = var_0.generate_string_by_mask(var_33, var_27, var_28)
    var_35 = len(var_34)
    assert var_35 == 4
    var_36 = var_34[var_6]
    var_37 = var_34[var_10]
    var_38 = var_34[var_8]
    var_39 = var_34[var_12]
    var_40 = '@!#$%'
    var_41 = var_0.generate_string_by_mask(var_40, var_27, var_28)
    var_42 = len(var_41)
    assert var_42 == 5
    var_43 = var_41[var_6]
    var_44 = '@@##€'
    var_45 = var_0.generate_string_by_mask(var_44, var_27, var_28)
    var_46 = len(var_45)
    assert var_46 == 5
    var_47 = var_45[var_6]
    var_48 = var_45[var_8]
    var_49 = var_45[var_10]
    var_50 = var_45[var_12]
    var_51 = 'All tests passed!'
    var_52 = print(var_51)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = module_0.Random()
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_5.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = module_0.Random()
    var_12 = 0
    var_13 = var_11.randints(var_12)
    var_14 = module_0.Random()
    var_15 = 'abc'
    var_16 = len(var_9)
    assert var_16 == 5
    var_17 = module_0.Random()
    var_18 = '@###'
    var_19 = '@'
    var_20 = '#'
    var_21 = var_17.generate_string_by_mask(var_18, var_19, var_20)
    var_22 = len(var_21)
    assert var_22 == 4
    var_23 = 0
    var_24 = var_21[var_23]
    var_25 = var_21[var_3:]
    var_26 = module_0.Random()
    var_27 = '@@@'
    var_28 = '@'
    var_29 = var_26.generate_string_by_mask(var_27, var_28, var_28)
    var_30 = module_0.Random()
    var_31 = 2.0
    var_32 = var_30.uniform(var_3, var_31, var_31)
    var_33 = str(var_32)
    var_34 = '.'
    var_35 = var_29.split(var_34)[var_3]
    var_36 = len(var_35)
    var_37 = module_0.Random()
    var_38 = 8
    var_39 = var_37.randbytes(var_38)
    var_40 = len(var_39)
    assert var_40 == 8
    var_41 = module_0.Random()
    var_42 = 'a'
    var_43 = 'b'
    var_44 = 'c'
    var_45 = 0.5
    var_46 = 0.3
    var_47 = 0.2
    var_48 = {var_42: var_45, var_43: var_46, var_44: var_47}
    var_49 = var_41.weighted_choice(var_48)
    var_50 = module_0.Random()
    var_51 = {}
    var_52 = var_50.weighted_choice(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = module_0.Random()
    var_57 = module_0.Random()
    var_58 = module_0.Random()
    var_59 = 42
    var_60 = module_0.Random()
    var_61 = module_0.Random()
    var_62 = module_0.Random()
    var_63 = module_0.Random()
    var_64 = module_0.Random()
    var_65 = 3
    var_66 = [var_3, var_31, var_65]
    var_67 = module_0.Random()
    var_68 = [var_3, var_31, var_65]
    var_69 = len(var_49)
    assert var_69 == 2
    var_70 = [var_3, var_31, var_65]
    var_71 = module_0.Random()
    var_72 = 4
    var_73 = [var_3, var_31, var_65, var_72, var_6]
    var_74 = set(var_73)
    var_75 = module_0.Random()
    var_76 = [var_3, var_31, var_65, var_72, var_6]
    var_77 = len(var_49)
    assert var_77 == 3
    var_78 = [var_3, var_31, var_65, var_72, var_6]
    var_79 = 'All test cases passed!'
    var_80 = print(var_79)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_4.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_12.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 100
    var_19 = 200
    var_20 = var_17.uniform(var_18, var_19, var_1)
    var_21 = str(var_20)
    var_22 = var_18.split(var_6)[var_2]
    var_23 = len(var_22)
    var_24 = module_0.Random()
    var_25 = 0.5
    var_26 = var_24.uniform(var_25, var_25, var_11)
    var_27 = str(var_26)
    var_28 = var_23.split(var_6)[var_2]
    var_29 = len(var_28)
    var_30 = module_0.Random()
    var_31 = -1000
    var_32 = 1000
    var_33 = 20
    var_34 = var_30.uniform(var_31, var_32, var_33)
    var_35 = str(var_34)
    var_36 = var_30.split(var_6)[var_2]
    var_37 = len(var_36)
    var_38 = 'All test cases pass'
    var_39 = print(var_38)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 'abc'
    var_7 = len(var_4)
    assert var_7 == 5
    var_8 = '@###'
    var_9 = '@'
    var_10 = '#'
    var_11 = var_0.generate_string_by_mask(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = var_11[var_2:]
    var_16 = 2.0
    var_17 = var_0.uniform(var_2, var_16, var_16)
    var_18 = 8
    var_19 = var_0.randbytes(var_18)
    var_20 = len(var_19)
    assert var_20 == 8
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = 0.5
    var_25 = 0.3
    var_26 = 0.2
    var_27 = {var_21: var_24, var_22: var_25, var_23: var_26}
    var_28 = var_0.weighted_choice(var_27)
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = 0
    var_33 = 1
    var_34 = 10
    var_35 = var_0.randints(var_32, var_33, var_34)
    var_36 = '@@##'
    var_37 = '@'
    var_38 = var_0.generate_string_by_mask(var_36, var_37, var_37)
    var_39 = {}
    var_40 = var_0.weighted_choice(var_39)
    var_41 = 'All test cases passed!'
    var_42 = print(var_41)



# Parsed testcases at query #15
#--------------------------



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
    var_10 = -1
    var_11 = var_0.randints(var_10)
    var_12 = 0
    var_13 = var_0.randints(var_12)



# Parsed testcases at query #16
#--------------------------



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
    var_9 = {}
    var_10 = var_7.weighted_choice(var_9)
    var_11 = 1.0
    var_12 = {var_10: var_11}
    var_13 = var_7.weighted_choice(var_12)
    assert var_13 == 'a'
    var_14 = 0.0
    var_15 = {var_10: var_14, var_1: var_11}
    var_16 = var_7.weighted_choice(var_15)
    assert var_16 == 'b'
    var_17 = -1.0
    var_18 = 2.0
    var_19 = {var_10: var_17, var_1: var_18}
    var_20 = var_7.weighted_choice(var_19)
    assert var_20 == 'b'



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 1
    var_4 = 100
    var_5 = module_0.Random()
    var_6 = 5
    var_7 = 10
    var_8 = 20
    var_9 = var_5.randints(var_6, var_7, var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = module_0.Random()
    var_12 = 0
    var_13 = var_11.randints(var_12)
    var_14 = module_0.Random()
    var_15 = 'abc'
    var_16 = module_0.Random()
    var_17 = '@###'
    var_18 = '@'
    var_19 = '#'
    var_20 = var_16.generate_string_by_mask(var_17, var_18, var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 0
    var_23 = var_20[var_22]
    var_24 = var_20[var_3:]
    var_25 = module_0.Random()
    var_26 = '@@@'
    var_27 = '@'
    var_28 = var_25.generate_string_by_mask(var_26, var_27, var_27)
    var_29 = module_0.Random()
    var_30 = 2.0
    var_31 = var_29.uniform(var_3, var_30, var_30)
    var_32 = str(var_31)
    var_33 = '.'
    var_34 = var_29.split(var_33)[var_3]
    var_35 = len(var_34)
    var_36 = module_0.Random()
    var_37 = 8
    var_38 = var_36.randbytes(var_37)
    var_39 = len(var_38)
    assert var_39 == 8
    var_40 = module_0.Random()
    var_41 = 'a'
    var_42 = 'b'
    var_43 = 'c'
    var_44 = 0.5
    var_45 = 0.3
    var_46 = 0.2
    var_47 = {var_41: var_44, var_42: var_45, var_43: var_46}
    var_48 = var_40.weighted_choice(var_47)
    var_49 = module_0.Random()
    var_50 = {}
    var_51 = var_49.weighted_choice(var_50)
    var_52 = module_0.Random()
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = module_0.Random()
    var_57 = 42
    var_58 = module_0.Random()
    var_59 = 123
    var_60 = module_0.Random()
    var_61 = module_0.Random()
    var_62 = 'All test cases passed!'
    var_63 = print(var_62)



# Parsed testcases at query #18
#--------------------------



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
    var_9 = -1
    var_10 = var_0.randbytes(var_9)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_4.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = -10
    var_10 = 10
    var_11 = 5
    var_12 = var_0.uniform(var_9, var_10, var_11)
    var_13 = str(var_12)
    var_14 = var_12.split(var_6)[var_2]
    var_15 = len(var_14)
    var_16 = 100
    var_17 = 200
    var_18 = var_0.uniform(var_16, var_17, var_1)
    var_19 = str(var_18)
    var_20 = var_18.split(var_6)[var_2]
    var_21 = len(var_20)
    var_22 = 0.5
    var_23 = var_0.uniform(var_22, var_22, var_10)
    var_24 = str(var_23)
    var_25 = var_23.split(var_6)[var_2]
    var_26 = len(var_25)
    var_27 = var_0.uniform(var_1, var_1, var_11)
    var_28 = str(var_27)
    var_29 = var_27.split(var_6)[var_2]
    var_30 = len(var_29)
    var_31 = -1
    var_32 = 2
    var_33 = var_0.uniform(var_31, var_2, var_32)
    var_34 = str(var_33)
    var_35 = var_33.split(var_6)[var_2]
    var_36 = len(var_35)
    var_37 = 20
    var_38 = var_0.uniform(var_10, var_37, var_2)
    var_39 = str(var_38)
    var_40 = var_38.split(var_6)[var_2]
    var_41 = len(var_40)
    var_42 = -100
    var_43 = -50
    var_44 = 3
    var_45 = var_0.uniform(var_42, var_43, var_44)
    var_46 = str(var_45)
    var_47 = var_45.split(var_6)[var_2]
    var_48 = len(var_47)
    var_49 = 0.001
    var_50 = 0.002
    var_51 = 6
    var_52 = var_0.uniform(var_49, var_50, var_51)
    var_53 = str(var_52)
    var_54 = var_52.split(var_6)[var_2]
    var_55 = len(var_54)
    var_56 = 1000
    var_57 = 2000
    var_58 = 4
    var_59 = var_0.uniform(var_56, var_57, var_58)
    var_60 = str(var_59)
    var_61 = var_59.split(var_6)[var_2]
    var_62 = len(var_61)
    var_63 = -0.5
    var_64 = 8
    var_65 = var_0.uniform(var_63, var_22, var_64)
    var_66 = str(var_65)
    var_67 = var_65.split(var_6)[var_2]
    var_68 = len(var_67)
    var_69 = 1.23
    var_70 = 4.56
    var_71 = var_0.uniform(var_69, var_70, var_32)
    var_72 = str(var_71)
    var_73 = var_71.split(var_6)[var_2]
    var_74 = len(var_73)
    var_75 = 0.0001
    var_76 = var_0.uniform(var_1, var_75, var_10)
    var_77 = str(var_76)
    var_78 = var_76.split(var_6)[var_2]
    var_79 = len(var_78)
    var_80 = -1000
    var_81 = -500
    var_82 = var_0.uniform(var_80, var_81, var_1)
    var_83 = str(var_82)
    var_84 = var_82.split(var_6)[var_2]
    var_85 = len(var_84)
    var_86 = 0.123456789
    var_87 = 0.987654321
    var_88 = 12
    var_89 = var_0.uniform(var_86, var_87, var_88)
    var_90 = str(var_89)
    var_91 = var_89.split(var_6)[var_2]
    var_92 = len(var_91)
    var_93 = var_0.uniform(var_10, var_10, var_11)
    var_94 = str(var_93)
    var_95 = var_93.split(var_6)[var_2]
    var_96 = len(var_95)
    var_97 = -0.001
    var_98 = var_0.uniform(var_97, var_49, var_44)
    var_99 = str(var_98)
    var_100 = var_98.split(var_6)[var_2]
    var_101 = len(var_100)
    var_102 = var_0.uniform(var_16, var_17, var_10)
    var_103 = str(var_102)
    var_104 = var_102.split(var_6)[var_2]
    var_105 = len(var_104)
    var_106 = 1e-06
    var_107 = 2e-06
    var_108 = var_0.uniform(var_106, var_107, var_3)
    var_109 = str(var_108)
    var_110 = var_108.split(var_6)[var_2]
    var_111 = len(var_110)
    var_112 = -100
    var_113 = var_0.uniform(var_112, var_16, var_2)
    var_114 = str(var_113)
    var_115 = var_113.split(var_6)[var_2]
    var_116 = len(var_115)
    var_117 = 1e-09
    var_118 = 9
    var_119 = var_0.uniform(var_1, var_117, var_118)
    var_120 = str(var_119)
    var_121 = var_119.split(var_6)[var_2]
    var_122 = len(var_121)
    var_123 = 1.5
    var_124 = 2.5
    var_125 = var_0.uniform(var_123, var_124, var_58)
    var_126 = str(var_125)
    var_127 = var_125.split(var_6)[var_2]
    var_128 = len(var_127)
    var_129 = -10
    var_130 = -5
    var_131 = var_0.uniform(var_129, var_130, var_32)
    var_132 = str(var_131)
    var_133 = var_131.split(var_6)[var_2]
    var_134 = len(var_133)
    var_135 = 0.123
    var_136 = 0.456
    var_137 = var_0.uniform(var_135, var_136, var_51)
    var_138 = str(var_137)
    var_139 = var_137.split(var_6)[var_2]
    var_140 = len(var_139)
    var_141 = var_0.uniform(var_56, var_56, var_1)
    var_142 = str(var_141)
    var_143 = var_141.split(var_6)[var_2]
    var_144 = len(var_143)
    var_145 = -0.0001
    var_146 = var_0.uniform(var_145, var_75, var_11)
    var_147 = str(var_146)
    var_148 = var_146.split(var_6)[var_2]
    var_149 = len(var_148)
    var_150 = var_0.uniform(var_49, var_50, var_44)
    var_151 = str(var_150)
    var_152 = var_150.split(var_6)[var_2]
    var_153 = len(var_152)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = 'Test the randbytes method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randbytes()
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 10
    var_5 = var_1.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 10
    var_7 = 0
    var_8 = var_1.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = -1
    var_11 = var_1.randbytes(var_10)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]
    var_7 = module_0.Random()
    var_8 = '@@##'
    var_9 = '@'
    var_10 = '#'
    var_11 = var_7.generate_string_by_mask(var_8, var_9, var_10)
    var_12 = len(var_11)
    assert var_12 == 4
    var_13 = 2
    var_14 = var_11[:var_13]
    var_15 = var_11[var_13:]
    var_16 = module_0.Random()
    var_17 = 'AA99'
    var_18 = 'A'
    var_19 = '9'
    var_20 = var_16.generate_string_by_mask(var_17, var_18, var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = var_20[:var_13]
    var_23 = var_20[var_13:]
    var_24 = module_0.Random()
    var_25 = '@@##'
    var_26 = '@'
    var_27 = var_24.generate_string_by_mask(var_25, var_26, var_26)
    var_28 = module_0.Random()
    var_29 = ''
    var_30 = var_28.generate_string_by_mask(var_29, var_9, var_10)
    assert var_30 == ''
    var_31 = module_0.Random()
    var_32 = '@@@'
    var_33 = var_31.generate_string_by_mask(var_32, var_9, var_10)
    var_34 = len(var_33)
    assert var_34 == 3
    var_35 = module_0.Random()
    var_36 = '###'
    var_37 = var_35.generate_string_by_mask(var_36, var_9, var_10)
    var_38 = len(var_37)
    assert var_38 == 3
    var_39 = module_0.Random()
    var_40 = '@#!@#'
    var_41 = var_39.generate_string_by_mask(var_40, var_9, var_10)
    var_42 = len(var_41)
    assert var_42 == 5
    var_43 = var_41[var_26]
    var_44 = var_41[var_5]
    var_45 = 3
    var_46 = var_41[var_45]
    var_47 = 4
    var_48 = var_41[var_47]
    var_49 = module_0.Random()
    var_50 = '@@@###'
    var_51 = var_49.generate_string_by_mask(var_50, var_9, var_10)
    var_52 = len(var_51)
    assert var_52 == 6
    var_53 = var_51[:var_45]
    var_54 = var_51[var_45:]
    var_55 = module_0.Random()
    var_56 = '###@@@'
    var_57 = var_55.generate_string_by_mask(var_56, var_9, var_10)
    var_58 = len(var_57)
    assert var_58 == 6
    var_59 = var_57[:var_45]
    var_60 = var_57[var_45:]
    var_61 = module_0.Random()
    var_62 = '@#@#@#'
    var_63 = var_61.generate_string_by_mask(var_62, var_9, var_10)
    var_64 = len(var_63)
    assert var_64 == 6
    var_65 = module_0.Random()
    var_66 = '!!!'
    var_67 = var_65.generate_string_by_mask(var_66, var_9, var_10)
    assert var_67 == '!!!'
    var_68 = module_0.Random()
    var_69 = '!@#'
    var_70 = var_68.generate_string_by_mask(var_69, var_9, var_10)
    var_71 = len(var_70)
    assert var_71 == 3
    var_72 = var_70[var_5]
    var_73 = var_70[var_13]
    var_74 = module_0.Random()
    var_75 = '@#!'
    var_76 = var_74.generate_string_by_mask(var_75, var_9, var_10)
    var_77 = len(var_76)
    assert var_77 == 3
    var_78 = var_76[var_26]
    var_79 = var_76[var_5]
    var_80 = module_0.Random()
    var_81 = '@!#'
    var_82 = var_80.generate_string_by_mask(var_81, var_9, var_10)
    var_83 = len(var_82)
    assert var_83 == 3
    var_84 = var_82[var_26]
    var_85 = var_82[var_13]
    var_86 = module_0.Random()
    var_87 = '@!#$%'
    var_88 = var_86.generate_string_by_mask(var_87, var_9, var_10)
    var_89 = len(var_88)
    assert var_89 == 5
    var_90 = var_88[var_26]
    var_91 = var_88[var_13]
    var_92 = module_0.Random()
    var_93 = '!@#$%^&*'
    var_94 = var_92.generate_string_by_mask(var_93, var_9, var_10)
    var_95 = len(var_94)
    assert var_95 == 8
    var_96 = var_94[var_5]
    var_97 = var_94[var_13]
    var_98 = module_0.Random()
    var_99 = var_98.generate_string_by_mask(var_9, var_9, var_10)
    var_100 = len(var_99)
    assert var_100 == 1
    var_101 = module_0.Random()
    var_102 = var_101.generate_string_by_mask(var_10, var_9, var_10)
    var_103 = len(var_102)
    assert var_103 == 1
    var_104 = module_0.Random()
    var_105 = '!'
    var_106 = var_104.generate_string_by_mask(var_105, var_9, var_10)
    assert var_106 == '!'
    var_107 = module_0.Random()
    var_108 = '@#@#@#@#'
    var_109 = var_107.generate_string_by_mask(var_108, var_9, var_10)
    var_110 = len(var_109)
    assert var_110 == 8
    var_111 = module_0.Random()
    var_112 = '@!#@#!@#'
    var_113 = var_111.generate_string_by_mask(var_112, var_9, var_10)
    var_114 = len(var_113)
    assert var_114 == 8
    var_115 = var_113[var_26]
    var_116 = var_113[var_13]
    var_117 = var_113[var_45]
    var_118 = var_113[var_47]
    var_119 = 6
    var_120 = var_113[var_119]
    var_121 = 7
    var_122 = var_113[var_121]
    var_123 = module_0.Random()



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = 0.5
    var_4 = 0.3
    var_5 = 0.2
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.Random()
    var_8 = var_7.weighted_choice(var_6)
    var_9 = {}
    var_10 = var_7.weighted_choice(var_9)
    var_11 = 1.0
    var_12 = {var_10: var_11}
    var_13 = var_7.weighted_choice(var_12)
    assert var_13 == 'A'
    var_14 = -0.5
    var_15 = {var_10: var_14, var_1: var_3}
    var_16 = var_7.weighted_choice(var_15)
    var_17 = 0.0
    var_18 = {var_10: var_17, var_1: var_11}
    var_19 = var_7.weighted_choice(var_18)
    assert var_19 == 'B'
    var_20 = 101
    var_21 = range(var_11, var_20)
    var_22 = 100
    var_23 = {i: i / var_22 for i in var_21}
    var_24 = var_7.weighted_choice(var_23)
    var_25 = {var_10: var_17, var_1: var_17}
    var_26 = var_7.weighted_choice(var_25)
    var_27 = -1.0
    var_28 = -2.0
    var_29 = {var_10: var_27, var_1: var_28}
    var_30 = var_7.weighted_choice(var_29)
    var_31 = 'inf'
    var_32 = float(var_31)
    var_33 = {var_10: var_32, var_1: var_11}
    var_34 = var_7.weighted_choice(var_33)
    var_35 = '-inf'
    var_36 = float(var_35)
    var_37 = {var_10: var_36, var_1: var_11}
    var_38 = var_7.weighted_choice(var_37)
    var_39 = 'nan'
    var_40 = float(var_39)
    var_41 = {var_10: var_40, var_1: var_11}
    var_42 = var_7.weighted_choice(var_41)
    var_43 = float(var_31)
    var_44 = float(var_35)
    var_45 = {var_10: var_43, var_1: var_44}
    var_46 = var_7.weighted_choice(var_45)
    var_47 = float(var_31)
    var_48 = float(var_39)
    var_49 = {var_10: var_47, var_1: var_48}
    var_50 = var_7.weighted_choice(var_49)
    var_51 = float(var_35)
    var_52 = float(var_39)
    var_53 = {var_10: var_51, var_1: var_52}
    var_54 = var_7.weighted_choice(var_53)
    var_55 = float(var_31)
    var_56 = float(var_35)
    var_57 = float(var_39)
    var_58 = {var_10: var_55, var_1: var_56, var_2: var_57}
    var_59 = var_7.weighted_choice(var_58)
    var_60 = 'D'
    var_61 = float(var_31)
    var_62 = float(var_35)
    var_63 = float(var_39)
    var_64 = {var_10: var_61, var_1: var_62, var_2: var_63, var_60: var_17}
    var_65 = var_7.weighted_choice(var_64)
    var_66 = 'E'
    var_67 = float(var_31)
    var_68 = float(var_35)
    var_69 = float(var_39)
    var_70 = -1.0
    var_71 = {var_10: var_67, var_1: var_68, var_2: var_69, var_60: var_17, var_66: var_70}
    var_72 = var_7.weighted_choice(var_71)
    var_73 = 'F'
    var_74 = float(var_31)
    var_75 = float(var_35)
    var_76 = float(var_39)
    var_77 = -1.0
    var_78 = {var_10: var_74, var_1: var_75, var_2: var_76, var_60: var_17, var_66: var_77, var_73: var_11}
    var_79 = var_7.weighted_choice(var_78)
    var_80 = 'G'
    var_81 = float(var_31)
    var_82 = float(var_35)
    var_83 = float(var_39)
    var_84 = -1.0
    var_85 = 1000000.0
    var_86 = {var_10: var_81, var_1: var_82, var_2: var_83, var_60: var_17, var_66: var_84, var_73: var_11, var_80: var_85}
    var_87 = var_7.weighted_choice(var_86)
    var_88 = 'H'
    var_89 = float(var_31)
    var_90 = float(var_35)
    var_91 = float(var_39)
    var_92 = -1.0
    var_93 = 1e-07
    var_94 = {var_10: var_89, var_1: var_90, var_2: var_91, var_60: var_17, var_66: var_92, var_73: var_11, var_80: var_85, var_88: var_93}
    var_95 = var_7.weighted_choice(var_94)



# Parsed testcases at query #23
#--------------------------



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
    var_9 = -1
    var_10 = var_0.randbytes(var_9)
    var_11 = 16
    var_12 = var_0.randbytes(var_11)
    var_13 = var_0.randbytes(var_11)
    var_14 = var_0.randbytes(var_11)
    var_15 = 1000
    var_16 = var_0.randbytes(var_15)
    var_17 = len(var_16)
    assert var_17 == 1000
    var_18 = 42
    var_19 = module_0.Random(var_18)
    var_20 = module_0.Random(var_18)
    var_21 = var_19.randbytes(var_11)
    var_22 = var_20.randbytes(var_11)
    var_23 = 12345
    var_24 = module_0.Random(var_23)
    var_25 = 10
    var_26 = var_24.randbytes(var_25)
    var_27 = var_24.randbytes(var_25)
    var_28 = 'All tests passed!'
    var_29 = print(var_28)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 'abc'
    var_7 = '@###'
    var_8 = var_0.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 4
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = var_8[var_2:]
    var_13 = 2.0
    var_14 = var_0.uniform(var_2, var_13)
    var_15 = 4
    var_16 = var_0.randbytes(var_15)
    var_17 = len(var_16)
    assert var_17 == 4
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 0.5
    var_21 = {var_18: var_20, var_19: var_20}
    var_22 = var_0.weighted_choice(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'Test the randbytes method of the Random class.'
    var_1 = module_0.Random()
    var_2 = var_1.randbytes()
    var_3 = len(var_2)
    assert var_3 == 16
    var_4 = 10
    var_5 = var_1.randbytes(var_4)
    var_6 = len(var_5)
    assert var_6 == 10
    var_7 = 0
    var_8 = var_1.randbytes(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = -1
    var_11 = var_1.randbytes(var_10)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 15
    var_4 = var_0.uniform(var_1, var_2, var_3)
    var_5 = str(var_4)
    var_6 = '.'
    var_7 = var_3.split(var_6)[var_2]
    var_8 = len(var_7)
    var_9 = module_0.Random()
    var_10 = -10
    var_11 = 10
    var_12 = 5
    var_13 = var_9.uniform(var_10, var_11, var_12)
    var_14 = str(var_13)
    var_15 = var_10.split(var_6)[var_2]
    var_16 = len(var_15)
    var_17 = module_0.Random()
    var_18 = 0.5
    var_19 = var_17.uniform(var_18, var_18, var_11)
    var_20 = module_0.Random()
    var_21 = var_20.uniform(var_1, var_1, var_1)
    assert var_21 == 0
    var_22 = module_0.Random()
    var_23 = -100
    var_24 = 100
    var_25 = 20
    var_26 = var_22.uniform(var_23, var_24, var_25)
    var_27 = str(var_26)
    var_28 = var_17.split(var_6)[var_2]
    var_29 = len(var_28)
    var_30 = module_0.Random()
    var_31 = 0.0001
    var_32 = var_30.uniform(var_1, var_31, var_11)
    var_33 = str(var_32)
    var_34 = var_21.split(var_6)[var_2]
    var_35 = len(var_34)
    var_36 = module_0.Random()
    var_37 = var_36.uniform(var_1, var_2, var_1)
    var_38 = str(var_37)
    var_39 = var_24.split(var_6)[var_2]
    var_40 = len(var_39)
    var_41 = module_0.Random()
    var_42 = var_41.uniform(var_1, var_2, var_2)
    var_43 = str(var_42)
    var_44 = var_27.split(var_6)[var_2]
    var_45 = len(var_44)
    var_46 = module_0.Random()
    var_47 = 2
    var_48 = var_46.uniform(var_1, var_2, var_47)
    var_49 = str(var_48)
    var_50 = var_31.split(var_6)[var_2]
    var_51 = len(var_50)
    var_52 = module_0.Random()
    var_53 = 3
    var_54 = var_52.uniform(var_1, var_2, var_53)
    var_55 = str(var_54)
    var_56 = var_35.split(var_6)[var_2]
    var_57 = len(var_56)
    var_58 = module_0.Random()
    var_59 = 4
    var_60 = var_58.uniform(var_1, var_2, var_59)
    var_61 = str(var_60)
    var_62 = var_39.split(var_6)[var_2]
    var_63 = len(var_62)
    var_64 = module_0.Random()
    var_65 = var_64.uniform(var_1, var_2, var_12)
    var_66 = str(var_65)
    var_67 = var_42.split(var_6)[var_2]
    var_68 = len(var_67)
    var_69 = module_0.Random()
    var_70 = 6
    var_71 = var_69.uniform(var_1, var_2, var_70)
    var_72 = str(var_71)
    var_73 = var_46.split(var_6)[var_2]
    var_74 = len(var_73)
    var_75 = module_0.Random()
    var_76 = 7
    var_77 = var_75.uniform(var_1, var_2, var_76)
    var_78 = str(var_77)
    var_79 = var_50.split(var_6)[var_2]
    var_80 = len(var_79)
    var_81 = module_0.Random()
    var_82 = 8
    var_83 = var_81.uniform(var_1, var_2, var_82)
    var_84 = str(var_83)
    var_85 = var_54.split(var_6)[var_2]
    var_86 = len(var_85)
    var_87 = module_0.Random()
    var_88 = 9
    var_89 = var_87.uniform(var_1, var_2, var_88)
    var_90 = str(var_89)
    var_91 = var_58.split(var_6)[var_2]
    var_92 = len(var_91)
    var_93 = module_0.Random()
    var_94 = var_93.uniform(var_1, var_2, var_11)
    var_95 = str(var_94)
    var_96 = var_61.split(var_6)[var_2]
    var_97 = len(var_96)
    var_98 = module_0.Random()
    var_99 = 11
    var_100 = var_98.uniform(var_1, var_2, var_99)
    var_101 = str(var_100)
    var_102 = var_65.split(var_6)[var_2]
    var_103 = len(var_102)
    var_104 = module_0.Random()
    var_105 = 12
    var_106 = var_104.uniform(var_1, var_2, var_105)
    var_107 = str(var_106)
    var_108 = var_69.split(var_6)[var_2]
    var_109 = len(var_108)
    var_110 = module_0.Random()
    var_111 = 13
    var_112 = var_110.uniform(var_1, var_2, var_111)
    var_113 = str(var_112)
    var_114 = var_73.split(var_6)[var_2]
    var_115 = len(var_114)
    var_116 = module_0.Random()
    var_117 = 14
    var_118 = var_116.uniform(var_1, var_2, var_117)
    var_119 = str(var_118)
    var_120 = var_77.split(var_6)[var_2]
    var_121 = len(var_120)
    var_122 = module_0.Random()
    var_123 = var_122.uniform(var_1, var_2, var_3)
    var_124 = str(var_123)
    var_125 = var_80.split(var_6)[var_2]
    var_126 = len(var_125)
    var_127 = module_0.Random()
    var_128 = 16
    var_129 = var_127.uniform(var_1, var_2, var_128)
    var_130 = str(var_129)
    var_131 = var_84.split(var_6)[var_2]
    var_132 = len(var_131)
    var_133 = module_0.Random()
    var_134 = 17
    var_135 = var_133.uniform(var_1, var_2, var_134)
    var_136 = str(var_135)
    var_137 = var_88.split(var_6)[var_2]
    var_138 = len(var_137)
    var_139 = module_0.Random()
    var_140 = 18
    var_141 = var_139.uniform(var_1, var_2, var_140)
    var_142 = str(var_141)
    var_143 = var_92.split(var_6)[var_2]
    var_144 = len(var_143)
    var_145 = module_0.Random()
    var_146 = 19
    var_147 = var_145.uniform(var_1, var_2, var_146)
    var_148 = str(var_147)
    var_149 = var_96.split(var_6)[var_2]
    var_150 = len(var_149)
    var_151 = module_0.Random()
    var_152 = var_151.uniform(var_1, var_2, var_25)
    var_153 = str(var_152)
    var_154 = var_99.split(var_6)[var_2]
    var_155 = len(var_154)
    var_156 = module_0.Random()
    var_157 = 21
    var_158 = var_156.uniform(var_1, var_2, var_157)
    var_159 = str(var_158)
    var_160 = var_103.split(var_6)[var_2]
    var_161 = len(var_160)
    var_162 = module_0.Random()
    var_163 = 22
    var_164 = var_162.uniform(var_1, var_2, var_163)
    var_165 = str(var_164)
    var_166 = var_107.split(var_6)[var_2]
    var_167 = len(var_166)
    var_168 = module_0.Random()
    var_169 = 23
    var_170 = var_168.uniform(var_1, var_2, var_169)



