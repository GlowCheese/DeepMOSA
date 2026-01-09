####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = {}
    var_13 = ''
    var_14 = 'different'
    var_15 = 12
    var_16 = module_0.ScalarToken(var_14, var_5, var_15, var_14)
    var_17 = {var_3: var_16}
    var_18 = 'key: different'
    var_19 = 1
    var_20 = 9
    var_21 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_22 = module_0.ScalarToken(var_14, var_1, var_6, var_14)
    var_23 = {var_22: var_7}
    var_24 = 'different: value'
    var_25 = module_0.ScalarToken(var_14, var_5, var_15, var_14)
    var_26 = {var_3: var_25}
    var_27 = 'different content'
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = module_0.Token(var_0, var_1, var_2, var_3)
    var_7 = 'different'
    var_8 = module_0.Token(var_7, var_1, var_2, var_3)
    var_9 = module_0.Token(var_0, var_1, var_2, var_3)
    var_10 = module_0.Token(var_0, var_1, var_2, var_3)
    var_11 = 1
    var_12 = module_0.Token(var_0, var_11, var_2, var_3)
    var_13 = module_0.Token(var_0, var_1, var_2, var_3)
    var_14 = 5
    var_15 = module_0.Token(var_0, var_1, var_14, var_3)
    var_16 = module_0.Token(var_0, var_1, var_2, var_3)
    var_17 = 'other'
    var_18 = module_0.Token(var_17, var_1, var_2, var_3)
    var_19 = 'content1'
    var_20 = module_0.Token(var_0, var_1, var_2, var_19)
    var_21 = 'content2'
    var_22 = module_0.Token(var_0, var_1, var_2, var_21)
    var_23 = module_0.Token(var_0, var_1, var_2, var_3)
    var_24 = module_0.Token(var_0, var_11, var_14, var_3)
    var_25 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_26 = 'key'
    var_27 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_28 = {var_26: var_27}
    var_29 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_30 = {var_26: var_29}
    var_31 = module_0.ScalarToken(var_17, var_1, var_2, var_3)
    var_32 = {var_26: var_31}
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_34 = {var_26: var_33}
    var_35 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_36 = {var_17: var_35}
    var_37 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_38 = [var_37]
    var_39 = module_0.ListToken(var_38, var_1, var_2, var_3)
    var_40 = module_0.ScalarToken(var_17, var_1, var_2, var_3)
    var_41 = [var_40]
    var_42 = module_0.ListToken(var_41, var_1, var_2, var_3)
    var_43 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_44 = [var_43]
    var_45 = module_0.ListToken(var_44, var_1, var_2, var_3)
    var_46 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_47 = 9
    var_48 = module_0.ScalarToken(var_17, var_14, var_47, var_3)
    var_49 = [var_46, var_48]
    var_50 = module_0.ListToken(var_49, var_1, var_47, var_3)
    var_51 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_52 = {var_26: var_51}
    var_53 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_54 = module_0.ScalarToken(var_17, var_14, var_47, var_3)
    var_55 = {var_26: var_53, var_17: var_54}
    var_56 = 'key1'
    var_57 = 'key2'
    var_58 = 'value1'
    var_59 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_60 = 'value2'
    var_61 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_62 = {var_56: var_59, var_57: var_61}
    var_63 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_64 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_65 = {var_57: var_63, var_56: var_64}
    var_66 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_67 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_68 = [var_66, var_67]
    var_69 = module_0.ListToken(var_68, var_1, var_47, var_3)
    var_70 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_71 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_72 = [var_70, var_71]
    var_73 = module_0.ListToken(var_72, var_1, var_47, var_3)
    var_74 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_75 = [var_74]
    var_76 = module_0.ListToken(var_75, var_1, var_2, var_3)
    var_77 = {var_26: var_76}
    var_78 = module_0.ScalarToken(var_17, var_1, var_2, var_3)
    var_79 = [var_78]
    var_80 = module_0.ListToken(var_79, var_1, var_2, var_3)
    var_81 = {var_26: var_80}
    var_82 = 'nested'
    var_83 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_84 = {var_82: var_83}
    var_85 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_86 = {var_17: var_85}
    var_87 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_88 = [var_87]
    var_89 = module_0.ListToken(var_88, var_1, var_2, var_3)
    var_90 = {var_26: var_89}
    var_91 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_92 = module_0.ScalarToken(var_17, var_14, var_47, var_3)
    var_93 = [var_91, var_92]
    var_94 = module_0.ListToken(var_93, var_1, var_47, var_3)
    var_95 = {var_26: var_94}
    var_96 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_97 = {var_82: var_96}
    var_98 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_99 = module_0.ScalarToken(var_17, var_14, var_47, var_3)
    var_100 = {var_82: var_98, var_17: var_99}
    var_101 = 'nested1'
    var_102 = 'nested2'
    var_103 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_104 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_105 = {var_101: var_103, var_102: var_104}
    var_106 = module_0.ScalarToken(var_60, var_14, var_47, var_3)
    var_107 = module_0.ScalarToken(var_58, var_1, var_2, var_3)
    var_108 = {var_102: var_106, var_101: var_107}



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'other'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_8 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_9 = 'value'
    var_10 = 'key'
    var_11 = 2
    var_12 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_13 = 8
    var_14 = module_0.ScalarToken(var_0, var_2, var_13, var_0)
    var_15 = {var_12: var_14}
    var_16 = 'key value'
    var_17 = {var_12: var_14}
    var_18 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_19 = {var_18: var_14}
    var_20 = 'other value'
    var_21 = [var_12, var_14]
    var_22 = module_0.ListToken(var_21, var_1, var_13, var_16)
    var_23 = [var_12, var_14]
    var_24 = module_0.ListToken(var_23, var_1, var_13, var_16)
    var_25 = [var_18, var_14]
    var_26 = module_0.ListToken(var_25, var_1, var_13, var_20)
    var_27 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_28 = 1
    var_29 = module_0.ScalarToken(var_0, var_28, var_2, var_0)
    var_30 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_31 = 5
    var_32 = module_0.ScalarToken(var_0, var_1, var_31, var_0)
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_34 = module_0.ScalarToken(var_0, var_1, var_2, var_6)
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'different'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_8 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_9 = 1
    var_10 = module_0.ScalarToken(var_0, var_9, var_2, var_0)
    var_11 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_12 = 4
    var_13 = module_0.ScalarToken(var_0, var_1, var_12, var_0)
    var_14 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = 5
    var_18 = module_0.ScalarToken(var_16, var_1, var_17, var_16)
    var_19 = {var_15: var_18}
    var_20 = 'content'
    var_21 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_22 = 2
    var_23 = module_0.ScalarToken(var_15, var_1, var_22, var_15)
    var_24 = 8
    var_25 = module_0.ScalarToken(var_16, var_12, var_24, var_16)
    var_26 = {var_23: var_25}
    var_27 = 'key value'
    var_28 = module_0.ScalarToken(var_15, var_1, var_22, var_15)
    var_29 = module_0.ScalarToken(var_16, var_12, var_24, var_16)
    var_30 = {var_28: var_29}
    var_31 = 'item1'
    var_32 = module_0.ScalarToken(var_31, var_1, var_12, var_31)
    var_33 = 'item2'
    var_34 = 6
    var_35 = 10
    var_36 = module_0.ScalarToken(var_33, var_34, var_35, var_33)
    var_37 = [var_32, var_36]
    var_38 = 'item1 item2'
    var_39 = module_0.ListToken(var_37, var_1, var_35, var_38)
    var_40 = module_0.ScalarToken(var_31, var_1, var_12, var_31)
    var_41 = module_0.ScalarToken(var_33, var_34, var_35, var_33)
    var_42 = [var_40, var_41]
    var_43 = module_0.ListToken(var_42, var_1, var_35, var_38)
    var_44 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_45 = 'different content'
    var_46 = module_0.ScalarToken(var_0, var_1, var_2, var_45)
    var_47 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_48 = 'other test'
    var_49 = module_0.ScalarToken(var_0, var_17, var_24, var_48)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_1


def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 1
    var_11 = module_1.Position(var_10, var_10, var_1)
    var_12 = 9
    var_13 = module_1.Position(var_10, var_12, var_6)
    var_14 = [var_0]
    var_15 = [var_0]
    var_16 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_17 = {var_3: var_7}
    var_18 = {var_3: var_7}
    var_19 = 'key: value '
    var_20 = {var_3: var_7}
    var_21 = {var_3: var_7}
    var_22 = {var_3: var_7}
    var_23 = {var_3: var_7}
    var_24 = {var_3: var_7}
    var_25 = {var_3: var_7}
    var_26 = {var_3: var_7}
    var_27 = {var_3: var_7}
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}
    var_37 = {var_3: var_7}
    var_38 = {var_3: var_7}
    var_39 = {var_3: var_7}
    var_40 = {var_3: var_7}
    var_41 = {var_3: var_7}
    var_42 = {var_3: var_7}
    var_43 = {var_3: var_7}
    var_44 = {var_3: var_7}
    var_45 = {var_3: var_7}
    var_46 = {var_3: var_7}
    var_47 = {var_3: var_7}
    var_48 = {var_3: var_7}
    var_49 = {var_3: var_7}
    var_50 = {var_3: var_7}
    var_51 = {var_3: var_7}
    var_52 = {var_3: var_7}
    var_53 = {var_3: var_7}
    var_54 = {var_3: var_7}
    var_55 = {var_3: var_7}
    var_56 = {var_3: var_7}
    var_57 = {var_3: var_7}
    var_58 = {var_3: var_7}
    var_59 = {var_3: var_7}
    var_60 = {var_3: var_7}
    var_61 = {var_3: var_7}
    var_62 = {var_3: var_7}
    var_63 = {var_3: var_7}
    var_64 = {var_3: var_7}
    var_65 = {var_3: var_7}
    var_66 = {var_3: var_7}
    var_67 = {var_3: var_7}
    var_68 = {var_3: var_7}
    var_69 = {var_3: var_7}
    var_70 = {var_3: var_7}
    var_71 = {var_3: var_7}
    var_72 = {var_3: var_7}
    var_73 = {var_3: var_7}
    var_74 = {var_3: var_7}
    var_75 = {var_3: var_7}



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = "{'key': 'value'}"
    var_6 = module_0.DictToken()
    var_7 = [var_0]
    var_8 = [var_0]
    var_9 = repr(var_6)
    assert var_9 == "DictToken({'key': 'value'})"
    var_10 = {var_0: var_1}
    var_11 = module_0.DictToken()



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 5
    var_6 = 9
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 10
    var_10 = 'key: value'
    var_11 = [var_0]
    var_12 = [var_0]
    var_13 = {}
    var_14 = ''
    var_15 = {}
    var_16 = 1
    var_17 = 11
    var_18 = 'different content'
    var_19 = module_0.ScalarToken(var_4, var_1, var_9, var_10)
    var_20 = 'other_key'
    var_21 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_22 = 'other_value'
    var_23 = module_0.ScalarToken(var_22, var_5, var_6, var_22)
    var_24 = {var_21: var_23}
    var_25 = 'other_key: other_value'
    var_26 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_27 = module_0.ScalarToken(var_22, var_5, var_6, var_22)
    var_28 = {var_26: var_27}
    var_29 = 'key: other_value'
    var_30 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_31 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_32 = {var_30: var_31}
    var_33 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_34 = module_0.ScalarToken(var_22, var_5, var_6, var_22)
    var_35 = {var_33: var_34}
    var_36 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_37 = module_0.ScalarToken(var_22, var_5, var_6, var_22)
    var_38 = {var_36: var_37}
    var_39 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_40 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_41 = {var_39: var_40}
    var_42 = 'other_key: value'
    var_43 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_44 = module_0.ScalarToken(var_22, var_5, var_6, var_22)
    var_45 = {var_43: var_44}
    var_46 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_47 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_48 = {var_46: var_47}
    var_49 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_50 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_51 = {var_49: var_50}
    var_52 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_53 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_54 = {var_52: var_53}
    var_55 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_56 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_57 = {var_55: var_56}
    var_58 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_59 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_60 = {var_58: var_59}
    var_61 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_62 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_63 = {var_61: var_62}



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = {}
    var_4 = {}
    var_5 = 1
    var_6 = {}
    var_7 = {}
    var_8 = 'a'
    var_9 = {}
    var_10 = module_0.ScalarToken(var_9, var_1, var_1, var_2)
    var_11 = {}
    var_12 = module_0.ListToken(var_11, var_1, var_1, var_2)
    var_13 = 2
    var_14 = {var_5: var_13}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = {}
    var_20 = {}
    var_21 = {}
    var_22 = {}
    var_23 = {}
    var_24 = {}
    var_25 = {}
    var_26 = {}
    var_27 = {}
    var_28 = {}
    var_29 = {}
    var_30 = {}
    var_31 = {}
    var_32 = {}
    var_33 = {}
    var_34 = {}
    var_35 = {}
    var_36 = {}
    var_37 = {}
    var_38 = {}
    var_39 = {}
    var_40 = {}
    var_41 = {}
    var_42 = {}
    var_43 = {}
    var_44 = {}
    var_45 = {}
    var_46 = {}
    var_47 = {}
    var_48 = {}
    var_49 = {}
    var_50 = {}
    var_51 = {}
    var_52 = {}
    var_53 = {}
    var_54 = {}
    var_55 = {}
    var_56 = {}
    var_57 = {}
    var_58 = {}
    var_59 = {}
    var_60 = {}
    var_61 = {}
    var_62 = {}
    var_63 = {}
    var_64 = {}
    var_65 = {}
    var_66 = {}
    var_67 = {}
    var_68 = {}
    var_69 = {}
    var_70 = {}
    var_71 = {}
    var_72 = {}
    var_73 = {}
    var_74 = {}
    var_75 = {}
    var_76 = {}
    var_77 = {}
    var_78 = {}
    var_79 = {}
    var_80 = {}
    var_81 = {}
    var_82 = {}
    var_83 = {}
    var_84 = {}
    var_85 = {}
    var_86 = {}
    var_87 = {}
    var_88 = {}
    var_89 = {}
    var_90 = {}
    var_91 = {}
    var_92 = {}
    var_93 = {}
    var_94 = {}
    var_95 = {}
    var_96 = {}
    var_97 = {}
    var_98 = {}
    var_99 = {}
    var_100 = {}
    var_101 = {}
    var_102 = {}
    var_103 = {}
    var_104 = {}
    var_105 = {}
    var_106 = {}
    var_107 = {}
    var_108 = {}
    var_109 = {}
    var_110 = {}
    var_111 = {}
    var_112 = {}
    var_113 = {}
    var_114 = {}
    var_115 = {}
    var_116 = {}
    var_117 = {}
    var_118 = {}



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 1
    var_11 = module_1.Position(var_10, var_10, var_1)
    var_12 = 9
    var_13 = module_1.Position(var_10, var_12, var_6)
    var_14 = [var_0]
    var_15 = [var_0]
    var_16 = {var_3: var_7}
    var_17 = {var_3: var_7}
    var_18 = 'key: value2'
    var_19 = {var_3: var_7}
    var_20 = {var_3: var_7}
    var_21 = {var_3: var_7}
    var_22 = {var_3: var_7}
    var_23 = {var_3: var_7}
    var_24 = {var_3: var_7}
    var_25 = {var_3: var_7}
    var_26 = {var_3: var_7}
    var_27 = {var_3: var_7}
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}
    var_37 = {var_3: var_7}
    var_38 = {var_3: var_7}
    var_39 = {var_3: var_7}
    var_40 = {var_3: var_7}
    var_41 = {var_3: var_7}
    var_42 = {var_3: var_7}
    var_43 = {var_3: var_7}
    var_44 = {var_3: var_7}
    var_45 = {var_3: var_7}
    var_46 = {var_3: var_7}
    var_47 = {var_3: var_7}
    var_48 = {var_3: var_7}
    var_49 = {var_3: var_7}
    var_50 = {var_3: var_7}
    var_51 = {var_3: var_7}
    var_52 = {var_3: var_7}
    var_53 = {var_3: var_7}
    var_54 = {var_3: var_7}
    var_55 = {var_3: var_7}
    var_56 = {var_3: var_7}
    var_57 = {var_3: var_7}
    var_58 = {var_3: var_7}
    var_59 = {var_3: var_7}
    var_60 = {var_3: var_7}
    var_61 = {var_3: var_7}
    var_62 = {var_3: var_7}
    var_63 = {var_3: var_7}
    var_64 = {var_3: var_7}
    var_65 = {var_3: var_7}
    var_66 = {var_3: var_7}
    var_67 = {var_3: var_7}
    var_68 = {var_3: var_7}
    var_69 = {var_3: var_7}
    var_70 = {var_3: var_7}
    var_71 = {var_3: var_7}
    var_72 = {var_3: var_7}
    var_73 = {var_3: var_7}
    var_74 = {var_3: var_7}
    var_75 = {var_3: var_7}
    var_76 = {var_3: var_7}
    var_77 = {var_3: var_7}
    var_78 = {var_3: var_7}
    var_79 = {var_3: var_7}
    var_80 = {var_3: var_7}
    var_81 = {var_3: var_7}
    var_82 = {var_3: var_7}
    var_83 = {var_3: var_7}
    var_84 = {var_3: var_7}
    var_85 = {var_3: var_7}
    var_86 = {var_3: var_7}
    var_87 = {var_3: var_7}



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = {var_3: var_7}
    var_9 = 'key1 value1'
    var_10 = module_0.DictToken()
    var_11 = module_0.ScalarToken(var_0, var_1, var_2)
    var_12 = module_0.ScalarToken(var_4, var_5, var_6)
    var_13 = {var_11: var_12}
    var_14 = module_0.ScalarToken(var_0, var_1, var_2)
    var_15 = {var_0: var_14}
    var_16 = module_0.ScalarToken(var_4, var_5, var_6)
    var_17 = {var_0: var_16}



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 1
    var_11 = module_1.Position(var_10, var_10, var_1)
    var_12 = 9
    var_13 = module_1.Position(var_10, var_12, var_6)
    var_14 = [var_0]
    var_15 = [var_0]
    var_16 = {var_3: var_7}
    var_17 = {var_3: var_7}
    var_18 = {var_3: var_7}
    var_19 = {var_3: var_7}
    var_20 = {var_3: var_7}
    var_21 = {var_3: var_7}
    var_22 = {var_3: var_7}
    var_23 = {var_3: var_7}
    var_24 = {var_3: var_7}
    var_25 = {var_3: var_7}
    var_26 = {var_3: var_7}
    var_27 = {var_3: var_7}
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}
    var_37 = {var_3: var_7}
    var_38 = {var_3: var_7}



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = 'different'
    var_13 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_14 = 5
    var_15 = [var_0]
    var_16 = [var_0]
    var_17 = [var_0]
    var_18 = [var_0]
    var_19 = [var_0]
    var_20 = [var_0]



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 1
    var_11 = module_1.Position(var_10, var_10, var_1)
    var_12 = 9
    var_13 = module_1.Position(var_10, var_12, var_6)
    var_14 = [var_0]
    var_15 = [var_0]
    var_16 = {var_3: var_7}
    var_17 = {var_3: var_7}
    var_18 = 'key: value2'
    var_19 = {var_3: var_7}
    var_20 = {var_3: var_7}
    var_21 = {var_3: var_7}
    var_22 = {var_3: var_7}
    var_23 = {var_3: var_7}
    var_24 = {var_3: var_7}
    var_25 = {var_3: var_7}
    var_26 = {var_3: var_7}
    var_27 = {var_3: var_7}
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}
    var_37 = {var_3: var_7}
    var_38 = {var_3: var_7}
    var_39 = {var_3: var_7}
    var_40 = {var_3: var_7}
    var_41 = {var_3: var_7}
    var_42 = {var_3: var_7}
    var_43 = {var_3: var_7}
    var_44 = {var_3: var_7}
    var_45 = {var_3: var_7}
    var_46 = {var_3: var_7}
    var_47 = {var_3: var_7}
    var_48 = {var_3: var_7}
    var_49 = {var_3: var_7}
    var_50 = {var_3: var_7}
    var_51 = {var_3: var_7}
    var_52 = {var_3: var_7}
    var_53 = {var_3: var_7}
    var_54 = {var_3: var_7}
    var_55 = {var_3: var_7}
    var_56 = {var_3: var_7}
    var_57 = {var_3: var_7}
    var_58 = {var_3: var_7}
    var_59 = {var_3: var_7}
    var_60 = {var_3: var_7}
    var_61 = {var_3: var_7}
    var_62 = {var_3: var_7}
    var_63 = {var_3: var_7}
    var_64 = {var_3: var_7}
    var_65 = {var_3: var_7}
    var_66 = {var_3: var_7}
    var_67 = {var_3: var_7}
    var_68 = {var_3: var_7}
    var_69 = {var_3: var_7}
    var_70 = {var_3: var_7}
    var_71 = {var_3: var_7}
    var_72 = {var_3: var_7}
    var_73 = {var_3: var_7}
    var_74 = {var_3: var_7}
    var_75 = {var_3: var_7}
    var_76 = {var_3: var_7}
    var_77 = {var_3: var_7}
    var_78 = {var_3: var_7}
    var_79 = {var_3: var_7}
    var_80 = {var_3: var_7}
    var_81 = {var_3: var_7}
    var_82 = {var_3: var_7}
    var_83 = {var_3: var_7}
    var_84 = {var_3: var_7}
    var_85 = {var_3: var_7}
    var_86 = {var_3: var_7}
    var_87 = {var_3: var_7}



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'other'
    var_7 = 5
    var_8 = 9
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_11 = 'value'
    var_12 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_13 = 1
    var_14 = module_0.ScalarToken(var_0, var_13, var_7, var_0)
    var_15 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_16 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_17 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_18 = 'different'
    var_19 = module_0.ScalarToken(var_0, var_1, var_2, var_18)
    var_20 = 'key'
    var_21 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_22 = {var_20: var_21}
    var_23 = 10
    var_24 = 'key: value'
    var_25 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_26 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_27 = [var_26]
    var_28 = '[value]'
    var_29 = module_0.ListToken(var_27, var_1, var_23, var_28)
    var_30 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_31 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_32 = {var_20: var_31}
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_34 = [var_33]
    var_35 = module_0.ListToken(var_34, var_1, var_23, var_28)
    var_36 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_37 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_38 = {var_20: var_37}
    var_39 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_40 = {var_20: var_39}
    var_41 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_42 = {var_20: var_41}
    var_43 = 'key: other'
    var_44 = 'key1'
    var_45 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_46 = {var_44: var_45}
    var_47 = 'key1: value'
    var_48 = 'key2'
    var_49 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_50 = {var_48: var_49}
    var_51 = 'key2: value'
    var_52 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_53 = {var_44: var_52}
    var_54 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_55 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_56 = {var_44: var_54, var_48: var_55}
    var_57 = 20
    var_58 = 'key1: value, key2: value'
    var_59 = 'value1'
    var_60 = module_0.ScalarToken(var_59, var_1, var_7, var_59)
    var_61 = 'value2'
    var_62 = 7
    var_63 = 12
    var_64 = module_0.ScalarToken(var_61, var_62, var_63, var_61)
    var_65 = {var_44: var_60, var_48: var_64}
    var_66 = 'key1: value1, key2: value2'
    var_67 = module_0.ScalarToken(var_61, var_62, var_63, var_61)
    var_68 = module_0.ScalarToken(var_59, var_1, var_7, var_59)
    var_69 = {var_48: var_67, var_44: var_68}
    var_70 = 'key2: value2, key1: value1'
    var_71 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_72 = {var_20: var_71}
    var_73 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_74 = [var_73]
    var_75 = module_0.ListToken(var_74, var_1, var_23, var_28)
    var_76 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_77 = {var_20: var_76}
    var_78 = module_0.ScalarToken(var_0, var_13, var_7, var_0)
    var_79 = {var_20: var_78}
    var_80 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_81 = {var_20: var_80}
    var_82 = module_0.ScalarToken(var_0, var_1, var_2, var_18)
    var_83 = {var_20: var_82}
    var_84 = 'key: different'
    var_85 = module_0.ScalarToken(var_59, var_1, var_7, var_59)
    var_86 = {var_20: var_85}
    var_87 = 'key: value1'
    var_88 = module_0.ScalarToken(var_61, var_1, var_7, var_61)
    var_89 = {var_20: var_88}
    var_90 = 'key: value2'
    var_91 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_92 = {var_20: var_91}
    var_93 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_94 = [var_93]
    var_95 = module_0.ListToken(var_94, var_1, var_23, var_28)
    var_96 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_97 = {var_20: var_96}
    var_98 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_99 = {var_20: var_98}
    var_100 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_101 = {var_20: var_100}
    var_102 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_103 = {var_20: var_102}



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'different'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_0)
    var_8 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_9 = 1
    var_10 = module_0.ScalarToken(var_0, var_9, var_2, var_0)
    var_11 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_12 = 4
    var_13 = module_0.ScalarToken(var_0, var_1, var_12, var_0)
    var_14 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = 5
    var_18 = module_0.ScalarToken(var_16, var_1, var_17, var_16)
    var_19 = {var_15: var_18}
    var_20 = 'content'
    var_21 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_22 = module_0.ScalarToken(var_16, var_1, var_17, var_16)
    var_23 = {var_15: var_22}
    var_24 = module_0.ScalarToken(var_16, var_1, var_17, var_16)
    var_25 = {var_15: var_24}
    var_26 = 'item'
    var_27 = module_0.ScalarToken(var_26, var_1, var_12, var_26)
    var_28 = [var_27]
    var_29 = module_0.ListToken(var_28, var_1, var_12, var_20)
    var_30 = module_0.ScalarToken(var_26, var_1, var_12, var_26)
    var_31 = [var_30]
    var_32 = module_0.ListToken(var_31, var_1, var_12, var_20)
    var_33 = ''
    var_34 = module_0.ScalarToken(var_33, var_1, var_1, var_33)
    var_35 = module_0.ScalarToken(var_33, var_1, var_1, var_33)
    var_36 = '\n\t'
    var_37 = 2
    var_38 = module_0.ScalarToken(var_36, var_1, var_37, var_36)
    var_39 = module_0.ScalarToken(var_36, var_1, var_37, var_36)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 14
    var_6 = 18
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 6
    var_10 = 12
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 20
    var_14 = 26
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = module_0.DictToken()
    var_19 = var_18.value
    var_20 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_21 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_22 = {var_0: var_20, var_4: var_21}
    var_23 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_24 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_25 = {var_0: var_23, var_4: var_24}
    var_26 = [var_0]
    var_27 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_28 = [var_0]
    var_29 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_30 = repr(var_18)
    assert var_30 == "DictToken('key1: value1, key2: value2')"
    var_31 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_32 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_33 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_34 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.DictToken()
    var_37 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_38 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_39 = {var_37: var_38}
    var_40 = 'key1: value1'
    var_41 = module_0.DictToken()
    var_42 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_43 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_44 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_45 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_46 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = 1
    var_49 = module_0.DictToken()
    var_50 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_51 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_52 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_53 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = 25
    var_56 = module_0.DictToken()
    var_57 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_58 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_59 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_60 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = 'key1: value1, key2: value2, key3: value3'
    var_63 = module_0.DictToken()
    var_64 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_65 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_66 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_67 = 'value3'
    var_68 = module_0.ScalarToken(var_67, var_13, var_14, var_67)
    var_69 = {var_64: var_66, var_65: var_68}
    var_70 = 'key1: value1, key2: value3'
    var_71 = module_0.DictToken()
    var_72 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_73 = 'key3'
    var_74 = module_0.ScalarToken(var_73, var_5, var_6, var_73)
    var_75 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_76 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_77 = {var_72: var_75, var_74: var_76}
    var_78 = 'key1: value1, key3: value2'
    var_79 = module_0.DictToken()
    var_80 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_81 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_82 = {var_80: var_81}
    var_83 = module_0.DictToken()
    var_84 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_85 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_86 = 28
    var_87 = 32
    var_88 = module_0.ScalarToken(var_73, var_86, var_87, var_73)
    var_89 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_90 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_91 = 34
    var_92 = 40
    var_93 = module_0.ScalarToken(var_67, var_91, var_92, var_67)
    var_94 = {var_84: var_89, var_85: var_90, var_88: var_93}
    var_95 = module_0.DictToken()
    var_96 = module_0.ScalarToken(var_4, var_1, var_2, var_4)
    var_97 = module_0.ScalarToken(var_0, var_5, var_6, var_0)
    var_98 = module_0.ScalarToken(var_12, var_9, var_10, var_12)
    var_99 = module_0.ScalarToken(var_8, var_13, var_14, var_8)
    var_100 = {var_96: var_98, var_97: var_99}
    var_101 = 'key2: value2, key1: value1'
    var_102 = module_0.DictToken()
    var_103 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_104 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_105 = module_0.ScalarToken(var_12, var_9, var_10, var_12)
    var_106 = module_0.ScalarToken(var_8, var_13, var_14, var_8)
    var_107 = {var_103: var_105, var_104: var_106}
    var_108 = 'key1: value2, key2: value1'
    var_109 = module_0.DictToken()



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'other'
    var_7 = 5
    var_8 = 9
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_11 = 'value'
    var_12 = 'key'
    var_13 = 2
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 8
    var_16 = module_0.ScalarToken(var_0, var_2, var_15, var_0)
    var_17 = {var_14: var_16}
    var_18 = 'key: value'
    var_19 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_20 = module_0.ScalarToken(var_0, var_2, var_15, var_0)
    var_21 = {var_19: var_20}
    var_22 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_23 = module_0.ScalarToken(var_0, var_2, var_15, var_0)
    var_24 = {var_22: var_23}
    var_25 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_26 = 'data'
    var_27 = 6
    var_28 = module_0.ScalarToken(var_26, var_27, var_8, var_26)
    var_29 = {var_25: var_28}
    var_30 = 'other: data'
    var_31 = 'item'
    var_32 = 3
    var_33 = module_0.ScalarToken(var_31, var_1, var_32, var_31)
    var_34 = [var_33]
    var_35 = module_0.ListToken(var_34, var_1, var_32, var_31)
    var_36 = module_0.ScalarToken(var_31, var_1, var_32, var_31)
    var_37 = [var_36]
    var_38 = module_0.ListToken(var_37, var_1, var_32, var_31)
    var_39 = module_0.ScalarToken(var_31, var_1, var_32, var_31)
    var_40 = [var_39]
    var_41 = module_0.ListToken(var_40, var_1, var_32, var_31)
    var_42 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_43 = [var_42]
    var_44 = module_0.ListToken(var_43, var_1, var_2, var_6)
    var_45 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_46 = 1
    var_47 = module_0.ScalarToken(var_0, var_46, var_7, var_0)
    var_48 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_49 = module_0.ScalarToken(var_0, var_1, var_7, var_0)
    var_50 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_51 = module_0.ScalarToken(var_0, var_1, var_2, var_6)
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = "{'key1': 'value1', 'key2': 'value2'}"
    var_8 = module_0.DictToken()
    var_9 = 1
    var_10 = module_1.Position(var_9, var_9, var_5)
    var_11 = 11
    var_12 = module_1.Position(var_9, var_11, var_6)
    var_13 = [var_0]
    var_14 = [var_1]
    var_15 = [var_0]
    var_16 = [var_1]
    var_17 = repr(var_8)
    assert var_17 == "DictToken({'key1': 'value1', 'key2': 'value2'})"
    var_18 = {var_0: var_2, var_1: var_3}
    var_19 = module_0.DictToken()
    var_20 = 'value3'
    var_21 = {var_0: var_2, var_1: var_20}
    var_22 = "{'key1': 'value1', 'key2': 'value3'}"
    var_23 = module_0.DictToken()
    var_24 = {var_0: var_2, var_1: var_3}
    var_25 = module_0.DictToken()
    var_26 = {var_0: var_2, var_1: var_3}
    var_27 = module_0.DictToken()
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = "{'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}"
    var_30 = module_0.DictToken()
    var_31 = 'key3'
    var_32 = {var_0: var_2, var_1: var_3, var_31: var_20}
    var_33 = module_0.DictToken()
    var_34 = {var_0: var_2, var_31: var_3}
    var_35 = "{'key1': 'value1', 'key3': 'value2'}"
    var_36 = module_0.DictToken()
    var_37 = {var_0: var_2, var_1: var_20}
    var_38 = module_0.DictToken()
    var_39 = 'value'
    var_40 = 4
    var_41 = module_0.ScalarToken(var_39, var_5, var_40, var_39)
    var_42 = [var_2, var_3]
    var_43 = "['value1', 'value2']"
    var_44 = module_0.ListToken(var_42, var_5, var_6, var_43)
    var_45 = {}
    var_46 = '{}'
    var_47 = module_0.DictToken()
    var_48 = ''
    var_49 = module_0.ScalarToken(var_48, var_5, var_5, var_48)
    var_50 = []
    var_51 = '[]'
    var_52 = module_0.ListToken(var_50, var_5, var_5, var_51)
    var_53 = None
    var_54 = 'None'
    var_55 = module_0.ScalarToken(var_53, var_5, var_5, var_54)
    var_56 = True
    var_57 = 'True'
    var_58 = module_0.ScalarToken(var_56, var_5, var_5, var_57)
    var_59 = '1'
    var_60 = module_0.ScalarToken(var_56, var_5, var_5, var_59)
    var_61 = '1.0'
    var_62 = module_0.ScalarToken(var_56, var_5, var_5, var_61)
    var_63 = '1j'
    var_64 = b'value'
    var_65 = "b'value'"
    var_66 = module_0.ScalarToken(var_64, var_5, var_5, var_65)
    var_67 = bytearray(var_64)
    var_68 = "bytearray(b'value')"
    var_69 = module_0.ScalarToken(var_67, var_5, var_5, var_68)
    var_70 = memoryview(var_64)
    var_71 = "memoryview(b'value')"
    var_72 = module_0.ScalarToken(var_70, var_5, var_5, var_71)
    var_73 = range(var_56)
    var_74 = 'range(0, 1)'
    var_75 = module_0.ScalarToken(var_73, var_5, var_5, var_74)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 9
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 'key1'
    var_11 = 3
    var_12 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_13 = 'value1'
    var_14 = 5
    var_15 = 11
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = 'key2'
    var_18 = 13
    var_19 = 16
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_17)
    var_21 = 'value2'
    var_22 = 18
    var_23 = 24
    var_24 = module_0.ScalarToken(var_21, var_22, var_23, var_21)
    var_25 = {var_12: var_16, var_20: var_24}
    var_26 = 'key1: value1, key2: value2'
    var_27 = {}
    var_28 = ''
    var_29 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_30 = 'nested_key'
    var_31 = module_0.ScalarToken(var_30, var_5, var_18, var_30)
    var_32 = 'nested_value'
    var_33 = 15
    var_34 = 26
    var_35 = module_0.ScalarToken(var_32, var_33, var_34, var_32)
    var_36 = {var_31: var_35}
    var_37 = 'nested_key: nested_value'
    var_38 = 'key: nested_key: nested_value'
    var_39 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_40 = 'item1'
    var_41 = 8
    var_42 = module_0.ScalarToken(var_40, var_5, var_41, var_40)
    var_43 = 'item2'
    var_44 = 10
    var_45 = 14
    var_46 = module_0.ScalarToken(var_43, var_44, var_45, var_43)
    var_47 = [var_42, var_46]
    var_48 = 'item1, item2'
    var_49 = module_0.ListToken(var_47, var_5, var_45, var_48)
    var_50 = {var_39: var_49}
    var_51 = 'key: item1, item2'
    var_52 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_53 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_54 = {var_52: var_53}
    var_55 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_56 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_57 = module_0.ScalarToken(var_17, var_18, var_19, var_17)
    var_58 = 22
    var_59 = module_0.ScalarToken(var_40, var_22, var_58, var_40)
    var_60 = 28
    var_61 = module_0.ScalarToken(var_43, var_23, var_60, var_43)
    var_62 = [var_59, var_61]
    var_63 = module_0.ListToken(var_62, var_22, var_60, var_48)
    var_64 = {var_55: var_56, var_57: var_63}
    var_65 = 'key1: value1, key2: item1, item2'
    var_66 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_67 = module_0.ScalarToken(var_13, var_5, var_44, var_13)
    var_68 = 12
    var_69 = module_0.ScalarToken(var_0, var_68, var_45, var_0)
    var_70 = module_0.ScalarToken(var_21, var_19, var_58, var_21)
    var_71 = {var_66: var_67, var_69: var_70}
    var_72 = 'key: value1, key: value2'
    var_73 = 123
    var_74 = '123'
    var_75 = module_0.ScalarToken(var_73, var_1, var_2, var_74)
    var_76 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_77 = {var_75: var_76}
    var_78 = '123: value'
    var_79 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_80 = None
    var_81 = 'null'
    var_82 = module_0.ScalarToken(var_80, var_5, var_41, var_81)
    var_83 = {var_79: var_82}
    var_84 = 'key: null'
    var_85 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_86 = True
    var_87 = 'true'
    var_88 = module_0.ScalarToken(var_86, var_5, var_41, var_87)
    var_89 = {var_85: var_88}
    var_90 = 'key: true'



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = 1
    var_11 = module_1.Position(var_10, var_10, var_1)
    var_12 = 9
    var_13 = module_1.Position(var_10, var_12, var_6)
    var_14 = [var_0]
    var_15 = [var_0]
    var_16 = {var_3: var_7}
    var_17 = {var_3: var_7}
    var_18 = 'key: value2'
    var_19 = {var_3: var_7}
    var_20 = {var_3: var_7}
    var_21 = {var_3: var_7}
    var_22 = {var_3: var_7}
    var_23 = {var_3: var_7}
    var_24 = {var_3: var_7}
    var_25 = {var_3: var_7}
    var_26 = {var_3: var_7}
    var_27 = {var_3: var_7}
    var_28 = {var_3: var_7}
    var_29 = {var_3: var_7}
    var_30 = {var_3: var_7}
    var_31 = {var_3: var_7}
    var_32 = {var_3: var_7}
    var_33 = {var_3: var_7}
    var_34 = {var_3: var_7}
    var_35 = {var_3: var_7}
    var_36 = {var_3: var_7}
    var_37 = {var_3: var_7}
    var_38 = {var_3: var_7}
    var_39 = {var_3: var_7}
    var_40 = {var_3: var_7}
    var_41 = {var_3: var_7}
    var_42 = {var_3: var_7}
    var_43 = {var_3: var_7}
    var_44 = {var_3: var_7}
    var_45 = {var_3: var_7}
    var_46 = {var_3: var_7}
    var_47 = {var_3: var_7}
    var_48 = {var_3: var_7}
    var_49 = {var_3: var_7}
    var_50 = {var_3: var_7}
    var_51 = {var_3: var_7}
    var_52 = {var_3: var_7}
    var_53 = {var_3: var_7}
    var_54 = {var_3: var_7}
    var_55 = {var_3: var_7}
    var_56 = {var_3: var_7}
    var_57 = {var_3: var_7}
    var_58 = {var_3: var_7}
    var_59 = {var_3: var_7}
    var_60 = {var_3: var_7}
    var_61 = {var_3: var_7}
    var_62 = {var_3: var_7}
    var_63 = {var_3: var_7}
    var_64 = {var_3: var_7}
    var_65 = {var_3: var_7}
    var_66 = {var_3: var_7}
    var_67 = {var_3: var_7}
    var_68 = {var_3: var_7}
    var_69 = {var_3: var_7}
    var_70 = {var_3: var_7}
    var_71 = {var_3: var_7}
    var_72 = {var_3: var_7}
    var_73 = {var_3: var_7}
    var_74 = {var_3: var_7}
    var_75 = {var_3: var_7}
    var_76 = {var_3: var_7}
    var_77 = {var_3: var_7}
    var_78 = {var_3: var_7}
    var_79 = {var_3: var_7}
    var_80 = {var_3: var_7}
    var_81 = {var_3: var_7}
    var_82 = {var_3: var_7}
    var_83 = {var_3: var_7}
    var_84 = {var_3: var_7}
    var_85 = {var_3: var_7}
    var_86 = {var_3: var_7}



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 12
    var_6 = 15
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 5
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_19 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_20 = {var_0: var_18, var_4: var_19}
    var_21 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_22 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_23 = {var_0: var_21, var_4: var_22}
    var_24 = [var_0]
    var_25 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_26 = [var_4]
    var_27 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_28 = [var_0]
    var_29 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_30 = [var_4]
    var_31 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_32 = 'key3'
    var_33 = module_0.ScalarToken(var_32, var_1, var_2, var_32)
    var_34 = 'value3'
    var_35 = module_0.ScalarToken(var_34, var_9, var_10, var_34)
    var_36 = {var_33: var_35}
    var_37 = 'key3: value3'
    var_38 = 'All tests passed!'
    var_39 = print(var_38)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = {}
    var_13 = ''
    var_14 = {}
    var_15 = 1
    var_16 = 9
    var_17 = 'different content'
    var_18 = module_0.ScalarToken(var_4, var_1, var_6, var_9)
    var_19 = 'different key'
    var_20 = module_0.ScalarToken(var_19, var_1, var_2, var_19)
    var_21 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_22 = {var_20: var_21}
    var_23 = 'different key: value'
    var_24 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_25 = 'different value'
    var_26 = module_0.ScalarToken(var_25, var_5, var_6, var_25)
    var_27 = {var_24: var_26}
    var_28 = 'key: different value'
    var_29 = module_0.ScalarToken(var_19, var_1, var_2, var_19)
    var_30 = module_0.ScalarToken(var_25, var_5, var_6, var_25)
    var_31 = {var_29: var_30}
    var_32 = 'different key: different value'
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_34 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_35 = {var_33: var_34}
    var_36 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_37 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_38 = {var_36: var_37}
    var_39 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_40 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_41 = {var_39: var_40}
    var_42 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_43 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_44 = {var_42: var_43}
    var_45 = module_0.ScalarToken(var_4, var_1, var_6, var_9)
    var_46 = module_0.ScalarToken(var_19, var_1, var_2, var_19)
    var_47 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_48 = {var_46: var_47}
    var_49 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_50 = module_0.ScalarToken(var_25, var_5, var_6, var_25)
    var_51 = {var_49: var_50}
    var_52 = module_0.ScalarToken(var_19, var_1, var_2, var_19)
    var_53 = module_0.ScalarToken(var_25, var_5, var_6, var_25)
    var_54 = {var_52: var_53}
    var_55 = module_0.ScalarToken(var_19, var_1, var_2, var_19)
    var_56 = module_0.ScalarToken(var_25, var_5, var_6, var_25)
    var_57 = {var_55: var_56}
    var_58 = module_0.ScalarToken(var_19, var_1, var_2, var_19)



