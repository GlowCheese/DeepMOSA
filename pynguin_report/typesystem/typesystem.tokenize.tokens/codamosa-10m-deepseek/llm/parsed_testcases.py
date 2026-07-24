####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "value"}'
    var_6 = [var_0]
    var_7 = [var_0]



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 1
    var_6 = 4
    var_7 = ' test'
    var_8 = module_0.Token(var_0, var_5, var_6, var_7)
    var_9 = module_0.Token(var_0, var_1, var_2, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 'value1'
    var_6 = 5
    var_7 = module_0.Token(var_5, var_1, var_6, var_5)
    var_8 = 'value2'
    var_9 = module_0.Token(var_8, var_1, var_6, var_8)
    var_10 = module_0.Token(var_0, var_1, var_2, var_0)
    var_11 = 'value'
    var_12 = module_0.Token(var_0, var_1, var_2, var_0)
    var_13 = 1
    var_14 = module_0.Token(var_0, var_13, var_6, var_0)
    var_15 = 'content1'
    var_16 = module_0.Token(var_0, var_1, var_2, var_15)
    var_17 = 'content2'
    var_18 = module_0.Token(var_0, var_1, var_2, var_17)
    var_19 = module_0.Token(var_5, var_1, var_6, var_5)
    var_20 = module_0.Token(var_8, var_1, var_6, var_8)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 1
    var_6 = 4
    var_7 = module_0.Token(var_0, var_5, var_6, var_0)
    var_8 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_9 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_10 = module_0.ScalarToken(var_0, var_5, var_6, var_0)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.DictToken()
    var_15 = {var_11: var_12}
    var_16 = module_0.DictToken()
    var_17 = {var_11: var_12}
    var_18 = module_0.DictToken()
    var_19 = [var_12]
    var_20 = module_0.ListToken(var_19, var_1, var_2, var_0)
    var_21 = [var_12]
    var_22 = module_0.ListToken(var_21, var_1, var_2, var_0)
    var_23 = [var_12]
    var_24 = module_0.ListToken(var_23, var_5, var_6, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1\nkey2: value2'
    var_20 = 1
    var_21 = module_1.Position(var_20, var_20, var_1)
    var_22 = 2
    var_23 = 12
    var_24 = module_1.Position(var_22, var_23, var_16)
    var_25 = [var_0]
    var_26 = [var_9]
    var_27 = [var_0]
    var_28 = [var_9]



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 'a'
    var_2 = 1
    var_3 = 2
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 5
    var_6 = module_0.ScalarToken(var_2, var_5, var_5, var_0)
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 12
    var_10 = [var_1]
    var_11 = [var_1]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 14
    var_12 = {}
    var_13 = 2
    var_14 = '{}'
    var_15 = None



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 5
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'key2'
    var_6 = 16
    var_7 = 20
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value1'
    var_10 = 8
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 23
    var_15 = 29
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_12, var_8: var_16}
    var_18 = 0
    var_19 = 30



# Parsed testcases at query #13
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
    var_10 = 'key1'
    var_11 = 3
    var_12 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_13 = 'value1'
    var_14 = 5
    var_15 = 10
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = 'key2'
    var_18 = 12
    var_19 = 15
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_17)
    var_21 = 'value2'
    var_22 = 17
    var_23 = 22
    var_24 = module_0.ScalarToken(var_21, var_22, var_23, var_21)
    var_25 = {var_12: var_16, var_20: var_24}
    var_26 = 'key1: value1, key2: value2'
    var_27 = ''
    var_28 = module_0.ScalarToken(var_0, var_1, var_2, var_27)
    var_29 = module_0.ScalarToken(var_4, var_5, var_6, var_27)
    var_30 = {var_28: var_29}
    var_31 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_32 = None
    var_33 = 7
    var_34 = 'null'
    var_35 = module_0.ScalarToken(var_32, var_5, var_33, var_34)
    var_36 = {var_31: var_35}
    var_37 = 'key: null'
    var_38 = {}



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 6
    var_6 = 12
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key1:value1'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0
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
    var_14 = {}
    var_15 = ''
    var_16 = module_1.Position(var_10, var_10, var_1)
    var_17 = module_1.Position(var_10, var_10, var_1)
    var_18 = 'key1'
    var_19 = 3
    var_20 = module_0.ScalarToken(var_18, var_1, var_19, var_18)
    var_21 = 'value1'
    var_22 = 5
    var_23 = 10
    var_24 = module_0.ScalarToken(var_21, var_22, var_23, var_21)
    var_25 = 'key2'
    var_26 = 12
    var_27 = 15
    var_28 = module_0.ScalarToken(var_25, var_26, var_27, var_25)
    var_29 = 'value2'
    var_30 = 17
    var_31 = 22
    var_32 = module_0.ScalarToken(var_29, var_30, var_31, var_29)
    var_33 = {var_20: var_24, var_28: var_32}
    var_34 = 'key1: value1, key2: value2'
    var_35 = module_1.Position(var_10, var_10, var_1)
    var_36 = 23
    var_37 = module_1.Position(var_10, var_36, var_31)
    var_38 = 'nested'
    var_39 = module_0.ScalarToken(var_38, var_1, var_22, var_38)
    var_40 = 'item'
    var_41 = 7
    var_42 = module_0.ScalarToken(var_40, var_41, var_23, var_40)
    var_43 = [var_42]
    var_44 = module_0.ListToken(var_43, var_41, var_23, var_40)
    var_45 = {var_39: var_44}
    var_46 = 'nested: item'
    var_47 = module_1.Position(var_10, var_10, var_1)
    var_48 = 11
    var_49 = module_1.Position(var_10, var_48, var_23)
    var_50 = 'complex'
    var_51 = 6
    var_52 = module_0.ScalarToken(var_50, var_1, var_51, var_50)
    var_53 = module_0.ScalarToken(var_0, var_6, var_23, var_0)
    var_54 = 16
    var_55 = module_0.ScalarToken(var_4, var_26, var_54, var_4)
    var_56 = {var_53: var_55}
    var_57 = 'complex: key: value'
    var_58 = module_1.Position(var_10, var_10, var_1)
    var_59 = module_1.Position(var_10, var_30, var_54)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'some content'



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 5
    var_5 = '{"key": "value"}'
    var_6 = 1
    var_7 = module_0.Position(var_6, var_6, var_3)
    var_8 = 18
    var_9 = 17
    var_10 = module_0.Position(var_6, var_8, var_9)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 5
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = {var_4: var_8}
    var_13 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_14 = {var_4: var_8}
    var_15 = 8
    var_16 = {var_4: var_8}
    var_17 = 1
    var_18 = {var_4: var_8}
    var_19 = 'key: value2'
    var_20 = {var_4: var_8}
    var_21 = {var_4: var_8}
    var_22 = {var_4: var_8}
    var_23 = {var_4: var_8}
    var_24 = {var_4: var_8}
    var_25 = {var_4: var_8}
    var_26 = {var_4: var_8}
    var_27 = {var_4: var_8}
    var_28 = {var_4: var_8}
    var_29 = {var_4: var_8}
    var_30 = {var_4: var_8}
    var_31 = {var_4: var_8}
    var_32 = {var_4: var_8}
    var_33 = {var_4: var_8}
    var_34 = {var_4: var_8}
    var_35 = {var_4: var_8}
    var_36 = {var_4: var_8}
    var_37 = {var_4: var_8}
    var_38 = {var_4: var_8}
    var_39 = {var_4: var_8}
    var_40 = {var_4: var_8}
    var_41 = {var_4: var_8}
    var_42 = {var_4: var_8}
    var_43 = {var_4: var_8}
    var_44 = {var_4: var_8}
    var_45 = {var_4: var_8}
    var_46 = {var_4: var_8}
    var_47 = {var_4: var_8}
    var_48 = {var_4: var_8}
    var_49 = {var_4: var_8}
    var_50 = {var_4: var_8}
    var_51 = {var_4: var_8}
    var_52 = {var_4: var_8}
    var_53 = {var_4: var_8}
    var_54 = {var_4: var_8}
    var_55 = {var_4: var_8}
    var_56 = {var_4: var_8}
    var_57 = {var_4: var_8}
    var_58 = {var_4: var_8}
    var_59 = {var_4: var_8}
    var_60 = {var_4: var_8}
    var_61 = {var_4: var_8}
    var_62 = {var_4: var_8}
    var_63 = {var_4: var_8}
    var_64 = {var_4: var_8}
    var_65 = {var_4: var_8}
    var_66 = {var_4: var_8}
    var_67 = {var_4: var_8}
    var_68 = {var_4: var_8}
    var_69 = {var_4: var_8}
    var_70 = {var_4: var_8}
    var_71 = {var_4: var_8}
    var_72 = {var_4: var_8}
    var_73 = {var_4: var_8}
    var_74 = {var_4: var_8}
    var_75 = {var_4: var_8}
    var_76 = {var_4: var_8}
    var_77 = {var_4: var_8}
    var_78 = {var_4: var_8}
    var_79 = {var_4: var_8}
    var_80 = {var_4: var_8}
    var_81 = {var_4: var_8}
    var_82 = {var_4: var_8}
    var_83 = {var_4: var_8}
    var_84 = {var_4: var_8}
    var_85 = {var_4: var_8}
    var_86 = {var_4: var_8}



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 'key'
    var_3 = 2
    var_4 = module_0.ScalarToken(var_2, var_1, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7)
    var_9 = {var_4: var_8}
    var_10 = 'key1'
    var_11 = 3
    var_12 = module_0.ScalarToken(var_10, var_1, var_11)
    var_13 = 'value1'
    var_14 = 5
    var_15 = 10
    var_16 = module_0.ScalarToken(var_13, var_14, var_15)
    var_17 = 'key2'
    var_18 = 12
    var_19 = 15
    var_20 = module_0.ScalarToken(var_17, var_18, var_19)
    var_21 = 'value2'
    var_22 = 17
    var_23 = 22
    var_24 = module_0.ScalarToken(var_21, var_22, var_23)
    var_25 = {var_12: var_16, var_20: var_24}
    var_26 = 'nested_key'
    var_27 = 9
    var_28 = module_0.ScalarToken(var_26, var_1, var_27)
    var_29 = 'nested_value'
    var_30 = 11
    var_31 = module_0.ScalarToken(var_29, var_30, var_23)
    var_32 = {var_28: var_31}
    var_33 = 'outer_key'
    var_34 = 24
    var_35 = 32
    var_36 = module_0.ScalarToken(var_33, var_34, var_35)
    var_37 = 'All test cases pass'
    var_38 = print(var_37)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = module_0.DictToken()



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 1
    var_4 = module_0.Position(var_3, var_3, var_1)
    var_5 = module_0.Position(var_3, var_3, var_1)
    var_6 = '{"key": "value"}'
    var_7 = 'key'
    var_8 = 3
    var_9 = module_1.ScalarToken(var_7, var_3, var_8, var_6)
    var_10 = 'value'
    var_11 = 7
    var_12 = 11
    var_13 = module_1.ScalarToken(var_10, var_11, var_12, var_6)
    var_14 = {var_9: var_13}
    var_15 = 12
    var_16 = module_0.Position(var_3, var_3, var_1)
    var_17 = 13
    var_18 = module_0.Position(var_3, var_17, var_15)
    var_19 = '{"key": {"nested_key": "nested_value"}}'
    var_20 = 'nested_key'
    var_21 = 9
    var_22 = 18
    var_23 = module_1.ScalarToken(var_20, var_21, var_22, var_19)
    var_24 = 'nested_value'
    var_25 = 22
    var_26 = 33
    var_27 = module_1.ScalarToken(var_24, var_25, var_26, var_19)
    var_28 = {var_23: var_27}
    var_29 = 34
    var_30 = module_1.ScalarToken(var_7, var_3, var_8, var_19)
    var_31 = 35
    var_32 = module_0.Position(var_3, var_3, var_1)
    var_33 = 36
    var_34 = module_0.Position(var_3, var_33, var_31)
    var_35 = '{"key1": "value1", "key2": "value2"}'
    var_36 = 'key1'
    var_37 = 4
    var_38 = module_1.ScalarToken(var_36, var_3, var_37, var_35)
    var_39 = 'value1'
    var_40 = 8
    var_41 = module_1.ScalarToken(var_39, var_40, var_17, var_35)
    var_42 = 'key2'
    var_43 = 16
    var_44 = 19
    var_45 = module_1.ScalarToken(var_42, var_43, var_44, var_35)
    var_46 = 'value2'
    var_47 = 23
    var_48 = 28
    var_49 = module_1.ScalarToken(var_46, var_47, var_48, var_35)
    var_50 = {var_38: var_41, var_45: var_49}
    var_51 = 29
    var_52 = module_0.Position(var_3, var_3, var_1)
    var_53 = 30
    var_54 = module_0.Position(var_3, var_53, var_51)
    var_55 = '{1: "one", 2: "two"}'
    var_56 = module_1.ScalarToken(var_3, var_3, var_3, var_55)
    var_57 = 'one'
    var_58 = 5
    var_59 = module_1.ScalarToken(var_57, var_58, var_11, var_55)
    var_60 = 2
    var_61 = 10
    var_62 = module_1.ScalarToken(var_60, var_61, var_61, var_55)
    var_63 = 'two'
    var_64 = 14
    var_65 = module_1.ScalarToken(var_63, var_64, var_43, var_55)
    var_66 = {var_56: var_59, var_62: var_65}
    var_67 = 17
    var_68 = module_0.Position(var_3, var_3, var_1)
    var_69 = module_0.Position(var_3, var_22, var_67)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 6
    var_7 = 10
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 11



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'
    var_29 = 123
    var_30 = '123'
    var_31 = module_0.ScalarToken(var_29, var_1, var_4, var_30)
    var_32 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_33 = {var_31: var_32}
    var_34 = '123: value'
    var_35 = module_0.ScalarToken(var_3, var_1, var_4, var_2)
    var_36 = module_0.ScalarToken(var_6, var_7, var_8, var_2)
    var_37 = {var_35: var_36}



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1, key2: value2'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14)
    var_16 = {var_3: var_7, var_11: var_15}



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = {var_3: var_7}
    var_9 = 1
    var_10 = module_1.Position(var_9, var_9, var_1)
    var_11 = 9
    var_12 = module_1.Position(var_9, var_11, var_6)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'different'
    var_7 = 8
    var_8 = module_0.ScalarToken(var_6, var_1, var_7, var_6)
    var_9 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_10 = 'test'
    var_11 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_12 = 1
    var_13 = 4
    var_14 = ' test'
    var_15 = module_0.ScalarToken(var_0, var_12, var_13, var_14)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 'other_value'
    var_6 = 6
    var_7 = module_0.Token(var_5, var_1, var_6, var_5)
    var_8 = 5
    var_9 = module_0.Token(var_0, var_1, var_8, var_0)
    var_10 = 1
    var_11 = module_0.Token(var_0, var_10, var_2, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value1'
    var_6 = 8
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 16
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 23
    var_15 = 29
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = 30
    var_20 = var_4._value
    var_21 = [var_20]
    var_22 = var_12._value
    var_23 = [var_22]
    var_24 = [var_4]
    var_25 = [var_12]
    var_26 = module_0.ScalarToken(var_5, var_6, var_7, var_0)



# Parsed testcases at query #6
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
    var_9 = 'keyvalue'
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = 'nested_key'
    var_13 = 10
    var_14 = 19
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = 'nested_value'
    var_17 = 21
    var_18 = 32
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_16)
    var_20 = {var_15: var_19}
    var_21 = 'nested_keynested_value'
    var_22 = 'keynested_keynested_value'
    var_23 = [var_0, var_12]
    var_24 = [var_0, var_12]
    var_25 = {}
    var_26 = ''
    var_27 = 'key'
    var_28 = [var_27]
    var_29 = 'key'
    var_30 = [var_29]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 9
    var_5 = "{'key': 'value'}"
    var_6 = {}
    var_7 = 0
    var_8 = 1
    var_9 = '{}'
    var_10 = 'nested_key'
    var_11 = 'nested_value'
    var_12 = {var_10: var_11}
    var_13 = {var_0: var_12}
    var_14 = 0
    var_15 = 29
    var_16 = "{'key': {'nested_key': 'nested_value'}}"



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 'key'
    var_3 = 2
    var_4 = module_0.ScalarToken(var_2, var_1, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7)
    var_9 = {var_4: var_8}
    var_10 = 'key1'
    var_11 = 3
    var_12 = module_0.ScalarToken(var_10, var_1, var_11)
    var_13 = 'value1'
    var_14 = 5
    var_15 = 10
    var_16 = module_0.ScalarToken(var_13, var_14, var_15)
    var_17 = 'key2'
    var_18 = 12
    var_19 = 15
    var_20 = module_0.ScalarToken(var_17, var_18, var_19)
    var_21 = 'value2'
    var_22 = 17
    var_23 = 22
    var_24 = module_0.ScalarToken(var_21, var_22, var_23)
    var_25 = {var_12: var_16, var_20: var_24}
    var_26 = module_0.ScalarToken(var_10, var_1, var_11)
    var_27 = module_0.ScalarToken(var_17, var_14, var_7)
    var_28 = module_0.ScalarToken(var_21, var_15, var_19)
    var_29 = {var_27: var_28}
    var_30 = module_0.ScalarToken(var_2, var_1, var_3)
    var_31 = 9
    var_32 = module_0.ScalarToken(var_13, var_6, var_31)
    var_33 = 11
    var_34 = 16
    var_35 = module_0.ScalarToken(var_21, var_33, var_34)
    var_36 = [var_32, var_35]
    var_37 = module_0.ListToken(var_36, var_6, var_34)
    var_38 = {var_30: var_37}
    var_39 = 'All test cases passed!'
    var_40 = print(var_39)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'nested_key'
    var_13 = 10
    var_14 = 19
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = 'nested_value'
    var_17 = 21
    var_18 = 32
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_16)
    var_20 = {var_15: var_19}
    var_21 = 'nested_key: nested_value'
    var_22 = 'key: {nested_key: nested_value}'



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = 456
    var_7 = '456'
    var_8 = module_0.ScalarToken(var_6, var_1, var_2, var_7)
    var_9 = 1
    var_10 = 3
    var_11 = module_0.ScalarToken(var_0, var_9, var_10, var_3)
    var_12 = module_0.ScalarToken(var_0, var_1, var_10, var_3)
    var_13 = 124
    var_14 = '124'
    var_15 = module_0.ScalarToken(var_13, var_1, var_2, var_14)
    var_16 = 'key'
    var_17 = '"key"'
    var_18 = module_0.ScalarToken(var_16, var_1, var_2, var_17)
    var_19 = 'value'
    var_20 = 4
    var_21 = 8
    var_22 = '"value"'
    var_23 = module_0.ScalarToken(var_19, var_20, var_21, var_22)
    var_24 = {var_18: var_23}
    var_25 = '{"key": "value"}'
    var_26 = module_0.ScalarToken(var_16, var_1, var_2, var_17)
    var_27 = module_0.ScalarToken(var_19, var_20, var_21, var_22)
    var_28 = {var_26: var_27}
    var_29 = '1'
    var_30 = module_0.ScalarToken(var_9, var_1, var_1, var_29)
    var_31 = [var_30]
    var_32 = '[1]'
    var_33 = module_0.ListToken(var_31, var_1, var_1, var_32)
    var_34 = module_0.ScalarToken(var_9, var_1, var_1, var_29)
    var_35 = [var_34]
    var_36 = module_0.ListToken(var_35, var_1, var_1, var_32)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0
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
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_13 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_14 = {var_12: var_13}
    var_15 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_16 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_17 = {var_15: var_16}
    var_18 = 1
    var_19 = module_1.Position(var_18, var_18, var_1)
    var_20 = 9
    var_21 = module_1.Position(var_18, var_20, var_6)
    var_22 = module_1.Position(var_18, var_18, var_1)
    var_23 = module_1.Position(var_18, var_20, var_6)
    var_24 = [var_0]
    var_25 = [var_0]
    var_26 = module_1.Position(var_18, var_18, var_1)
    var_27 = module_1.Position(var_18, var_20, var_6)
    var_28 = module_1.Position(var_18, var_18, var_1)
    var_29 = module_1.Position(var_18, var_20, var_6)
    var_30 = [var_0]
    var_31 = [var_0]
    var_32 = module_1.Position(var_18, var_18, var_1)
    var_33 = module_1.Position(var_18, var_20, var_6)
    var_34 = module_1.Position(var_18, var_18, var_1)
    var_35 = module_1.Position(var_18, var_20, var_6)
    var_36 = [var_0]
    var_37 = [var_0]
    var_38 = module_1.Position(var_18, var_18, var_1)
    var_39 = module_1.Position(var_18, var_20, var_6)
    var_40 = module_1.Position(var_18, var_18, var_1)
    var_41 = module_1.Position(var_18, var_20, var_6)
    var_42 = [var_0]
    var_43 = [var_0]
    var_44 = module_1.Position(var_18, var_18, var_1)
    var_45 = module_1.Position(var_18, var_20, var_6)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'nested_key'
    var_13 = 10
    var_14 = 19
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = 'nested_value'
    var_17 = 21
    var_18 = 32
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_16)
    var_20 = {var_15: var_19}
    var_21 = 'nested_key: nested_value'
    var_22 = 'key: {nested_key: nested_value}'
    var_23 = {var_5: var_9}
    var_24 = 'different content'
    var_25 = {var_5: var_9}
    var_26 = 5
    var_27 = 15



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'key2'
    var_5 = 12
    var_6 = 15
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 'value1'
    var_9 = 5
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 25
    var_18 = 'key1: value1, key2: value2'



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'
    var_29 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_30 = module_0.ScalarToken(var_19, var_16, var_8, var_19)
    var_31 = module_0.ScalarToken(var_23, var_17, var_21, var_23)
    var_32 = {var_30: var_31}
    var_33 = 'key2: value2'
    var_34 = 'key1: {key2: value2}'
    var_35 = 123
    var_36 = '123'
    var_37 = module_0.ScalarToken(var_35, var_1, var_4, var_36)
    var_38 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_39 = {var_37: var_38}
    var_40 = '123: value'



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = hash(var_3)
    var_5 = hash(var_0)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value2'
    var_5 = 10
    var_6 = 15
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'value1value2'
    var_10 = var_3._value
    var_11 = [var_10]



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 14
    var_12 = [var_10]



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'test_content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = repr(var_2)
    assert var_3 == "ScalarToken('')"
    var_4 = 'example'
    var_5 = 6
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = repr(var_6)
    assert var_7 == "ScalarToken('example')"
    var_8 = 'a\nb\tc'
    var_9 = 4
    var_10 = module_0.ScalarToken(var_8, var_1, var_9, var_8)
    var_11 = repr(var_10)
    assert var_11 == "ScalarToken('a\nb\tc')"



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_1.Position(var_5, var_5, var_1)
    var_7 = 3
    var_8 = module_1.Position(var_5, var_7, var_2)
    var_9 = repr(var_4)
    assert var_9 == "Token('abc')"
    var_10 = module_0.Token(var_0, var_1, var_2, var_3)
    var_11 = var_4.__eq__(var_10)
    assert var_11 is True
    var_12 = 456
    var_13 = module_0.Token(var_12, var_1, var_2, var_3)
    var_14 = var_4.__eq__(var_13)
    assert var_14 is False



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 0
    var_2 = 2
    var_3 = 'abcdef'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('abc')"



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 12
    var_12 = [var_1]
    var_13 = [var_1]



# Parsed testcases at query #2
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
    var_9 = 'key value'
    var_10 = [var_0]
    var_11 = [var_0]



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.ScalarToken(var_0, var_1, var_0, var_2)
    var_4 = module_0.ScalarToken(var_0, var_1, var_0, var_2)
    var_5 = 2
    var_6 = module_0.ScalarToken(var_5, var_1, var_0, var_2)
    var_7 = module_0.ScalarToken(var_0, var_0, var_5, var_2)
    var_8 = 'b'
    var_9 = module_0.ScalarToken(var_0, var_1, var_0, var_8)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 'value'
    var_2 = 0
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = module_0.Token(var_1, var_2, var_3, var_0)
    var_6 = 'different_value'
    var_7 = module_0.Token(var_6, var_2, var_3, var_0)
    var_8 = 1
    var_9 = module_0.Token(var_1, var_8, var_3, var_0)
    var_10 = 5
    var_11 = module_0.Token(var_1, var_2, var_10, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1)
    var_3 = 2
    var_4 = module_0.ScalarToken(var_3, var_0, var_0)
    var_5 = {var_2: var_4}
    var_6 = '12'
    var_7 = module_0.ScalarToken(var_0, var_1, var_1)
    var_8 = {var_0: var_7}
    var_9 = module_0.ScalarToken(var_3, var_0, var_0)
    var_10 = {var_0: var_9}



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14)
    var_16 = 'key1: value1, key2: value2'
    var_17 = {var_3: var_7, var_11: var_15}



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 5
    var_8 = 9
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = '{key: value}'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 6
    var_17 = 11
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 14
    var_21 = 17
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 20
    var_25 = 25
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = '{key1: value1, key2: value2}'
    var_29 = 'nested_key'
    var_30 = module_0.ScalarToken(var_29, var_1, var_8, var_29)
    var_31 = 'nested_value'
    var_32 = 12
    var_33 = 23
    var_34 = module_0.ScalarToken(var_31, var_32, var_33, var_31)
    var_35 = {var_30: var_34}
    var_36 = '{nested_key: nested_value}'
    var_37 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_38 = '{key: {nested_key: nested_value}}'
    var_39 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_40 = 10
    var_41 = module_0.ScalarToken(var_15, var_7, var_40, var_15)
    var_42 = module_0.ScalarToken(var_23, var_32, var_21, var_23)
    var_43 = [var_41, var_42]
    var_44 = '[value1, value2]'
    var_45 = module_0.ListToken(var_43, var_7, var_21, var_44)
    var_46 = {var_39: var_45}
    var_47 = '{key: [value1, value2]}'
    var_48 = 'All test cases passed!'
    var_49 = print(var_48)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 1
    var_5 = 3
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_3)
    var_7 = 'value'
    var_8 = 5
    var_9 = 9
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_7)
    var_11 = {var_6: var_10}
    var_12 = 'key: value'
    var_13 = 'key1'
    var_14 = 4
    var_15 = module_0.ScalarToken(var_13, var_4, var_14, var_13)
    var_16 = 'value1'
    var_17 = 6
    var_18 = 11
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_16)
    var_20 = 'key2'
    var_21 = 13
    var_22 = 16
    var_23 = module_0.ScalarToken(var_20, var_21, var_22, var_20)
    var_24 = 'value2'
    var_25 = 18
    var_26 = 23
    var_27 = module_0.ScalarToken(var_24, var_25, var_26, var_24)
    var_28 = {var_15: var_19, var_23: var_27}
    var_29 = 'key1: value1, key2: value2'



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 'other_value'
    var_6 = module_0.Token(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.Token(var_0, var_7, var_2, var_0)
    var_9 = 5
    var_10 = module_0.Token(var_0, var_1, var_9, var_0)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 1
    var_5 = 3
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_3)
    var_7 = 'value'
    var_8 = 5
    var_9 = 9
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_7)
    var_11 = {var_6: var_10}
    var_12 = 10
    var_13 = 'key: value'
    var_14 = 'nested_key'
    var_15 = 12
    var_16 = 21
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = 'nested_value'
    var_19 = 23
    var_20 = 34
    var_21 = module_0.ScalarToken(var_18, var_19, var_20, var_18)
    var_22 = {var_17: var_21}
    var_23 = 11
    var_24 = 35
    var_25 = 'nested_key: nested_value'
    var_26 = 'key: nested_key: nested_value'



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = module_0.DictToken()
    var_4 = 1
    var_5 = module_1.Position(var_4, var_4)
    var_6 = 2
    var_7 = module_1.Position(var_4, var_6)
    var_8 = 'key'
    var_9 = 3
    var_10 = "{'key': 'value'}"
    var_11 = module_0.ScalarToken(var_8, var_4, var_9, var_10)
    var_12 = 'value'
    var_13 = 6
    var_14 = 10
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_10)
    var_16 = {var_11: var_15}
    var_17 = 12
    var_18 = module_0.DictToken()
    var_19 = module_1.Position(var_4, var_4)
    var_20 = 13
    var_21 = module_1.Position(var_4, var_20)
    var_22 = 'key1'
    var_23 = 4
    var_24 = "{'key1': 'value1', 'key2': 'value2'}"
    var_25 = module_0.ScalarToken(var_22, var_4, var_23, var_24)
    var_26 = 'value1'
    var_27 = 7
    var_28 = module_0.ScalarToken(var_26, var_27, var_17, var_24)
    var_29 = 'key2'
    var_30 = 15
    var_31 = 18
    var_32 = module_0.ScalarToken(var_29, var_30, var_31, var_24)
    var_33 = 'value2'
    var_34 = 21
    var_35 = 26
    var_36 = module_0.ScalarToken(var_33, var_34, var_35, var_24)
    var_37 = {var_25: var_28, var_32: var_36}
    var_38 = 28
    var_39 = module_0.DictToken()
    var_40 = module_1.Position(var_4, var_4)
    var_41 = 29
    var_42 = module_1.Position(var_4, var_41)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 'value1'
    var_2 = 8
    var_3 = 14
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_6 = 'value2'
    var_7 = 25
    var_8 = 31
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_0)
    var_10 = '{"key1": "value1", "key2": "value2"}'
    var_11 = module_0.ScalarToken(var_1, var_2, var_3, var_10)
    var_12 = module_0.ScalarToken(var_1, var_2, var_3, var_10)
    var_13 = '{"key1": "value1", "key2": "value3"}'
    var_14 = module_0.ScalarToken(var_1, var_2, var_3, var_13)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = [var_0]
    var_19 = [var_8]
    var_20 = [var_0]
    var_21 = [var_8]
    var_22 = {var_3: var_7, var_11: var_15}
    var_23 = {var_3: var_7}
    var_24 = 'key1: value1'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '\n    Unit test for constructor of class DictToken\n    '
    var_1 = '{"key": "value"}'
    var_2 = 0
    var_3 = len(var_1)
    var_4 = 1
    var_5 = var_3 - var_4
    var_6 = 'key'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_4, var_7, var_1)
    var_9 = 'value'
    var_10 = 7
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_1)
    var_13 = {var_8: var_12}
    var_14 = module_0.ScalarToken(var_6, var_4, var_7, var_1)
    var_15 = module_0.ScalarToken(var_9, var_10, var_11, var_1)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'key2'
    var_5 = 13
    var_6 = 16
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = 'value1'
    var_9 = 5
    var_10 = 11
    var_11 = module_0.ScalarToken(var_8, var_9, var_10)
    var_12 = 'value2'
    var_13 = 18
    var_14 = 24
    var_15 = module_0.ScalarToken(var_12, var_13, var_14)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = '{"key1": "value1", "key2": "value2"}'
    var_18 = [var_0]
    var_19 = [var_4]
    var_20 = [var_0]



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'
    var_29 = 123
    var_30 = '123'
    var_31 = module_0.ScalarToken(var_29, var_1, var_4, var_30)
    var_32 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_33 = {var_31: var_32}
    var_34 = '123: value'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6)
    var_8 = {var_3: var_7}



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value'
    var_7 = 4
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = {var_5: var_9}
    var_11 = 'key: value'
    var_12 = 'key1'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = 'value1'
    var_16 = 5
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = 'key2'
    var_20 = 12
    var_21 = 15
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_19)
    var_23 = 'value2'
    var_24 = 17
    var_25 = 22
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_23)
    var_27 = {var_14: var_18, var_22: var_26}
    var_28 = 'key1: value1, key2: value2'



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 13
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 18
    var_14 = 24
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1, key2: value2'
    var_20 = [var_0]
    var_21 = [var_9]
    var_22 = [var_0]
    var_23 = [var_9]



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 0
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3)
    var_5 = {var_0: var_4}
    var_6 = 10
    var_7 = 2
    var_8 = module_0.ScalarToken(var_0, var_2, var_7)
    var_9 = {var_0: var_8}
    var_10 = module_0.ScalarToken(var_1, var_2, var_3)
    var_11 = {var_0: var_10}



