####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 'b'
    var_4 = 1
    var_5 = module_0.ScalarToken(var_3, var_4, var_4, var_3)
    var_6 = [var_2, var_5]
    var_7 = 2
    var_8 = '1'
    var_9 = module_0.ScalarToken(var_4, var_7, var_7, var_8)
    var_10 = 3
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_7, var_10, var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = var_6[var_1]
    var_15 = var_6[var_4]
    var_16 = var_13[var_1]
    var_17 = var_13[var_4]
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'a=1;b=2'
    var_20 = len(var_19)
    var_21 = var_20 - var_4
    var_22 = len(var_19)
    var_23 = var_22 - var_4



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 'test1'
    var_6 = module_0.Token(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.Token(var_0, var_7, var_2, var_0)
    var_9 = 4
    var_10 = module_0.Token(var_0, var_1, var_9, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = module_0.Token(var_0, var_1, var_2, var_3)
    var_7 = 'test2'
    var_8 = module_0.Token(var_7, var_1, var_2, var_3)
    var_9 = module_0.Token(var_0, var_1, var_2, var_3)
    var_10 = 1
    var_11 = module_0.Token(var_0, var_10, var_2, var_3)
    var_12 = module_0.Token(var_0, var_1, var_2, var_3)
    var_13 = 4
    var_14 = module_0.Token(var_0, var_1, var_13, var_3)
    var_15 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = 'test1'
    var_6 = 4
    var_7 = module_0.Token(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = module_0.Token(var_0, var_8, var_6, var_0)
    var_10 = module_0.Token(var_0, var_1, var_6, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = module_1.Position(var_19, var_19, var_1)
    var_28 = len(var_24)
    var_29 = len(var_24)
    var_30 = var_29 - var_19
    var_31 = module_1.Position(var_19, var_28, var_30)
    var_32 = [var_1]
    var_33 = [var_1]
    var_34 = len(var_24)
    var_35 = var_34 - var_19
    var_36 = {}
    var_37 = -1
    var_38 = ''



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'
    var_20 = len(var_19)
    var_21 = 1
    var_22 = var_20 - var_21
    var_23 = len(var_19)
    var_24 = var_23 - var_21



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

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
    var_17 = 'key1=value1,key2=value2'
    var_18 = 1
    var_19 = module_1.Position(var_18, var_18, var_1)
    var_20 = module_1.Position(var_18, var_13, var_14)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 10
    var_3 = 'key1'
    var_4 = 0
    var_5 = 3
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = 'key2'
    var_8 = 11
    var_9 = 14
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = 'value1'
    var_12 = 5
    var_13 = 9
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 'value2'
    var_16 = 16
    var_17 = 20
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_0)
    var_19 = {var_6: var_14, var_10: var_18}
    var_20 = module_0.ScalarToken(var_3, var_4, var_5, var_0)
    var_21 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_22 = {var_3: var_20, var_7: var_21}
    var_23 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_24 = module_0.ScalarToken(var_15, var_16, var_17, var_0)
    var_25 = {var_3: var_23, var_7: var_24}



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 'b'
    var_4 = 2
    var_5 = module_0.ScalarToken(var_3, var_4, var_4, var_3)
    var_6 = [var_2, var_5]
    var_7 = 1
    var_8 = '1'
    var_9 = module_0.ScalarToken(var_7, var_7, var_7, var_8)
    var_10 = 3
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = var_6[var_1]
    var_15 = var_6[var_7]
    var_16 = var_13[var_1]
    var_17 = var_13[var_7]
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'a1b2'



# Parsed testcases at query #10
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
    var_14 = 21
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1=value1, key2=value2'



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 'test1'
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = 1
    var_9 = module_0.Token(var_0, var_8, var_2, var_3)
    var_10 = 4
    var_11 = module_0.Token(var_0, var_1, var_10, var_3)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = 1
    var_13 = 4
    var_14 = ' test'
    var_15 = module_0.ScalarToken(var_0, var_12, var_13, var_14)
    var_16 = hash(var_3)
    var_17 = hash(var_15)
    var_18 = 42
    var_19 = '42'
    var_20 = module_0.ScalarToken(var_18, var_1, var_12, var_19)
    var_21 = module_0.ScalarToken(var_18, var_1, var_12, var_19)
    var_22 = hash(var_20)
    var_23 = hash(var_21)
    var_24 = None
    var_25 = 'null'
    var_26 = module_0.ScalarToken(var_24, var_1, var_2, var_25)
    var_27 = module_0.ScalarToken(var_24, var_1, var_2, var_25)
    var_28 = hash(var_26)
    var_29 = hash(var_27)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = [var_3, var_4]



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 0
    var_8 = module_1.Position(var_5, var_6, var_7)
    var_9 = module_1.Position(var_5, var_6, var_5)
    var_10 = repr(var_4)
    assert var_10 == "ScalarToken('42')"



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = hash(var_3)
    var_13 = hash(var_3)
    var_14 = 42
    var_15 = 1
    var_16 = '42'
    var_17 = module_0.ScalarToken(var_14, var_1, var_15, var_16)
    var_18 = module_0.ScalarToken(var_14, var_1, var_15, var_16)
    var_19 = hash(var_17)
    var_20 = hash(var_18)
    var_21 = 99
    var_22 = '99'
    var_23 = module_0.ScalarToken(var_21, var_1, var_15, var_22)
    var_24 = hash(var_17)
    var_25 = hash(var_23)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 3
    var_3 = 'a'
    var_4 = 0
    var_5 = module_0.ScalarToken(var_3, var_4, var_4, var_0)
    var_6 = 'b'
    var_7 = 1
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = [var_5, var_8]
    var_10 = module_0.ListToken(var_9, var_1, var_2, var_0)
    var_11 = module_1.Position(var_7, var_7, var_4)
    var_12 = 4
    var_13 = 3
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = [var_4]
    var_16 = [var_7]
    var_17 = repr(var_10)
    assert var_17 == "ListToken('test')"
    var_18 = module_0.ListToken(var_9, var_1, var_2, var_0)
    var_19 = 'c'
    var_20 = 2
    var_21 = module_0.ScalarToken(var_19, var_20, var_20, var_0)
    var_22 = [var_21]
    var_23 = module_0.ListToken(var_22, var_1, var_2, var_0)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('test')"



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('test')"



# Parsed testcases at query #19
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
    var_17 = 'key1=value1,key2=value2'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = []
    var_5 = var_3.lookup(var_4)
    var_6 = module_0.Token(var_0, var_1, var_2, var_0)
    var_7 = 'child'
    var_8 = 1
    var_9 = 2
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = [var_1]
    var_12 = var_6.lookup(var_11)
    var_13 = module_0.Token(var_7, var_8, var_9, var_0)
    var_14 = module_0.Token(var_0, var_1, var_2, var_0)
    var_15 = module_0.Token(var_7, var_8, var_9, var_0)
    var_16 = [var_1]
    var_17 = var_14.lookup(var_16)
    var_18 = 'grandchild'
    var_19 = module_0.Token(var_18, var_9, var_2, var_0)
    var_20 = [var_1, var_1]
    var_21 = var_14.lookup(var_20)
    var_22 = module_0.Token(var_18, var_9, var_2, var_0)
    var_23 = 'key'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = 10
    var_27 = '{"key": "value"}'
    var_28 = 7
    var_29 = 11
    var_30 = module_0.Token(var_24, var_28, var_29, var_27)
    var_31 = [var_23]
    var_32 = module_0.Token(var_24, var_28, var_29, var_27)
    var_33 = 'item1'
    var_34 = 'item2'
    var_35 = [var_33, var_34]
    var_36 = 12
    var_37 = '["item1", "item2"]'
    var_38 = module_0.ListToken(var_35, var_1, var_36, var_37)
    var_39 = 6
    var_40 = module_0.Token(var_33, var_8, var_39, var_37)
    var_41 = [var_1]
    var_42 = module_0.Token(var_33, var_8, var_39, var_37)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)



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



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1: value1, key2: value2'
    var_1 = 'key1'
    var_2 = 0
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'key2'
    var_6 = 12
    var_7 = 15
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value1'
    var_10 = 5
    var_11 = 10
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_12, var_8: var_16}
    var_18 = module_0.DictToken()
    var_19 = [var_2, var_1]
    var_20 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_21 = 0
    var_22 = 'invalid_key'
    var_23 = [var_21, var_22]
    var_24 = 'outer: {inner: value}'
    var_25 = 'outer'
    var_26 = 4
    var_27 = module_0.ScalarToken(var_25, var_22, var_26, var_24)
    var_28 = 'inner'
    var_29 = 7
    var_30 = 11
    var_31 = module_0.ScalarToken(var_28, var_29, var_30, var_24)
    var_32 = 'value'
    var_33 = 13
    var_34 = module_0.ScalarToken(var_32, var_33, var_14, var_24)
    var_35 = {var_31: var_34}
    var_36 = module_0.DictToken()
    var_37 = {var_27: var_36}
    var_38 = module_0.DictToken()
    var_39 = [var_22, var_25, var_28]
    var_40 = module_0.ScalarToken(var_28, var_29, var_30, var_24)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3.value
    var_5 = 1
    var_6 = 4
    var_7 = module_1.Position(var_5, var_6, var_1)
    var_8 = module_1.Position(var_5, var_6, var_2)
    var_9 = 0
    var_10 = [var_9]
    var_11 = var_3.lookup(var_10)
    var_12 = 0
    var_13 = [var_12]
    var_14 = var_3.lookup_key(var_13)
    var_15 = repr(var_3)
    assert var_15 == "Token('test')"
    var_16 = module_0.Token(var_12, var_13, var_14, var_12)
    var_17 = 'test2'
    var_18 = module_0.Token(var_17, var_13, var_6, var_17)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = len(var_24)
    var_28 = var_27 - var_19
    var_29 = module_1.Position(var_19, var_19, var_1)
    var_30 = len(var_24)
    var_31 = len(var_24)
    var_32 = var_31 - var_19
    var_33 = module_1.Position(var_19, var_30, var_32)
    var_34 = [var_0]
    var_35 = [var_4]
    var_36 = [var_0]
    var_37 = [var_4]
    var_38 = len(var_24)
    var_39 = var_38 - var_19



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1=value1,key2=value2'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 5
    var_3 = 'item1'
    var_4 = 0
    var_5 = 4
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = 'item2'
    var_8 = 6
    var_9 = 10
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = [var_6, var_10]
    var_12 = module_0.ListToken(var_11, var_1, var_2, var_0)
    var_13 = 1
    var_14 = module_1.Position(var_13, var_13, var_1)
    var_15 = var_2 + var_13
    var_16 = module_1.Position(var_13, var_15, var_2)
    var_17 = [var_4]
    var_18 = [var_13]
    var_19 = repr(var_12)
    assert var_19 == "ListToken('test ')"



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = hash(var_3)
    var_13 = var_3._value
    var_14 = hash(var_13)
    var_15 = 42
    var_16 = 1
    var_17 = '42'
    var_18 = module_0.ScalarToken(var_15, var_1, var_16, var_17)
    var_19 = module_0.ScalarToken(var_15, var_1, var_16, var_17)
    var_20 = hash(var_18)
    var_21 = hash(var_19)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 5
    var_3 = 'a'
    var_4 = 0
    var_5 = module_0.ScalarToken(var_3, var_4, var_4, var_0)
    var_6 = 'b'
    var_7 = 1
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = [var_5, var_8]
    var_10 = module_0.ListToken(var_9, var_1, var_2, var_0)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1: value1, key2: value2'
    var_1 = 'key1'
    var_2 = 0
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'key2'
    var_6 = 13
    var_7 = 16
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value1'
    var_10 = 5
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 18
    var_15 = 24
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_12, var_8: var_16}
    var_18 = [var_2, var_1]
    var_19 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_20 = 0
    var_21 = 'nonexistent_key'
    var_22 = [var_20, var_21]
    var_23 = []
    var_24 = 1
    var_25 = 'key1'
    var_26 = [var_24, var_25]



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = hash(var_4)
    var_7 = hash(var_5)
    var_8 = hash(var_4)
    var_9 = hash(var_0)
    var_10 = 'hello'
    var_11 = 4
    var_12 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_13 = module_0.ScalarToken(var_10, var_1, var_11, var_10)
    var_14 = hash(var_12)
    var_15 = hash(var_13)
    var_16 = hash(var_12)
    var_17 = hash(var_10)
    var_18 = 10
    var_19 = '10'
    var_20 = module_0.ScalarToken(var_18, var_1, var_2, var_19)
    var_21 = 20
    var_22 = '20'
    var_23 = module_0.ScalarToken(var_21, var_1, var_2, var_22)
    var_24 = hash(var_20)
    var_25 = hash(var_23)
    var_26 = '1'
    var_27 = module_0.ScalarToken(var_2, var_1, var_1, var_26)
    var_28 = True
    var_29 = 3
    var_30 = 'True'
    var_31 = module_0.ScalarToken(var_28, var_1, var_29, var_30)
    var_32 = hash(var_27)
    var_33 = hash(var_31)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = module_0.Token(var_0, var_1, var_2, var_3)
    var_7 = 'diff'
    var_8 = module_0.Token(var_7, var_1, var_2, var_3)
    var_9 = module_0.Token(var_0, var_1, var_2, var_3)
    var_10 = 1
    var_11 = module_0.Token(var_0, var_10, var_2, var_3)
    var_12 = module_0.Token(var_0, var_1, var_2, var_3)
    var_13 = 4
    var_14 = module_0.Token(var_0, var_1, var_13, var_3)
    var_15 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = None
    var_2 = 0
    var_3 = len(var_0)
    var_4 = 1
    var_5 = var_3 - var_4
    var_6 = module_0.Token(var_1, var_2, var_5, var_0)
    var_7 = [var_4]
    var_8 = var_6.lookup(var_7)
    var_9 = 2
    var_10 = 3
    var_11 = [var_4, var_9, var_10]
    var_12 = var_6.lookup(var_11)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = 2
    var_9 = module_1.Position(var_5, var_8, var_8)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = module_0.Token(var_0, var_1, var_2, var_3)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.Token(var_7, var_1, var_8, var_3)
    var_10 = module_0.Token(var_0, var_1, var_2, var_3)
    var_11 = 1
    var_12 = 4
    var_13 = module_0.Token(var_0, var_11, var_12, var_3)
    var_14 = module_0.Token(var_0, var_1, var_2, var_3)
    var_15 = module_0.Token(var_0, var_1, var_12, var_3)
    var_16 = module_0.Token(var_0, var_1, var_2, var_3)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = len(var_24)
    var_28 = var_27 - var_19



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 21
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1=value1;key2=value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = [var_0]
    var_28 = [var_0]
    var_29 = module_1.Position(var_19, var_19, var_1)
    var_30 = len(var_24)
    var_31 = len(var_24)
    var_32 = var_31 - var_19
    var_33 = module_1.Position(var_19, var_30, var_32)
    var_34 = len(var_24)
    var_35 = var_34 - var_19
    var_36 = {}
    var_37 = ''



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 'test1'
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = 'test2'
    var_9 = module_0.Token(var_8, var_1, var_2, var_3)
    var_10 = module_0.Token(var_0, var_1, var_2, var_3)
    var_11 = 1
    var_12 = module_0.Token(var_0, var_11, var_2, var_3)
    var_13 = module_0.Token(var_0, var_1, var_2, var_3)
    var_14 = 4
    var_15 = module_0.Token(var_0, var_1, var_14, var_3)
    var_16 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 'test1'
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = 1
    var_9 = module_0.Token(var_0, var_8, var_2, var_3)
    var_10 = 4
    var_11 = module_0.Token(var_0, var_1, var_10, var_3)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1\nkey2: value2'



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



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1\nkey2: value2'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 21
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



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
    var_17 = 'key1: value1, key2: value2'
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = len(var_17)
    var_22 = var_21 - var_19



# Parsed testcases at query #12
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
    var_17 = 'key1=value1,key2=value2'



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 21
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1=value1 key2=value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = len(var_24)
    var_28 = var_27 - var_19



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
    var_17 = 'key1=value1,key2=value2'



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



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1=value1,key2=value2'
    var_20 = len(var_19)
    var_21 = 1
    var_22 = var_20 - var_21
    var_23 = module_1.Position(var_21, var_21, var_1)
    var_24 = len(var_19)
    var_25 = len(var_19)
    var_26 = var_25 - var_21
    var_27 = module_1.Position(var_21, var_24, var_26)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 21
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = 'key1: value1'
    var_6 = module_0.ScalarToken(var_3, var_1, var_4, var_5)
    var_7 = 'value1'
    var_8 = 5
    var_9 = 10
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_5)
    var_11 = 'key2'
    var_12 = 12
    var_13 = 15
    var_14 = 'key2: value2'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'value2'
    var_17 = 17
    var_18 = 22
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_14)
    var_20 = {var_6: var_10, var_15: var_19}
    var_21 = 'key1: value1\nkey2: value2'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 21
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1: value1, key2: value2'



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'key2'
    var_6 = 10
    var_7 = 13
    var_8 = 'key2: value2'
    var_9 = module_0.ScalarToken(var_5, var_6, var_7, var_8)
    var_10 = 'value1'
    var_11 = 5
    var_12 = module_0.ScalarToken(var_10, var_11, var_6, var_3)
    var_13 = 'value2'
    var_14 = 15
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_8)
    var_17 = {var_4: var_12, var_9: var_16}
    var_18 = 'key1: value1\nkey2: value2'



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'test content'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 0
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = 'value2'
    var_10 = 6
    var_11 = 10
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_2)
    var_13 = {var_3: var_8, var_4: var_12}
    var_14 = module_0.ScalarToken(var_5, var_6, var_7, var_2)
    var_15 = module_0.ScalarToken(var_9, var_10, var_11, var_2)
    var_16 = {var_3: var_14, var_4: var_15}
    var_17 = module_0.ScalarToken(var_5, var_6, var_7, var_2)
    var_18 = module_0.ScalarToken(var_9, var_10, var_11, var_2)
    var_19 = {var_3: var_17, var_4: var_18}



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 0
    var_25 = 22
    var_26 = 'key1: value1, key2: value2'



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_14)
    var_18 = {var_5: var_9, var_13: var_17}
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'
    var_25 = len(var_24)
    var_26 = var_25 - var_19
    var_27 = len(var_24)
    var_28 = var_27 - var_19
    var_29 = module_1.Position(var_19, var_19, var_1)
    var_30 = len(var_24)
    var_31 = len(var_24)
    var_32 = var_31 - var_19
    var_33 = module_1.Position(var_19, var_30, var_32)
    var_34 = [var_0]
    var_35 = [var_0]
    var_36 = len(var_24)
    var_37 = var_36 - var_19
    var_38 = var_8[var_1]
    var_39 = var_17[var_19]
    var_40 = {var_38: var_39}
    var_41 = len(var_24)
    var_42 = var_41 - var_19



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = zip(var_8, var_17)
    var_19 = dict(var_18)
    var_20 = 'key1: value1, key2: value2'
    var_21 = len(var_20)
    var_22 = 1
    var_23 = var_21 - var_22
    var_24 = len(var_20)
    var_25 = var_24 - var_22
    var_26 = module_1.Position(var_22, var_22, var_1)
    var_27 = len(var_20)
    var_28 = len(var_20)
    var_29 = var_28 - var_22
    var_30 = module_1.Position(var_22, var_27, var_29)
    var_31 = [var_0]
    var_32 = [var_0]
    var_33 = len(var_20)
    var_34 = var_33 - var_22
    var_35 = len(var_20)



