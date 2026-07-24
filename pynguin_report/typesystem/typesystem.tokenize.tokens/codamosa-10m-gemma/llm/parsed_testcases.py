####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'val    2'
    var_13 = 18
    var_14 = 23
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: val1, key2: val    2'
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = module_0.DictToken()
    var_22 = len(var_17)
    var_23 = var_22 - var_19
    var_24 = module_0.ScalarToken(var_0, var_1, var_2, var_0)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'diff'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.ScalarToken(var_0, var_1, var_2, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = '10'
    var_4 = 'test'
    var_5 = 3
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_8 = 'diff'
    var_9 = module_0.ScalarToken(var_8, var_1, var_5, var_8)
    var_10 = 4
    var_11 = 'test!'
    var_12 = module_0.ScalarToken(var_4, var_1, var_10, var_11)
    var_13 = 1
    var_14 = ' test'
    var_15 = module_0.ScalarToken(var_4, var_13, var_10, var_14)
    var_16 = 'k'
    var_17 = module_0.ScalarToken(var_16, var_1, var_1, var_16)
    var_18 = '1'
    var_19 = module_0.ScalarToken(var_13, var_2, var_2, var_18)
    var_20 = module_0.ScalarToken(var_16, var_1, var_1, var_16)
    var_21 = module_0.ScalarToken(var_13, var_2, var_2, var_18)
    var_22 = {var_16: var_19}
    var_23 = 'k: 1'
    var_24 = {var_16: var_21}
    var_25 = {var_16: var_19}
    var_26 = 5
    var_27 = 'k: 123'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = '10'
    var_4 = 20
    var_5 = '20'
    var_6 = 1
    var_7 = '0'
    var_8 = 3
    var_9 = '100'
    var_10 = 'abc'



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = [var_0]
    var_18 = [var_0]



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 'val2'
    var_12 = 18
    var_13 = 22
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_11)
    var_15 = 'key1: val1, key2: val2'
    var_16 = len(var_15)
    var_17 = 1
    var_18 = var_16 - var_17
    var_19 = [var_0]
    var_20 = [var_0]
    var_21 = len(var_15)
    var_22 = var_21 - var_17



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = [var_0]
    var_21 = [var_0]
    var_22 = len(var_16)
    var_23 = var_22 - var_18



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 6
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 13
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 18
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: Alice, age: 30'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = [var_0]
    var_24 = [var_0]
    var_25 = len(var_18)
    var_26 = var_25 - var_20
    var_27 = module_0.DictToken()



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
    var_6 = 1
    var_7 = 4
    var_8 = '1'
    var_9 = module_0.ScalarToken(var_6, var_7, var_7, var_8)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = module_0.ScalarToken(var_6, var_7, var_7, var_8)
    var_14 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_15 = {var_2: var_13, var_5: var_14}
    var_16 = 'a: 1, b: 2'
    var_17 = len(var_16)
    var_18 = var_17 - var_6
    var_19 = module_0.DictToken()
    var_20 = len(var_16)
    var_21 = var_20 - var_6
    var_22 = module_0.DictToken()



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 100
    var_5 = 6
    var_6 = 9
    var_7 = '100'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 11
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 17
    var_15 = 22
    var_16 = 'key1: 100, key2: hello'



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 'val2'
    var_12 = 18
    var_13 = 22
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_11)
    var_15 = 'key1: val1, key2: val2'



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 123
    var_13 = 18
    var_14 = 21
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'key1: val1, key2: 123'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = []
    var_23 = [var_0]
    var_24 = [var_0]



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 6
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 13
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 18
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_17 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_18 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_19 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_20 = {var_16: var_17, var_18: var_19}
    var_21 = {var_16: var_17, var_18: var_19}
    var_22 = 'name: Alice, age: 30'
    var_23 = len(var_22)
    var_24 = 1
    var_25 = var_23 - var_24
    var_26 = module_0.DictToken()
    var_27 = len(var_22)
    var_28 = var_27 - var_24

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 2
    var_5 = 3
    var_6 = '1'
    var_7 = module_0.ScalarToken(var_2, var_4, var_5, var_6)
    var_8 = 'b'
    var_9 = 4
    var_10 = 5
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 6
    var_13 = 7
    var_14 = '2'
    var_15 = module_0.ScalarToken(var_4, var_12, var_13, var_14)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'a: 1, b: 2'
    var_18 = len(var_17)
    var_19 = var_18 - var_2



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 123
    var_13 = 18
    var_14 = 21
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = 'key1: val1, key2: 123'
    var_18 = {var_3: var_7, var_11: var_16}
    var_19 = len(var_17)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_17)
    var_24 = var_23 - var_20



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = [var_0]
    var_18 = [var_0]



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 6
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 13
    var_10 = 16
    var_11 = 30
    var_12 = 18
    var_13 = 19
    var_14 = '30'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'name: Alice, age: 30'



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = {var_0: var_4, var_8: var_11}
    var_17 = 'key1: val1, key2: 123'
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = 'non_existent'
    var_22 = 'non_existent'
    var_23 = [var_22]
    var_24 = [var_22]



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 123
    var_13 = 18
    var_14 = 21
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'key1: val1, key2: 123'
    var_19 = 20



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key1_val'
    var_4 = 'key2'
    var_5 = 6
    var_6 = 10
    var_7 = 'key2_val'
    var_8 = 100
    var_9 = 5
    var_10 = 7
    var_11 = '100'
    var_12 = 'hello'
    var_13 = 11
    var_14 = 15
    var_15 = 'key1_val: 100, key2_val: hello'
    var_16 = len(var_15)
    var_17 = 1
    var_18 = var_16 - var_17
    var_19 = len(var_15)
    var_20 = var_19 - var_17



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 6
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 13
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 18
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = "name: 'Alice', age: 30"
    var_19 = module_0.DictToken()
    var_20 = module_0.DictToken()
    var_21 = repr(var_19)
    assert var_21 == 'DictToken("name: \'Alice\', age: 30")'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'hello world\nnext line'
    var_1 = 'hello'
    var_2 = 0
    var_3 = 4
    var_4 = 'world'
    var_5 = 6
    var_6 = 10
    var_7 = 1
    var_8 = 5
    var_9 = 'different content'



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'diff'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.ScalarToken(var_0, var_1, var_2, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = len(var_16)
    var_21 = var_20 - var_18



# Parsed testcases at query #6
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
    var_11 = 13
    var_12 = module_0.DictToken()
    var_13 = [var_1]
    var_14 = [var_1]
    var_15 = module_0.ScalarToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 10
    var_2 = 0
    var_3 = 1
    var_4 = 20
    var_5 = 2
    var_6 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_7 = module_0.ScalarToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'diff'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.ScalarToken(var_0, var_1, var_2, var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'val2'
    var_13 = 18
    var_14 = 22
    var_15 = 'key1: val1, key2: val2'
    var_16 = len(var_15)
    var_17 = 1
    var_18 = var_16 - var_17
    var_19 = len(var_15)
    var_20 = var_19 - var_17



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = [var_0]
    var_18 = [var_0]



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 10
    var_2 = 0
    var_3 = 1
    var_4 = 20
    var_5 = 2
    var_6 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_7 = module_0.ScalarToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = var_3._value
    var_17 = 'key1: val1, key2: 123'
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = [var_0]
    var_22 = [var_0]
    var_23 = len(var_17)
    var_24 = var_23 - var_19



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 'test'
    var_2 = 0
    var_3 = 3
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = 0
    var_18 = len(var_16)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = [var_0]
    var_22 = [var_0]

def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = 0
    var_3 = 'nonexistent'
    var_4 = 'nonexistent'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 123
    var_13 = 18
    var_14 = 21
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = var_3._value
    var_18 = var_11._value
    var_19 = {var_17: var_7, var_18: var_16}
    var_20 = 'key1: val1, key2: 123'
    var_21 = len(var_20)
    var_22 = 1
    var_23 = var_21 - var_22
    var_24 = module_0.DictToken()
    var_25 = len(var_20)
    var_26 = var_25 - var_22
    var_27 = module_0.DictToken()



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = []
    var_18 = [var_0]
    var_19 = [var_0]



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 'val2'
    var_12 = 18
    var_13 = 22
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_11)
    var_15 = 'key1: val1, key2: val2'
    var_16 = len(var_15)
    var_17 = 1
    var_18 = var_16 - var_17
    var_19 = [var_0]
    var_20 = [var_0]



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1, key2: 123'
    var_17 = 20
    var_18 = [var_0]
    var_19 = [var_0]
    var_20 = 'different content'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 123
    var_13 = 18
    var_14 = 21
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = var_3._value
    var_18 = var_11._value
    var_19 = {var_17: var_7, var_18: var_16}
    var_20 = 'key1: val1, key2: 123'



