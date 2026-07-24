####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = {}
    var_21 = 1
    var_22 = ''
    var_23 = 'single'
    var_24 = module_0.ScalarToken(var_23, var_1, var_5, var_23)
    var_25 = 'val'
    var_26 = 7
    var_27 = 9
    var_28 = 'singleval'
    var_29 = module_0.ScalarToken(var_25, var_26, var_27, var_28)
    var_30 = {var_24: var_29}



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'other'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 3
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = {}
    var_12 = 2
    var_13 = '{}'
    var_14 = module_0.DictToken()
    var_15 = {}
    var_16 = module_0.DictToken()
    var_17 = []
    var_18 = '[]'
    var_19 = module_0.ListToken(var_17, var_1, var_12, var_18)
    var_20 = 123
    var_21 = '123'
    var_22 = module_0.ScalarToken(var_20, var_1, var_12, var_21)
    var_23 = module_0.ScalarToken(var_20, var_1, var_12, var_21)
    var_24 = 123.5
    var_25 = '123.5'
    var_26 = module_0.ScalarToken(var_24, var_1, var_2, var_25)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = 'different'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_3)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_3)
    var_10 = 6
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_3)
    var_12 = 'item'
    var_13 = 3
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_15 = module_0.ScalarToken(var_12, var_1, var_13, var_12)
    var_16 = [var_14]
    var_17 = 'content'
    var_18 = module_0.ListToken(var_16, var_1, var_2, var_17)
    var_19 = [var_15]
    var_20 = module_0.ListToken(var_19, var_1, var_2, var_17)
    var_21 = 'key'
    var_22 = 2
    var_23 = module_0.ScalarToken(var_21, var_1, var_22, var_21)
    var_24 = 'val'
    var_25 = module_0.ScalarToken(var_24, var_13, var_2, var_24)
    var_26 = module_0.ScalarToken(var_21, var_1, var_22, var_21)
    var_27 = module_0.ScalarToken(var_24, var_13, var_2, var_24)
    var_28 = {var_23: var_25}
    var_29 = 8
    var_30 = 'key:val'
    var_31 = {var_26: var_27}
    var_32 = 'different_content'
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_32)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'other'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 3
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = 'different_content'
    var_12 = module_0.ScalarToken(var_0, var_1, var_2, var_11)
    var_13 = {}
    var_14 = 2
    var_15 = '{}'
    var_16 = module_0.DictToken()
    var_17 = {}
    var_18 = module_0.DictToken()
    var_19 = []
    var_20 = '[]'
    var_21 = module_0.ListToken(var_19, var_1, var_14, var_20)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 4
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 13
    var_12 = 'key2value'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 14
    var_16 = 19
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1valuekey2value'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 4
    var_6 = 7
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'nameJohn'



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 10
    var_18 = 'name: John\nage: 30'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 10

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'mykey'
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'myvalue'
    var_6 = 6
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 10

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'testkey'
    var_1 = 0
    var_2 = 6
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'testvalue'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 10

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'



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
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'
    var_17 = 'name'
    var_18 = 'name: John'
    var_19 = module_0.ScalarToken(var_17, var_1, var_2, var_18)
    var_20 = 'John'
    var_21 = 9
    var_22 = module_0.ScalarToken(var_20, var_6, var_21, var_18)
    var_23 = {var_19: var_22}
    var_24 = 10
    var_25 = {}
    var_26 = 2
    var_27 = '{}'



# Parsed testcases at query #8
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

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'nameJohn'



# Parsed testcases at query #9
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 18
    var_16 = 24
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = {}
    var_21 = ''



# Parsed testcases at query #10
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 10
    var_16 = 'key1: value1, key2: value2'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'inner'
    var_5 = module_0.ScalarToken(var_4, var_1, var_2, var_4)
    var_6 = 'data'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_1, var_7, var_6)
    var_9 = {var_5: var_8}
    var_10 = 10
    var_11 = '{inner: data}'
    var_12 = 15
    var_13 = 'outer: {inner: data}'



# Parsed testcases at query #11
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 0
    var_2 = 2
    var_3 = 'age: 30'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 30
    var_6 = 5
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'status'
    var_1 = 0
    var_2 = 5
    var_3 = 'status: active'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'active'
    var_6 = 8
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'a: 1, b: 2'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 3
    var_6 = module_0.ScalarToken(var_4, var_5, var_5, var_2)
    var_7 = 'b'
    var_8 = 6
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_2)
    var_10 = 2
    var_11 = 9
    var_12 = module_0.ScalarToken(var_10, var_11, var_11, var_2)
    var_13 = {var_3: var_6, var_9: var_12}
    var_14 = 10



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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = 'key2: value2'
    var_17 = module_0.ScalarToken(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 'key1: value1, key2: value2'
    var_20 = {var_0: var_4, var_9: var_13}
    var_21 = {}
    var_22 = '{}'
    var_23 = 'name'
    var_24 = 1
    var_25 = 4
    var_26 = '{name}'
    var_27 = module_0.ScalarToken(var_23, var_24, var_25, var_26)
    var_28 = 'John'
    var_29 = 7
    var_30 = '{name: John}'
    var_31 = module_0.ScalarToken(var_28, var_29, var_6, var_30)
    var_32 = {var_27: var_31}
    var_33 = 11



# Parsed testcases at query #13
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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = 11

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 16
    var_16 = 17
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'name: John, age: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'



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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 18
    var_15 = 23
    var_16 = 'key2: value2'
    var_17 = module_0.ScalarToken(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 'key1: value1, key2: value2'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = 'name: John'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = 10

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'inner'
    var_5 = 7
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'data'
    var_9 = 13
    var_10 = 16
    var_11 = 'inner: data'
    var_12 = module_0.ScalarToken(var_8, var_9, var_10, var_11)
    var_13 = {var_7: var_12}
    var_14 = 6
    var_15 = 17
    var_16 = '{inner: data}'
    var_17 = 18
    var_18 = 'outer: {inner: data}'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 16
    var_16 = 17
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 20
    var_20 = 'name: John\nage: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 0
    var_2 = 3
    var_3 = 'list: [1, 2]'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 7
    var_7 = '[1, 2]'
    var_8 = module_0.ScalarToken(var_5, var_6, var_6, var_7)
    var_9 = 2
    var_10 = 10
    var_11 = module_0.ScalarToken(var_9, var_10, var_10, var_7)
    var_12 = [var_8, var_11]
    var_13 = 6
    var_14 = 11
    var_15 = module_0.ListToken(var_12, var_13, var_14, var_7)
    var_16 = {var_4: var_15}



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 16
    var_16 = 17
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 10
    var_20 = 'name: John\nage: 30'

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
    var_17 = 'key1: value1\nkey2: value2'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 5
    var_6 = 9
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'name: Alice'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 3
    var_6 = 5
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = 'id: 123'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'data'
    var_5 = 7
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'outer'
    var_10 = 4
    var_11 = module_0.ScalarToken(var_9, var_1, var_10, var_9)
    var_12 = 'nested: data'
    var_13 = 'outer: nested: data'



# Parsed testcases at query #17
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = 'key1value1key2value2'
    var_21 = 19
    var_22 = {var_0: var_4, var_9: var_14}
    var_23 = {}
    var_24 = ''
    var_25 = 'name'
    var_26 = module_0.ScalarToken(var_25, var_1, var_2, var_25)
    var_27 = 'John'
    var_28 = 8
    var_29 = 'nameJohn'
    var_30 = module_0.ScalarToken(var_27, var_5, var_28, var_29)
    var_31 = {var_26: var_30}



# Parsed testcases at query #18
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
    var_17 = 25
    var_18 = 'key1: value1, key2: value2'
    var_19 = {}
    var_20 = ''
    var_21 = 'name'
    var_22 = module_0.ScalarToken(var_21, var_1, var_2, var_21)
    var_23 = 'John'
    var_24 = 8
    var_25 = module_0.ScalarToken(var_23, var_5, var_24, var_23)
    var_26 = {var_22: var_25}
    var_27 = 'name: John'



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
    var_17 = 'key1: value1, key2: value2'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'name: John'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'inner'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 42
    var_9 = 12
    var_10 = 13
    var_11 = '42'
    var_12 = module_0.ScalarToken(var_8, var_9, var_10, var_11)
    var_13 = {var_7: var_12}
    var_14 = 5
    var_15 = 14
    var_16 = 'inner: 42'
    var_17 = 'outer: inner: 42'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 10
    var_18 = 'name: John\nage: 30'



# Parsed testcases at query #21
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = 'name'
    var_21 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_22 = 'John'
    var_23 = 8
    var_24 = 'nameJohn'
    var_25 = module_0.ScalarToken(var_22, var_5, var_23, var_24)
    var_26 = {var_21: var_25}
    var_27 = {}
    var_28 = 1
    var_29 = '{}'



# Parsed testcases at query #22
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 18
    var_16 = 24
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = {}
    var_21 = ''
    var_22 = 'a'
    var_23 = module_0.ScalarToken(var_22, var_1, var_1, var_22)
    var_24 = 1
    var_25 = 'a1'
    var_26 = module_0.ScalarToken(var_24, var_24, var_24, var_25)
    var_27 = {var_23: var_26}



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
    var_17 = 25
    var_18 = 'key1value1key2value2'
    var_19 = {var_0: var_4, var_8: var_12}
    var_20 = {}
    var_21 = ''



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 10
    var_18 = 'name: John, age: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 0
    var_2 = 4
    var_3 = 'items: [1, 2]'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_6, var_6, var_3)
    var_8 = 2
    var_9 = 11
    var_10 = module_0.ScalarToken(var_8, var_9, var_9, var_3)
    var_11 = [var_7, var_10]
    var_12 = 7
    var_13 = 12
    var_14 = module_0.ListToken(var_11, var_12, var_13, var_3)
    var_15 = {var_4: var_14}



# Parsed testcases at query #25
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
    var_18 = {}
    var_19 = ''
    var_20 = 'name'
    var_21 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_22 = 'John'
    var_23 = 8
    var_24 = module_0.ScalarToken(var_22, var_5, var_23, var_22)
    var_25 = {var_21: var_24}
    var_26 = 'nameJohn'



# Parsed testcases at query #26
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
    var_18 = {}
    var_19 = ''
    var_20 = 'name'
    var_21 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_22 = 'Alice'
    var_23 = 9
    var_24 = module_0.ScalarToken(var_22, var_5, var_23, var_22)
    var_25 = {var_21: var_24}
    var_26 = 'nameAlice'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1=value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 'key2=value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'test_content'
    var_16 = {var_0: var_5, var_9: var_12}
    var_17 = {}
    var_18 = ''



# Parsed testcases at query #28
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
    var_12 = 'key1: value1, key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1, key2: value2'
    var_20 = 26
    var_21 = {}
    var_22 = ''
    var_23 = 'name'
    var_24 = 'name: john'
    var_25 = module_0.ScalarToken(var_23, var_1, var_2, var_24)
    var_26 = 'john'
    var_27 = 9
    var_28 = module_0.ScalarToken(var_26, var_6, var_27, var_24)
    var_29 = {var_25: var_28}



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = 'name: John'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'age'
    var_10 = 11
    var_11 = 13
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 30
    var_14 = 15
    var_15 = 16
    var_16 = 'age: 30'
    var_17 = module_0.ScalarToken(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 20
    var_20 = 'name: John, age: 30'
    var_21 = {}
    var_22 = 1
    var_23 = '{}'
    var_24 = 'key'
    var_25 = 2
    var_26 = module_0.ScalarToken(var_24, var_1, var_25, var_24)
    var_27 = 'value'
    var_28 = 9
    var_29 = 'key: value'
    var_30 = module_0.ScalarToken(var_27, var_5, var_28, var_29)
    var_31 = {var_26: var_30}
    var_32 = 10



# Parsed testcases at query #30
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 10
    var_10 = 12
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 14
    var_14 = 15
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = '10'
    var_7 = module_0.ScalarToken(var_3, var_4, var_5, var_6)
    var_8 = 'y'
    var_9 = 5
    var_10 = module_0.ScalarToken(var_8, var_9, var_9, var_8)
    var_11 = 20
    var_12 = 7
    var_13 = 8
    var_14 = '20'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = {var_2: var_7, var_10: var_15}
    var_17 = 'x: 10, y: 20'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'foo'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'bar'
    var_5 = 4
    var_6 = 6
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'foo: bar'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'inner: value'
    var_10 = 'outer'
    var_11 = 12
    var_12 = 16
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'outer: inner: value'
    var_15 = [var_10, var_0]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'parent'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'child'
    var_5 = 7
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'data'
    var_9 = 13
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = {var_7: var_11}
    var_13 = 'parent: child: data'
    var_14 = [var_0, var_4]

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 42
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7)
    var_9 = None
    var_10 = 'abc'
    var_11 = module_0.Token(var_9, var_1, var_1, var_10)
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = 'data'
    var_17 = module_0.Token(var_15, var_12, var_6, var_16)
    var_18 = 'x'
    var_19 = 'xyz'
    var_20 = module_0.Token(var_18, var_1, var_1, var_19)
    var_21 = 'large'
    var_22 = 100
    var_23 = 200
    var_24 = 'a'
    var_25 = 201
    var_26 = var_24 * var_25
    var_27 = module_0.Token(var_21, var_22, var_23, var_26)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 42
    var_6 = 5
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7)
    var_9 = None
    var_10 = 'null'
    var_11 = module_0.ScalarToken(var_9, var_1, var_1, var_10)
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = 9
    var_17 = '[1, 2, 3]'
    var_18 = module_0.ScalarToken(var_15, var_1, var_16, var_17)
    var_19 = 'key'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = 'dict'
    var_23 = module_0.ScalarToken(var_21, var_1, var_6, var_22)
    var_24 = 'a'
    var_25 = module_0.ScalarToken(var_24, var_1, var_1, var_24)
    var_26 = 'end'
    var_27 = 1000
    var_28 = 2000
    var_29 = 'x'
    var_30 = 2001
    var_31 = var_29 * var_30
    var_32 = module_0.ScalarToken(var_26, var_27, var_28, var_31)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 42
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = '[1,2,3]'
    var_13 = module_0.Token(var_11, var_1, var_10, var_12)
    var_14 = None
    var_15 = -1
    var_16 = 'content'
    var_17 = module_0.Token(var_14, var_15, var_1, var_16)
    var_18 = 'x'
    var_19 = 1000
    var_20 = 2000
    var_21 = 'a'
    var_22 = 2001
    var_23 = var_21 * var_22
    var_24 = module_0.Token(var_18, var_19, var_20, var_23)
    var_25 = 'empty'
    var_26 = ''
    var_27 = module_0.Token(var_25, var_1, var_1, var_26)
    var_28 = '0'
    var_29 = module_0.Token(var_1, var_1, var_1, var_28)
    var_30 = 'key'
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = 'dict_content'
    var_34 = module_0.Token(var_32, var_1, var_5, var_33)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = hash(var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = hash(var_3)
    var_7 = hash(var_5)
    var_8 = 'different_value'
    var_9 = module_0.ScalarToken(var_8, var_1, var_2, var_8)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = 42
    var_13 = 2
    var_14 = '42'
    var_15 = module_0.ScalarToken(var_12, var_1, var_13, var_14)
    var_16 = module_0.ScalarToken(var_14, var_1, var_13, var_14)
    var_17 = hash(var_15)
    var_18 = hash(var_16)
    var_19 = None
    var_20 = 4
    var_21 = 'None'
    var_22 = module_0.ScalarToken(var_19, var_1, var_20, var_21)
    var_23 = hash(var_22)
    var_24 = 3.14
    var_25 = '3.14'
    var_26 = module_0.ScalarToken(var_24, var_1, var_20, var_25)
    var_27 = hash(var_26)
    var_28 = {var_3, var_5}
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'value1'
    var_31 = {var_3: var_30}



# Parsed testcases at query #35
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 5
    var_6 = 9
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 11
    var_10 = 13
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 15
    var_14 = 16
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: Alice, age: 30'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 10
    var_4 = 2
    var_5 = 3
    var_6 = '10'
    var_7 = module_0.ScalarToken(var_3, var_4, var_5, var_6)
    var_8 = {var_2: var_7}
    var_9 = 'x: 10'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'test'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'data: test'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'inner: value'
    var_10 = 'outer'
    var_11 = 12
    var_12 = 16
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 'outer: inner: value'



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = [var_3]
    var_5 = 20
    var_6 = 'content'
    var_7 = module_0.ListToken(var_4, var_1, var_5, var_6)
    var_8 = 'key'
    var_9 = 2
    var_10 = module_0.ScalarToken(var_8, var_1, var_9, var_8)
    var_11 = {var_10: var_7}
    var_12 = 30
    var_13 = [var_8]
    var_14 = [var_8, var_1]
    var_15 = []
    var_16 = 'a'
    var_17 = 1
    var_18 = module_0.ScalarToken(var_16, var_1, var_17, var_16)
    var_19 = 'b'
    var_20 = module_0.ScalarToken(var_19, var_1, var_17, var_19)
    var_21 = [var_18, var_20]
    var_22 = module_0.ListToken(var_21, var_1, var_2, var_6)
    var_23 = [var_1]
    var_24 = [var_17]
    var_25 = [var_22]
    var_26 = module_0.ListToken(var_25, var_1, var_5, var_6)
    var_27 = [var_1, var_17]



# Parsed testcases at query #37
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 18
    var_16 = 24
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = 'nameJohn'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_6 = 'other'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_6)
    var_8 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_9 = 1
    var_10 = module_0.ScalarToken(var_0, var_9, var_2, var_0)
    var_11 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_12 = 4
    var_13 = module_0.ScalarToken(var_0, var_1, var_12, var_0)
    var_14 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_15 = []
    var_16 = '[]'
    var_17 = module_0.ListToken(var_15, var_1, var_9, var_16)
    var_18 = []
    var_19 = module_0.ListToken(var_18, var_1, var_9, var_16)
    var_20 = 'a'
    var_21 = module_0.ScalarToken(var_20, var_1, var_1, var_20)
    var_22 = [var_21]
    var_23 = '[a]'
    var_24 = module_0.ListToken(var_22, var_1, var_9, var_23)
    var_25 = []
    var_26 = module_0.ListToken(var_25, var_1, var_9, var_16)
    var_27 = {}
    var_28 = '{}'
    var_29 = module_0.DictToken()
    var_30 = {}
    var_31 = module_0.DictToken()



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = module_0.ScalarToken(var_4, var_5, var_5, var_2)
    var_7 = [var_3, var_6]
    var_8 = module_0.ListToken(var_7, var_1, var_5, var_2)
    var_9 = []
    var_10 = ''
    var_11 = module_0.ListToken(var_9, var_1, var_1, var_10)
    var_12 = 1
    var_13 = 5
    var_14 = '[1, 2, 3]'
    var_15 = module_0.ScalarToken(var_12, var_1, var_13, var_14)
    var_16 = 3
    var_17 = module_0.ScalarToken(var_5, var_16, var_13, var_14)
    var_18 = [var_15, var_17]
    var_19 = 8
    var_20 = module_0.ListToken(var_18, var_1, var_19, var_14)
    var_21 = 'x'
    var_22 = '[x]'
    var_23 = module_0.ScalarToken(var_21, var_12, var_12, var_22)
    var_24 = [var_23]
    var_25 = module_0.ListToken(var_24, var_1, var_5, var_22)
    var_26 = var_25._value
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 42
    var_6 = 5
    var_7 = 7
    var_8 = module_0.ScalarToken(var_5, var_6, var_7)
    var_9 = 'hello'
    var_10 = 'hello world'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'world'
    var_13 = 6
    var_14 = 10
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_10)
    var_16 = 123
    var_17 = 2
    var_18 = '123'
    var_19 = module_0.ScalarToken(var_16, var_1, var_17, var_18)
    var_20 = None
    var_21 = 3
    var_22 = 'null'
    var_23 = module_0.ScalarToken(var_20, var_1, var_21, var_22)
    var_24 = True
    var_25 = 'true'
    var_26 = module_0.ScalarToken(var_24, var_1, var_21, var_25)
    var_27 = 3.14
    var_28 = '3.14'
    var_29 = module_0.ScalarToken(var_27, var_1, var_21, var_28)
    var_30 = 'x'
    var_31 = module_0.ScalarToken(var_30, var_1, var_1, var_30)
    var_32 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_34 = 'different'
    var_35 = module_0.ScalarToken(var_34, var_1, var_2, var_3)



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'
    var_17 = {var_0: var_5, var_9: var_12}
    var_18 = {}
    var_19 = '{}'
    var_20 = 'nested'
    var_21 = 5
    var_22 = module_0.ScalarToken(var_20, var_1, var_21, var_20)
    var_23 = 'inner'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = 10
    var_27 = module_0.ScalarToken(var_25, var_1, var_26, var_20)
    var_28 = {var_22: var_27}
    var_29 = 15
    var_30 = 'nested: value'



# Parsed testcases at query #2
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
    var_18 = {}
    var_19 = ''



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = 'different'
    var_7 = module_0.ScalarToken(var_6, var_1, var_2, var_3)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_3)
    var_10 = 6
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_3)
    var_12 = 'a'
    var_13 = module_0.ScalarToken(var_12, var_1, var_1, var_12)
    var_14 = [var_13]
    var_15 = 'list1'
    var_16 = module_0.ListToken(var_14, var_1, var_2, var_15)
    var_17 = module_0.ScalarToken(var_12, var_1, var_1, var_12)
    var_18 = [var_17]
    var_19 = module_0.ListToken(var_18, var_1, var_2, var_15)
    var_20 = 'key'
    var_21 = 2
    var_22 = module_0.ScalarToken(var_20, var_1, var_21, var_20)
    var_23 = 'val'
    var_24 = 4
    var_25 = module_0.ScalarToken(var_23, var_24, var_10, var_23)
    var_26 = {var_22: var_25}
    var_27 = 10
    var_28 = 'dict1'
    var_29 = module_0.ScalarToken(var_20, var_1, var_21, var_20)
    var_30 = module_0.ScalarToken(var_23, var_24, var_10, var_23)
    var_31 = {var_29: var_30}



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_0)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 6
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = []
    var_12 = 'content'
    var_13 = module_0.ListToken(var_11, var_1, var_2, var_12)
    var_14 = []
    var_15 = module_0.ListToken(var_14, var_1, var_2, var_12)
    var_16 = {}
    var_17 = {}
    var_18 = module_0.ScalarToken(var_17, var_1, var_2, var_12)



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
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key1: value1, key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = {var_0: var_5, var_9: var_14}
    var_20 = {}
    var_21 = 2
    var_22 = '{}'
    var_23 = 'name'
    var_24 = 'name: John'
    var_25 = module_0.ScalarToken(var_23, var_1, var_2, var_24)
    var_26 = 'John'
    var_27 = 9
    var_28 = module_0.ScalarToken(var_26, var_6, var_27, var_24)
    var_29 = {var_25: var_28}



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 10
    var_18 = 'name: John\nage: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

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



# Parsed testcases at query #7
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
    var_18 = {}
    var_19 = ''
    var_20 = 'single'
    var_21 = module_0.ScalarToken(var_20, var_1, var_5, var_20)
    var_22 = 42
    var_23 = 7
    var_24 = 8
    var_25 = '42'
    var_26 = module_0.ScalarToken(var_22, var_23, var_24, var_25)
    var_27 = {var_21: var_26}
    var_28 = 'single: 42'



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'name: value, age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 17
    var_16 = 18
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 19
    var_20 = {var_0: var_5, var_9: var_14}
    var_21 = {}
    var_22 = 1
    var_23 = '{}'
    var_24 = 'key'
    var_25 = 2
    var_26 = 'key: val'
    var_27 = module_0.ScalarToken(var_24, var_1, var_25, var_26)
    var_28 = 'val'
    var_29 = 5
    var_30 = 7
    var_31 = module_0.ScalarToken(var_28, var_29, var_30, var_26)
    var_32 = {var_27: var_31}



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_0)
    var_10 = 3
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_0)
    var_12 = {}
    var_13 = ''
    var_14 = {}
    var_15 = []
    var_16 = module_0.ListToken(var_15, var_1, var_1, var_13)
    var_17 = 'key1'
    var_18 = module_0.ScalarToken(var_17, var_1, var_10, var_17)
    var_19 = 'val1'
    var_20 = 5
    var_21 = module_0.ScalarToken(var_19, var_20, var_6, var_19)
    var_22 = {var_18: var_21}
    var_23 = 'key1val1'
    var_24 = module_0.ScalarToken(var_17, var_1, var_10, var_17)
    var_25 = module_0.ScalarToken(var_19, var_20, var_6, var_19)
    var_26 = {var_24: var_25}
    var_27 = 'item'
    var_28 = module_0.ScalarToken(var_27, var_1, var_10, var_27)
    var_29 = [var_28]
    var_30 = module_0.ListToken(var_29, var_1, var_10, var_27)
    var_31 = module_0.ScalarToken(var_27, var_1, var_10, var_27)
    var_32 = [var_31]
    var_33 = module_0.ListToken(var_32, var_1, var_10, var_27)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 17
    var_16 = 18
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 20
    var_20 = 'name: John, age: 30'
    var_21 = {}
    var_22 = 1
    var_23 = '{}'
    var_24 = 'key'
    var_25 = 2
    var_26 = 'key: value'
    var_27 = module_0.ScalarToken(var_24, var_1, var_25, var_26)
    var_28 = 'value'
    var_29 = 5
    var_30 = module_0.ScalarToken(var_28, var_29, var_7, var_26)
    var_31 = {var_27: var_30}



# Parsed testcases at query #11
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'key1: value1, key2: value2'
    var_16 = {}
    var_17 = ''
    var_18 = 'name'
    var_19 = 'name: John'
    var_20 = module_0.ScalarToken(var_18, var_1, var_2, var_19)
    var_21 = 'John'
    var_22 = 9
    var_23 = module_0.ScalarToken(var_21, var_6, var_22, var_19)
    var_24 = {var_20: var_23}



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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 18
    var_15 = 23
    var_16 = 'key2: value2'
    var_17 = module_0.ScalarToken(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 'key1: value1, key2: value2'
    var_20 = {var_0: var_4, var_9: var_13}
    var_21 = {}
    var_22 = 1
    var_23 = '{}'
    var_24 = 'nested'
    var_25 = module_0.ScalarToken(var_24, var_1, var_5, var_24)
    var_26 = 42
    var_27 = 7
    var_28 = 8
    var_29 = '42'
    var_30 = module_0.ScalarToken(var_26, var_27, var_28, var_29)
    var_31 = {var_25: var_30}
    var_32 = 'nested: 42'



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 12
    var_11 = 14
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 16
    var_16 = 17
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 20
    var_20 = 'name: John, age: 30'
    var_21 = {}
    var_22 = ''
    var_23 = 'nested'
    var_24 = 5
    var_25 = 'nested: value'
    var_26 = module_0.ScalarToken(var_23, var_1, var_24, var_25)
    var_27 = 'data'
    var_28 = 8
    var_29 = 11
    var_30 = 'nested: data'
    var_31 = module_0.ScalarToken(var_27, var_28, var_29, var_30)
    var_32 = {var_26: var_31}



# Parsed testcases at query #14
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
    var_10 = 14
    var_11 = 17
    var_12 = 'key1: value1, key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 20
    var_16 = 25
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = {}
    var_20 = 1
    var_21 = '{}'
    var_22 = 'name'
    var_23 = 'name: John'
    var_24 = module_0.ScalarToken(var_22, var_1, var_2, var_23)
    var_25 = 'John'
    var_26 = 9
    var_27 = module_0.ScalarToken(var_25, var_6, var_26, var_23)
    var_28 = {var_24: var_27}



# Parsed testcases at query #15
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'key1: value1, key2: value2'
    var_16 = {}
    var_17 = 1
    var_18 = '{}'
    var_19 = 'name'
    var_20 = 'name: Alice'
    var_21 = module_0.ScalarToken(var_19, var_1, var_2, var_20)
    var_22 = 'Alice'
    var_23 = 10
    var_24 = module_0.ScalarToken(var_22, var_6, var_23, var_20)
    var_25 = {var_21: var_24}



# Parsed testcases at query #16
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

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'nameJohn'



# Parsed testcases at query #17
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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 17
    var_12 = 'key1: value1, key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 25
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_3: var_8, var_13: var_17}
    var_19 = {}
    var_20 = ''
    var_21 = 'single'
    var_22 = module_0.ScalarToken(var_21, var_1, var_5, var_21)
    var_23 = 'val'
    var_24 = 7
    var_25 = 9
    var_26 = 'single: val'
    var_27 = module_0.ScalarToken(var_23, var_24, var_25, var_26)
    var_28 = {var_22: var_27}
    var_29 = 10



# Parsed testcases at query #18
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
    var_18 = {}
    var_19 = ''
    var_20 = 'single'
    var_21 = module_0.ScalarToken(var_20, var_1, var_5, var_20)
    var_22 = 42
    var_23 = 7
    var_24 = 8
    var_25 = '42'
    var_26 = module_0.ScalarToken(var_22, var_23, var_24, var_25)
    var_27 = {var_21: var_26}
    var_28 = 'single: 42'



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'Alice'
    var_6 = 6
    var_7 = 10
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 13
    var_11 = 15
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 18
    var_16 = 19
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 20
    var_20 = 'name: Alice\nage: 30'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'val1'
    var_6 = 5
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 13
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_3)
    var_13 = 42
    var_14 = 15
    var_15 = 16
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 20

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'Bob'
    var_6 = 5
    var_7 = 7
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 10

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'identifier'
    var_1 = 0
    var_2 = 9
    var_3 = 'content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'xyz'
    var_6 = 12
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 15

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'



# Parsed testcases at query #20
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
    var_7 = 'key1: value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = 'key2: value2'
    var_17 = module_0.ScalarToken(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 'key1: value1, key2: value2'
    var_20 = 26

def test_case_0():
    var_0 = {}
    var_1 = ''
    var_2 = 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 5
    var_6 = 8
    var_7 = 'name: John'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = 'name: John'
    var_11 = 9

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'inner'
    var_5 = 7
    var_6 = 11
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'data'
    var_9 = 14
    var_10 = 17
    var_11 = 'inner: data'
    var_12 = module_0.ScalarToken(var_8, var_9, var_10, var_11)
    var_13 = {var_7: var_12}
    var_14 = 6
    var_15 = 18
    var_16 = 'outer: {inner: data}'
    var_17 = 'outer: {inner: data}'
    var_18 = 19



# Parsed testcases at query #21
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
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = 'name'
    var_21 = module_0.ScalarToken(var_20, var_1, var_2, var_20)
    var_22 = 'John'
    var_23 = 8
    var_24 = 'nameJohn'
    var_25 = module_0.ScalarToken(var_22, var_5, var_23, var_24)
    var_26 = {var_21: var_25}
    var_27 = {}
    var_28 = ''



# Parsed testcases at query #22
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = 'key2'
    var_8 = module_0.ScalarToken(var_7, var_1, var_2, var_7)
    var_9 = 42
    var_10 = 1
    var_11 = '42'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = {var_3: var_6, var_8: var_12}
    var_14 = 10
    var_15 = 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'mykey'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'myvalue'
    var_5 = 6
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = {var_3: var_6}
    var_8 = 10
    var_9 = 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'mykey'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'myvalue'
    var_5 = 6
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = {var_3: var_6}
    var_8 = 10
    var_9 = 'content'



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
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = {}
    var_21 = ''



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 20
    var_18 = 'name: John, age: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

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



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 2
    var_11 = 'age: 30'
    var_12 = module_0.ScalarToken(var_9, var_1, var_10, var_11)
    var_13 = 30
    var_14 = 5
    var_15 = module_0.ScalarToken(var_13, var_14, var_6, var_11)
    var_16 = {var_4: var_8, var_12: var_15}
    var_17 = 10
    var_18 = 'name: John, age: 30'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = {var_3: var_6}
    var_8 = 10
    var_9 = 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_key'
    var_1 = 0
    var_2 = 7
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'test_value'
    var_5 = 9
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = {var_3: var_6}
    var_8 = 10
    var_9 = 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'mykey'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'myvalue'
    var_5 = 6
    var_6 = module_0.ScalarToken(var_4, var_1, var_5, var_4)
    var_7 = {var_3: var_6}
    var_8 = 10
    var_9 = 'content'



# Parsed testcases at query #26
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'
    var_17 = {}
    var_18 = '{}'
    var_19 = 'name'
    var_20 = 'name: John'
    var_21 = module_0.ScalarToken(var_19, var_1, var_2, var_20)
    var_22 = 'John'
    var_23 = 9
    var_24 = module_0.ScalarToken(var_22, var_6, var_23, var_20)
    var_25 = {var_21: var_24}



# Parsed testcases at query #27
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
    var_18 = {var_0: var_4, var_8: var_12}
    var_19 = {}
    var_20 = ''



# Parsed testcases at query #28
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'key1: value1, key2: value2'
    var_16 = {}
    var_17 = ''
    var_18 = 'name'
    var_19 = 'name: John'
    var_20 = module_0.ScalarToken(var_18, var_1, var_2, var_19)
    var_21 = 'John'
    var_22 = 9
    var_23 = module_0.ScalarToken(var_21, var_6, var_22, var_19)
    var_24 = {var_20: var_23}



# Parsed testcases at query #29
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
    var_10 = 'key2: value2'
    var_11 = module_0.ScalarToken(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.ScalarToken(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 20
    var_16 = 'key1: value1, key2: value2'
    var_17 = {}
    var_18 = 2
    var_19 = '{}'
    var_20 = {var_0: var_5, var_9: var_12}
    var_21 = 'name'
    var_22 = 'name: John'
    var_23 = module_0.ScalarToken(var_21, var_1, var_2, var_22)
    var_24 = 'John'
    var_25 = 9
    var_26 = module_0.ScalarToken(var_24, var_6, var_25, var_22)
    var_27 = {var_23: var_26}



# Parsed testcases at query #30
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
    var_18 = {}
    var_19 = ''
    var_20 = 'nested'
    var_21 = module_0.ScalarToken(var_20, var_1, var_5, var_20)
    var_22 = 'item'
    var_23 = module_0.ScalarToken(var_22, var_1, var_2, var_22)
    var_24 = [var_23]
    var_25 = module_0.ListToken(var_24, var_1, var_2, var_22)
    var_26 = {var_21: var_25}



