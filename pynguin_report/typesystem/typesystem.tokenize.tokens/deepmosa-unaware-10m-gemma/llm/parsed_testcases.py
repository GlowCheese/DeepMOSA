####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 17
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'different_content'
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4
    var_7 = 'value'
    var_8 = 'start_index'
    var_9 = 'end_index'
    var_10 = {var_7: var_0, var_8: var_1, var_9: var_2}



# Parsed testcases at query #3
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
    var_24 = 'diff'
    var_25 = module_0.ScalarToken(var_24, var_1, var_2, var_24)
    var_26 = {var_3: var_25}
    var_27 = 'key1: diff'
    var_28 = module_0.DictToken()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'hello world'
    var_1 = 'test'
    var_2 = 0
    var_3 = 3
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4
    var_7 = 'not a token'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'hello\nworld'
    var_1 = 'test'
    var_2 = 0
    var_3 = 3
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 'val'
    var_2 = 0
    var_3 = 3
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 'hello'
    var_2 = 0
    var_3 = 4
    var_4 = 'world'
    var_5 = 6
    var_6 = 10
    var_7 = 1
    var_8 = 5
    var_9 = 123
    var_10 = 2
    var_11 = '123'
    var_12 = module_0.ScalarToken(var_9, var_2, var_10, var_11)
    var_13 = module_0.ScalarToken(var_9, var_2, var_10, var_11)
    var_14 = 456
    var_15 = '456'
    var_16 = module_0.ScalarToken(var_14, var_2, var_10, var_15)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 'val'
    var_2 = 0
    var_3 = 3
    var_4 = 'diff'
    var_5 = 1
    var_6 = 4
    var_7 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_8 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_9 = 'other'



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 10
    var_2 = 0
    var_3 = 1
    var_4 = 'different content'
    var_5 = 20
    var_6 = 2
    var_7 = 5
    var_8 = '5'
    var_9 = module_0.ScalarToken(var_7, var_2, var_2, var_8)
    var_10 = module_0.ScalarToken(var_7, var_2, var_2, var_8)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = '10'
    var_4 = 20
    var_5 = '20'
    var_6 = 1
    var_7 = 3
    var_8 = '101'
    var_9 = module_0.ScalarToken(var_0, var_1, var_2, var_3)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.ScalarToken(var_3, var_1, var_2, var_0)
    var_5 = hash(var_4)
    var_6 = hash(var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 6
    var_2 = 9
    var_3 = 'line'
    var_4 = module_0.ScalarToken(var_3, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'different content'
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_4)
    var_6 = 'other'
    var_7 = 5
    var_8 = module_0.ScalarToken(var_6, var_1, var_7, var_6)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "['item1', 'item2']"
    var_1 = 'item1'
    var_2 = 2
    var_3 = 7
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 10
    var_7 = 15
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_1, var_5]
    var_10 = 0
    var_11 = 16
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = 1
    var_14 = [var_10]
    var_15 = [var_13]
    var_16 = [var_1, var_5]
    var_17 = module_0.ListToken(var_16, var_10, var_11, var_0)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 5
    var_5 = 8
    var_6 = module_0.ScalarToken(var_0, var_4, var_5, var_0)
    var_7 = 'other'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_6)
    var_12 = hash(var_3)
    var_13 = hash(var_9)
    var_14 = {var_3, var_6, var_9}
    var_15 = len(var_14)
    assert var_15 == 2



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "['item1', 'item2']"
    var_1 = 'item1'
    var_2 = 2
    var_3 = 7
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 11
    var_7 = 16
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_4, var_8]
    var_10 = 0
    var_11 = 17
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = 1
    var_14 = [var_4, var_8]
    var_15 = module_0.ListToken(var_14, var_10, var_11, var_0)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 1
    var_3 = '10'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = 6
    var_7 = module_0.ScalarToken(var_0, var_5, var_6, var_3)
    var_8 = 20
    var_9 = '20'
    var_10 = module_0.ScalarToken(var_8, var_1, var_2, var_9)
    var_11 = hash(var_4)
    var_12 = hash(var_7)
    var_13 = hash(var_4)
    var_14 = hash(var_10)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 5
    var_5 = 7
    var_6 = 'xyzabc'
    var_7 = module_0.ScalarToken(var_0, var_4, var_5, var_6)
    var_8 = 'def'
    var_9 = module_0.ScalarToken(var_8, var_1, var_2, var_8)
    var_10 = hash(var_3)
    var_11 = hash(var_7)
    var_12 = hash(var_3)
    var_13 = hash(var_9)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "val1", "key2": {"subkey": "subval"}}'
    var_1 = 'val1'
    var_2 = 9
    var_3 = 13
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'subval'
    var_6 = 35
    var_7 = 41
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key1'
    var_10 = 1
    var_11 = 4
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'key2'
    var_14 = 17
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = 'subkey'
    var_18 = 23
    var_19 = 28
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_0)
    var_21 = {var_20: var_8}
    var_22 = {var_20: var_8}
    var_23 = [var_9]
    var_24 = [var_13, var_17]
    var_25 = 'nonexistent'
    var_26 = [var_25]
    var_27 = 'key2'
    var_28 = 'nonexistent'
    var_29 = [var_27, var_28]



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.ScalarToken(var_3, var_1, var_2, var_0)
    var_5 = "ScalarToken('hello')"
    var_6 = repr(var_4)
    var_7 = 'line1\nline2'
    var_8 = 6
    var_9 = 10
    var_10 = 'line2'
    var_11 = module_0.ScalarToken(var_10, var_8, var_9, var_7)
    var_12 = "ScalarToken('line2')"
    var_13 = repr(var_11)
    var_14 = ''
    var_15 = 0
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_14)
    var_17 = repr(var_16)
    assert var_17 == "ScalarToken('')"



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "['item1', 'item2']"
    var_1 = 'item1'
    var_2 = 2
    var_3 = 7
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 11
    var_7 = 16
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_4, var_8]
    var_10 = 0
    var_11 = len(var_0)
    var_12 = 1
    var_13 = var_11 - var_12
    var_14 = module_0.ListToken(var_9, var_10, var_13, var_0)
    var_15 = repr(var_14)
    assert var_15 == 'ListToken("[\'item1\', \'item2\']")'
    var_16 = len(var_0)
    var_17 = var_16 - var_12
    var_18 = module_0.ListToken(var_9, var_10, var_17, var_0)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('123')"
    var_6 = 'hello'
    var_7 = 5
    var_8 = 9
    var_9 = 'abcdehellofg'
    var_10 = module_0.ScalarToken(var_6, var_7, var_8, var_9)
    var_11 = repr(var_10)
    assert var_11 == "ScalarToken('hello')"
    var_12 = 'a'
    var_13 = module_0.ScalarToken(var_12, var_1, var_1, var_12)
    var_14 = 1
    var_15 = '1'
    var_16 = module_0.ScalarToken(var_14, var_2, var_2, var_15)
    var_17 = 'b'
    var_18 = 4
    var_19 = module_0.ScalarToken(var_17, var_18, var_18, var_17)
    var_20 = 6
    var_21 = '2'
    var_22 = module_0.ScalarToken(var_2, var_20, var_20, var_21)
    var_23 = {var_13: var_16, var_19: var_22}
    var_24 = 'a: 1, b: 2'
    var_25 = module_0.DictToken()
    var_26 = repr(var_25)
    assert var_26 == "DictToken('a: 1, b: 2')"
    var_27 = [var_16, var_22]
    var_28 = '[1, 2]'
    var_29 = module_0.ListToken(var_27, var_1, var_20, var_28)
    var_30 = repr(var_29)
    assert var_30 == "ListToken('[1, 2]')"
    var_31 = ''
    var_32 = -1
    var_33 = module_0.ScalarToken(var_31, var_1, var_32, var_31)
    var_34 = repr(var_33)
    assert var_34 == "ScalarToken('')"



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('123')"
    var_6 = 'hello world'
    var_7 = 'hello'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_6)
    var_10 = repr(var_9)
    assert var_10 == "ScalarToken('hello')"
    var_11 = 'test'
    var_12 = 3
    var_13 = 'abc'
    var_14 = 1
    var_15 = 'xabcy'
    var_16 = module_0.ScalarToken(var_13, var_14, var_2, var_15)
    var_17 = repr(var_16)
    assert var_17 == "ScalarToken('ab')"



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "['item1', 'item2']"
    var_1 = 'item1'
    var_2 = 7
    var_3 = 12
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 15
    var_7 = 20
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_4, var_8]
    var_10 = 1
    var_11 = 17
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = 0
    var_14 = [var_13]



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = 'abc'
    var_5 = 1
    var_6 = 2
    var_7 = 'abcdef'
    var_8 = 'world'
    var_9 = 999
    var_10 = 'hello'
    var_11 = None
    var_12 = 0
    var_13 = 'a\nb'



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '["item1", "item2"]'
    var_1 = 'item1'
    var_2 = 7
    var_3 = 12
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 14
    var_7 = 19
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_4, var_8]
    var_10 = 0
    var_11 = 18
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = 1



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'hello\nworld'
    var_1 = 'hello'
    var_2 = 0
    var_3 = 4
    var_4 = 'world'
    var_5 = 6
    var_6 = 10
    var_7 = 'quote"'
    var_8 = 'quote" '
    var_9 = ''
    var_10 = -1



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 10
    var_2 = 0
    var_3 = 4
    var_4 = 'different content but same slice'
    var_5 = 20
    var_6 = 1
    var_7 = 5
    var_8 = module_0.ScalarToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "['a', 'b']"
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'b'
    var_6 = 7
    var_7 = 8
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_1, var_5]
    var_10 = 0
    var_11 = module_0.ListToken(var_9, var_10, var_7, var_0)
    var_12 = 1
    var_13 = [var_10]



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = repr(var_3)
    var_5 = print(var_4)
    var_6 = 'hello'
    var_7 = 4
    var_8 = 'hello world'
    var_9 = module_0.ScalarToken(var_6, var_5, var_7, var_8)
    var_10 = repr(var_9)
    assert var_10 == "ScalarToken('hello')"
    var_11 = 'key'
    var_12 = 2
    var_13 = 'key: value'
    var_14 = module_0.ScalarToken(var_11, var_5, var_12, var_13)
    var_15 = 'value'
    var_16 = 6
    var_17 = 10
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_13)
    var_19 = {var_14: var_18}
    var_20 = module_0.DictToken()
    var_21 = repr(var_20)
    assert var_21 == "DictToken('key: value')"
    var_22 = [var_18]
    var_23 = module_0.ListToken(var_22, var_16, var_17, var_13)
    var_24 = repr(var_23)
    assert var_24 == "ListToken('value')"
    var_25 = ''
    var_26 = -1
    var_27 = module_0.ScalarToken(var_25, var_5, var_26, var_25)
    var_28 = repr(var_27)
    assert var_28 == "ScalarToken('')"
    var_29 = 'mid'
    var_30 = 3
    var_31 = 'abcde'
    var_32 = module_0.ScalarToken(var_29, var_4, var_30, var_31)
    var_33 = repr(var_32)
    assert var_33 == "ScalarToken('bcd')"



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1: value1\nkey2: value2'
    var_1 = 'value1'
    var_2 = 6
    var_3 = 12
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value2'
    var_6 = 19
    var_7 = 25
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key1'
    var_10 = 0
    var_11 = 4
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'key2'
    var_14 = 13
    var_15 = 17
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_12: var_4, var_16: var_8}
    var_18 = 'other'
    var_19 = 30
    var_20 = 35
    var_21 = 'key1: value1\nkey2: value2\nother'
    var_22 = module_0.ScalarToken(var_18, var_19, var_20, var_21)
    var_23 = [var_9]
    var_24 = [var_13]
    var_25 = [var_10]
    var_26 = 1
    var_27 = [var_26]
    var_28 = [var_10, var_9]
    var_29 = [var_9]
    var_30 = 'nonexistent'
    var_31 = [var_30]
    var_32 = 99
    var_33 = [var_32]



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'leaf'
    var_1 = 10
    var_2 = 14
    var_3 = 'root_content_leaf'
    var_4 = 'key'
    var_5 = 5
    var_6 = 8
    var_7 = 'root_content_key'
    var_8 = 'child'
    var_9 = 4
    var_10 = 9
    var_11 = 'root_content_child'
    var_12 = '0'
    var_13 = 'root'
    var_14 = 0
    var_15 = 18
    var_16 = 'root_content_child_key'
    var_17 = 'list_node'
    var_18 = []
    var_19 = [var_17]
    var_20 = [var_17, var_12]
    var_21 = 'non_existent'
    var_22 = [var_21]
    var_23 = [var_17]



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = hash(var_4)
    var_6 = hash(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 10
    var_2 = 14
    var_3 = 'line1\nline2: hello'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.start



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '["item1", "item2"]'
    var_1 = 'item1'
    var_2 = 2
    var_3 = 7
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'item2'
    var_6 = 10
    var_7 = 15
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = [var_1, var_5]
    var_10 = 0
    var_11 = 16
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = 1



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 4
    var_3 = '12345'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_6 = 999
    var_7 = 0
    var_8 = 2
    var_9 = '999'
    var_10 = module_0.ScalarToken(var_6, var_7, var_8, var_9)
    var_11 = repr(var_4)
    assert var_11 == "ScalarToken('1234')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = None
    var_2 = 7
    var_3 = module_0.ScalarToken(var_1, var_2, var_2, var_0)
    var_4 = var_3.start

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3.value
    var_5 = 'key'
    var_6 = 'key'



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 6
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 16
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = var_3._value
    var_18 = var_12._value
    var_19 = {var_17: var_8, var_18: var_16}
    var_20 = 'key1: 123, key2: hello'
    var_21 = 21
    var_22 = [var_0]
    var_23 = [var_0]
    var_24 = 'key1: 123, key2: hello'



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'foo\nbar'
    var_1 = 'foo'
    var_2 = 0
    var_3 = 2
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 4
    var_6 = 6
    var_7 = module_0.ScalarToken(var_1, var_5, var_6, var_0)
    var_8 = hash(var_4)
    var_9 = hash(var_7)
    var_10 = 'bar'
    var_11 = module_0.ScalarToken(var_10, var_5, var_6, var_0)
    var_12 = hash(var_4)
    var_13 = hash(var_11)
    var_14 = 123
    var_15 = '123'
    var_16 = module_0.ScalarToken(var_14, var_2, var_3, var_15)
    var_17 = hash(var_16)
    var_18 = module_0.ScalarToken(var_14, var_2, var_3, var_15)
    var_19 = hash(var_18)
    var_20 = None
    var_21 = 'n'
    var_22 = module_0.ScalarToken(var_20, var_2, var_2, var_21)
    var_23 = hash(var_22)
    var_24 = 'x'
    var_25 = module_0.ScalarToken(var_20, var_2, var_2, var_24)
    var_26 = hash(var_25)
    var_27 = {var_4, var_7, var_11}
    var_28 = len(var_27)
    assert var_28 == 2



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'different_content'
    var_4 = 'other'
    var_5 = 1
    var_6 = 4
    var_7 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_8 = 'diff'
    var_9 = module_0.ScalarToken(var_8, var_1, var_2, var_0)



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = [var_0]

def test_case_0():
    var_0 = 'only_one'
    var_1 = [var_0]
    var_2 = []



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 'foo'
    var_2 = 0
    var_3 = 2
    var_4 = 'bar'
    var_5 = 1
    var_6 = 3



# Parsed testcases at query #3
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
    var_16 = 'a'
    var_17 = module_0.ScalarToken(var_16, var_1, var_1, var_16)
    var_18 = 1
    var_19 = 2
    var_20 = '1'
    var_21 = module_0.ScalarToken(var_18, var_19, var_19, var_20)
    var_22 = 'b'
    var_23 = module_0.ScalarToken(var_22, var_2, var_2, var_22)
    var_24 = '2'
    var_25 = module_0.ScalarToken(var_19, var_5, var_5, var_24)
    var_26 = {var_17: var_21, var_23: var_25}
    var_27 = 'a: 1\nb: 2'
    var_28 = len(var_27)
    var_29 = var_28 - var_18
    var_30 = module_0.DictToken()
    var_31 = len(var_27)
    var_32 = var_31 - var_18
    var_33 = module_0.DictToken()



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
    var_7 = 3
    var_8 = '10-'
    var_9 = 'ABC'
    var_10 = 'a'
    var_11 = {var_10: var_6}
    var_12 = 5
    var_13 = "{'a': 1}"
    var_14 = {var_10: var_6}
    var_15 = 'different'



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
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()



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
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1\nkey2: 123'
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = len(var_16)
    var_21 = var_20 - var_18
    var_22 = 'diff'
    var_23 = module_0.ScalarToken(var_22, var_1, var_2, var_22)
    var_24 = {var_3: var_23}
    var_25 = 5
    var_26 = 'key1: diff'
    var_27 = module_0.DictToken()
    var_28 = [var_0]
    var_29 = [var_0]



# Parsed testcases at query #7
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
    var_7 = 'name: Alice'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'age'
    var_10 = 13
    var_11 = 16
    var_12 = 'age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 17
    var_16 = 19
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_3: var_8, var_13: var_17}
    var_19 = 'name: Alice\nage: 30'
    var_20 = len(var_19)
    var_21 = 1
    var_22 = var_20 - var_21
    var_23 = module_0.DictToken()
    var_24 = len(var_19)
    var_25 = var_24 - var_21
    var_26 = module_0.DictToken()



# Parsed testcases at query #8
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
    var_16 = 'key1: val1\nkey2: 123'



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
    var_11 = 123
    var_12 = 18
    var_13 = 21
    var_14 = '123'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'key1: val1\nkey2: 123'
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = []



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
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = [var_0]
    var_21 = [var_0]
    var_22 = len(var_16)
    var_23 = var_22 - var_18



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: 1, b: 2'
    var_16 = module_0.DictToken()



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 6
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 16
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = {var_3: var_8, var_12: var_16}
    var_18 = 'key1: 123, key2: hello'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()
    var_26 = 'non_existent'
    var_27 = 'non_existent'



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a: 1\nb: 2'
    var_15 = 9
    var_16 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_17 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_18 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_19 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_20 = {var_16: var_17, var_18: var_19}
    var_21 = 'a: 1\nb: 2'
    var_22 = [var_0]
    var_23 = [var_0]



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 17
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()
    var_26 = [var_0]
    var_27 = [var_0]



# Parsed testcases at query #15
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
    var_12 = 25
    var_13 = 18
    var_14 = 19
    var_15 = '25'
    var_16 = module_0.ScalarToken(var_15, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: Alice, age: 25'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = len(var_18)
    var_23 = var_22 - var_20



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
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 'val2'
    var_13 = 18
    var_14 = 22
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = 'key1: val1, key2: val2'
    var_17 = {var_3: var_7, var_11: var_15}
    var_18 = len(var_16)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = module_0.DictToken()



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
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = 'different content but same indices'
    var_26 = module_0.DictToken()
    var_27 = [var_0]
    var_28 = [var_0]



# Parsed testcases at query #18
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
    var_14 = 20
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



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 17
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'



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
    var_11 = 'a'
    var_12 = module_0.ScalarToken(var_11, var_1, var_1, var_11)
    var_13 = 1
    var_14 = 2
    var_15 = '1'
    var_16 = module_0.ScalarToken(var_13, var_14, var_14, var_15)
    var_17 = 'b'
    var_18 = module_0.ScalarToken(var_17, var_2, var_2, var_17)
    var_19 = '2'
    var_20 = module_0.ScalarToken(var_14, var_5, var_5, var_19)
    var_21 = {var_12: var_16, var_18: var_20}
    var_22 = 'a 1 b 2'
    var_23 = 'a 1 b 2'
    var_24 = 'nonexistent'
    var_25 = 'nonexistent'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = module_0.ScalarToken(var_4, var_5, var_5, var_2)
    var_7 = 'b'
    var_8 = 4
    var_9 = 'abc\ndef'
    var_10 = module_0.ScalarToken(var_7, var_8, var_8, var_9)
    var_11 = 6
    var_12 = module_0.ScalarToken(var_5, var_11, var_11, var_9)
    var_13 = {var_3: var_6, var_10: var_12}
    var_14 = 'abc\ndef'



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "val1", "key2": 10}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 5
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'val1'
    var_6 = 8
    var_7 = 12
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 16
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 10
    var_14 = 22
    var_15 = 23
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = 25
    var_20 = [var_1]
    var_21 = [var_1]



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "val1", "key2": 123}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 5
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'val1'
    var_6 = 8
    var_7 = 12
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 16
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 123
    var_14 = 22
    var_15 = 24
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = len(var_0)
    var_20 = var_19 - var_2
    var_21 = module_0.DictToken()
    var_22 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_23 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_24 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_25 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = len(var_0)
    var_28 = var_27 - var_2
    var_29 = module_0.DictToken()



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a: 1, b: 2'
    var_15 = 9
    var_16 = 'a: 1, b: 2'



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 6
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 14
    var_12 = 'hello'
    var_13 = 16
    var_14 = 20
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_12)
    var_16 = 'key1: 123, key2: hello'
    var_17 = len(var_16)
    var_18 = 1
    var_19 = var_17 - var_18
    var_20 = len(var_16)
    var_21 = var_20 - var_18
    var_22 = [var_0]
    var_23 = [var_0]



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a: 1\nb: 2'
    var_15 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_16 = module_0.ScalarToken(var_7, var_1, var_1, var_7)
    var_17 = module_0.ScalarToken(var_3, var_1, var_1, var_5)
    var_18 = module_0.ScalarToken(var_4, var_1, var_1, var_11)
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'a: 1\nb: 2'
    var_21 = 'non_existent'
    var_22 = 'non_existent'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a 1 b 2'
    var_15 = module_0.DictToken()
    var_16 = 'a 1 b 2'
    var_17 = module_0.DictToken()
    var_18 = 'non_existent'
    var_19 = 'non_existent'



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 17
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = [var_0]
    var_23 = [var_0]



# Parsed testcases at query #29
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
    var_7 = 'name: Alice'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'age'
    var_10 = 13
    var_11 = 16
    var_12 = 'name: Alice, age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 18
    var_16 = 20
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_3: var_8, var_13: var_17}
    var_19 = module_0.DictToken()
    var_20 = {var_3: var_8, var_13: var_17}
    var_21 = module_0.DictToken()



# Parsed testcases at query #30
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
    var_22 = module_0.DictToken()
    var_23 = {var_0: var_4, var_8: var_12}



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 6
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 16
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = {var_3: var_8, var_12: var_16}
    var_18 = 'key1: 123\nkey2: hello'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()



# Parsed testcases at query #32
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
    var_11 = 'a'
    var_12 = module_0.ScalarToken(var_11, var_1, var_1, var_11)
    var_13 = 1
    var_14 = 2
    var_15 = '1'
    var_16 = module_0.ScalarToken(var_13, var_14, var_14, var_15)
    var_17 = 'b'
    var_18 = module_0.ScalarToken(var_17, var_2, var_2, var_17)
    var_19 = '2'
    var_20 = module_0.ScalarToken(var_14, var_5, var_5, var_19)
    var_21 = var_12._value
    var_22 = var_18._value
    var_23 = {var_21: var_16, var_22: var_20}
    var_24 = 'a: 1\nb: 2'
    var_25 = 7
    var_26 = var_12._value
    var_27 = var_18._value
    var_28 = {var_26: var_16, var_27: var_20}
    var_29 = [var_11]
    var_30 = [var_11]



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = 30
    var_12 = 17
    var_13 = 19
    var_14 = '30'
    var_15 = module_0.ScalarToken(var_11, var_12, var_13, var_14)
    var_16 = 'name: John, age: 30'
    var_17 = 18
    var_18 = []



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 6
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.ScalarToken(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 16
    var_15 = 20
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_13)
    var_17 = {var_3: var_8, var_12: var_16}
    var_18 = 'key1: 123\nkey2: hello'
    var_19 = module_0.DictToken()



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'John'
    var_5 = 6
    var_6 = 10
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 17
    var_14 = 19
    var_15 = '30'
    var_16 = module_0.ScalarToken(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: John, age: 30'
    var_19 = len(var_18)
    var_20 = 1
    var_21 = var_19 - var_20
    var_22 = module_0.DictToken()
    var_23 = len(var_18)
    var_24 = var_23 - var_20
    var_25 = module_0.DictToken()
    var_26 = [var_0]
    var_27 = [var_0]



