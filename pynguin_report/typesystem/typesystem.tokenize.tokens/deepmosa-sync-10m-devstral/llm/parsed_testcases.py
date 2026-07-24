####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'
    var_10 = [var_8, var_1, var_6, var_9]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = var_12._child_keys
    var_14 = bool(var_12._child_keys == {'key': var_3})
    assert var_14 is True
    var_15 = var_12._child_tokens
    var_16 = bool(var_12._child_tokens == {'key': var_7})
    assert var_16 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 10
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_7})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_11, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'content'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == 'content'



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = 'content2'
    var_7 = module_0.Token(var_5, var_1, var_2, var_6)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dicttoken_constructor_initializes_child_keys_and_tokens. Retrieved 11/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = '_child_keys'
    var_8 = hasattr(var_6, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = '_child_tokens'
    var_11 = hasattr(var_6, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = var_6._child_keys
    var_14 = var_6._child_tokens



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 21
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = bool(not var_9._value)
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'some content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
    var_5 = var_3._start_index
    assert var_5 == 5
    var_6 = var_3._end_index
    assert var_6 == 10
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'test content'



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    assert var_7 == 'test'
    var_8 = var_6._start_index
    assert var_8 == 0
    var_9 = var_6._end_index
    assert var_9 == 3
    var_10 = var_6._content
    assert var_10 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key2'
    var_6 = 5
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_12, var_8: var_16}
    var_18 = [var_17, var_1, var_15, var_3]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_4, 'key2': var_8})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_12, 'key2': var_16})
    assert var_24 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 2
    var_8 = var_4._content
    assert var_8 == 'abc'



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1:value1,key2:value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 2
    var_4 = module_0.Token(var_3, var_1, var_1)
    var_5 = {var_2: var_4}
    var_6 = ''
    var_7 = [var_5, var_1, var_1, var_6]
    var_8 = {}
    var_9 = module_0.DictToken(*var_7, **var_8)
    var_10 = var_9._child_keys
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 1
    var_13 = bool(1 in var_9._child_keys)
    assert var_13 is True
    var_14 = var_9._child_keys[1]._value
    assert var_14 == 1



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 10
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_7})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_11, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = -1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    var_6 = bool(var_4._start_index > var_4._end_index)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content_string'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'content_string'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 5
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 4
    var_6 = '1, 2, 3'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    var_11 = bool(var_7._start_index == var_4)
    assert var_11 is True
    var_12 = var_7._end_index
    var_13 = bool(var_7._end_index == var_5)
    assert var_13 is True
    var_14 = var_7._content
    var_15 = bool(var_7._content == var_6)
    assert var_15 is True



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 0



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'test content'



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'Hello, World!'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'Hello, World!'



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 'b'
    var_4 = 2
    var_5 = module_0.Token(var_3, var_4, var_4, var_3)
    var_6 = [var_2, var_5]
    var_7 = 1
    var_8 = 4
    var_9 = '1'
    var_10 = module_0.Token(var_7, var_8, var_8, var_9)
    var_11 = 6
    var_12 = '2'
    var_13 = module_0.Token(var_4, var_11, var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = var_6[var_1]
    var_16 = var_6[var_7]
    var_17 = var_14[var_1]
    var_18 = var_14[var_7]
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'a: 1, b: 2'
    var_21 = [var_19, var_1, var_11, var_20]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'a': var_6[0], 'b': var_6[1]})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'a': var_14[0], 'b': var_14[1]})
    assert var_27 is True



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = bool(not var_9._child_keys)
    assert var_10 is True
    var_11 = bool(not var_9._child_tokens)
    assert var_11 is True



# Parsed testcases at query #38
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'test content'



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #40
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 'b'
    var_4 = 1
    var_5 = module_0.Token(var_3, var_4, var_4)
    var_6 = {var_2: var_5}
    var_7 = 'ab'
    var_8 = []
    var_9 = 'value'
    var_10 = 'start_index'
    var_11 = 'end_index'
    var_12 = 'content'
    var_13 = {var_9: var_6, var_10: var_1, var_11: var_4, var_12: var_7}
    var_14 = module_0.DictToken(*var_8, **var_13)
    var_15 = module_0.Token(var_0, var_1, var_1)
    var_16 = {var_0: var_15}
    var_17 = var_14._child_keys
    var_18 = bool(var_14._child_keys == var_16)
    assert var_18 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == '[]'



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'sample content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_4)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_5)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_6)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_7)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = bool(var_10._child_keys == {})
    assert var_20 is True
    var_21 = var_10._child_tokens
    var_22 = bool(var_10._child_tokens == {})
    assert var_22 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'diff'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different content'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = 'content2'
    var_7 = module_0.Token(var_5, var_1, var_2, var_6)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 5



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'test content'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'sample content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_4)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_5)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_6)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_7)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens
    var_21 = bool(var_10._child_tokens == var_4)
    assert var_21 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test'



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'some content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'some content'



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different_value'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 5



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 23/28 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = module_0.Token(var_0, var_5, var_5)
    var_7 = module_0.Token(var_1, var_2, var_2)
    var_8 = [var_6, var_7]
    var_9 = module_0.Token(var_0, var_5, var_5)
    var_10 = module_0.Token(var_1, var_2, var_2)
    var_11 = module_0.Token(var_2, var_3, var_3)
    var_12 = 3
    var_13 = module_0.Token(var_3, var_12, var_12)
    var_14 = {var_9: var_11, var_10: var_13}
    var_15 = 'ab12'
    var_16 = [var_4, var_5, var_12, var_15]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_8[var_5]
    var_20 = var_8[var_2]
    var_21 = var_8[var_5]
    var_22 = var_14[var_21]
    var_23 = var_8[var_2]
    var_24 = var_14[var_23]
    var_25 = var_18._child_keys
    var_26 = var_18._child_tokens



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = (var_3, var_7)
    var_17 = (var_11, var_15)
    var_18 = {var_16, var_17}
    var_19 = 'key1value1key2value2'
    var_20 = [var_18, var_1, var_14, var_19]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_26 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = [var_3, var_7]
    var_9 = 'value1'
    var_10 = 10
    var_11 = 15
    var_12 = module_0.Token(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = module_0.Token(var_13, var_14, var_15, var_13)
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
    var_27 = [var_23, var_1, var_26, var_24]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = var_29._child_keys
    var_31 = bool(var_29._child_keys == {'key1': var_8[0], 'key2': var_8[1]})
    assert var_31 is True
    var_32 = var_29._child_tokens
    var_33 = bool(var_29._child_tokens == {'key1': var_17[0], 'key2': var_17[1]})
    assert var_33 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 'b'
    var_4 = 1
    var_5 = module_0.Token(var_3, var_4, var_4, var_3)
    var_6 = {var_2: var_5}
    var_7 = 'ab'
    var_8 = []
    var_9 = 'value'
    var_10 = 'start_index'
    var_11 = 'end_index'
    var_12 = 'content'
    var_13 = {var_9: var_6, var_10: var_1, var_11: var_4, var_12: var_7}
    var_14 = module_0.DictToken(*var_8, **var_13)
    var_15 = module_0.Token(var_0, var_1, var_1, var_0)
    var_16 = {var_0: var_15}
    var_17 = var_14._child_keys
    var_18 = bool(var_14._child_keys == var_16)
    assert var_18 is True
    var_19 = module_0.Token(var_3, var_4, var_4, var_3)
    var_20 = {var_0: var_19}
    var_21 = var_14._child_tokens
    var_22 = bool(var_14._child_tokens == var_20)
    assert var_22 is True



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == var_0)
    assert var_6 is True
    var_7 = var_4._start_index
    var_8 = bool(var_4._start_index == var_1)
    assert var_8 is True
    var_9 = var_4._end_index
    var_10 = bool(var_4._end_index == var_2)
    assert var_10 is True
    var_11 = var_4._content
    var_12 = bool(var_4._content == var_3)
    assert var_12 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 2
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_child_keys_and_tokens. Retrieved 15/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 5
    var_7 = 'a: 1, b: 2'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = '_child_keys'
    var_12 = hasattr(var_10, var_11)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = '_child_tokens'
    var_15 = hasattr(var_10, var_14)
    var_16 = bool(var_15)
    assert var_16 is True
    var_17 = var_10._child_keys
    var_18 = var_10._child_tokens



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = 'some_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'content'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == 'content'



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_2, var_1, var_2, var_3)
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_16, var_20: var_1, var_21: var_14, var_22: var_17}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = bool(var_24._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_26 is True
    var_27 = var_24._child_tokens
    var_28 = bool(var_24._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_28 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_child_keys_and_tokens. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'some content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._child_keys
    var_12 = var_10._child_tokens



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'content'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == 'content'



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 7/28 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'some content'



# Parsed testcases at query #38
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = bool(not var_9._value)
    assert var_10 is True



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'some content'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.DictToken(*var_6, **var_7)
    var_9 = var_8._value
    var_10 = bool(var_8._value == var_2)
    assert var_10 is True
    var_11 = var_8._start_index
    var_12 = bool(var_8._start_index == var_3)
    assert var_12 is True
    var_13 = var_8._end_index
    var_14 = bool(var_8._end_index == var_4)
    assert var_14 is True
    var_15 = var_8._content
    var_16 = bool(var_8._content == var_5)
    assert var_16 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 0
    var_10 = 10
    var_11 = 'some content'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._child_keys
    var_16 = bool(var_14._child_keys == {'key': var_3})
    assert var_16 is True
    var_17 = var_14._child_tokens
    var_18 = bool(var_14._child_tokens == {'key': var_7})
    assert var_18 is True



# Parsed testcases at query #40
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'test content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._child_keys
    var_12 = bool(var_10._child_keys == {'a': 'a', 'b': 'b'})
    assert var_12 is True
    var_13 = var_10._child_tokens
    var_14 = bool(var_10._child_tokens == {'a': 1, 'b': 2})
    assert var_14 is True



# Parsed testcases at query #41
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True



# Parsed testcases at query #42
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'content'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == 'content'



# Parsed testcases at query #43
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_24 is True



# Parsed testcases at query #44
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'example'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'example'



# Parsed testcases at query #45
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #46
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = -1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._end_index
    assert var_5 == -1



# Parsed testcases at query #47
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1



# Parsed testcases at query #48
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 0
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    var_6 = bool(var_4._start_index > var_4._end_index)
    assert var_6 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_parent_and_child_attributes. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'some content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_4)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_5)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_6)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_7)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 5
    var_7 = 'test content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_4)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_5)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_6)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_7)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 13/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 0
    var_10 = 10
    var_11 = 'key1: value1'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._child_keys
    var_16 = var_14._child_tokens



# Parsed testcases at query #52
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 4
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #53
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 0
    var_4 = '1'
    var_5 = module_0.Token(var_2, var_3, var_3, var_4)
    var_6 = 2
    var_7 = '2'
    var_8 = module_0.Token(var_6, var_2, var_2, var_7)
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = 0
    var_11 = 1
    var_12 = 'ab'
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._child_keys
    var_17 = bool(var_15._child_keys == {1: var_9['a'], 2: var_9['b']})
    assert var_17 is True



# Parsed testcases at query #54
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_2}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = bool(not var_9._child_keys)
    assert var_10 is True
    var_11 = bool(not var_9._child_tokens)
    assert var_11 is True



# Parsed testcases at query #55
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1
    var_6 = var_4._start_index
    var_7 = bool(var_4._start_index < 0)
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 12
    var_6 = 15
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 5
    var_10 = 10
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1: value1, key2: value2'
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = [var_16, var_1, var_20, var_17]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = module_0.Token(var_0, var_1, var_2, var_0)
    var_25 = module_0.Token(var_4, var_5, var_6, var_4)
    var_26 = {var_0: var_24, var_4: var_25}
    var_27 = var_23._child_keys
    var_28 = bool(var_23._child_keys == var_26)
    assert var_28 is True
    var_29 = module_0.Token(var_8, var_9, var_10, var_8)
    var_30 = module_0.Token(var_12, var_13, var_14, var_12)
    var_31 = {var_0: var_29, var_4: var_30}
    var_32 = var_23._child_tokens
    var_33 = bool(var_23._child_tokens == var_31)
    assert var_33 is True



# Parsed testcases at query #57
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 0



# Parsed testcases at query #58
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #59
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 3
    var_8 = var_4._content
    assert var_8 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #60
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 12
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = (var_3, var_7)
    var_17 = (var_11, var_15)
    var_18 = {var_16, var_17}
    var_19 = 'key1value1key2value2'
    var_20 = [var_18, var_1, var_14, var_19]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_26 is True



