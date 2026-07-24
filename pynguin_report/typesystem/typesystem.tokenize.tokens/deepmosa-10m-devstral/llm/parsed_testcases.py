####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------




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
    var_11 = var_10._child_keys
    var_12 = bool(var_10._child_keys == {'a': 'a', 'b': 'b'})
    assert var_12 is True
    var_13 = var_10._child_tokens
    var_14 = bool(var_10._child_tokens == {'a': 1, 'b': 2})
    assert var_14 is True



# Parsed testcases at query #4
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
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.Token(var_5, var_1, var_6, var_3)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 4
    var_7 = module_0.Token(var_0, var_5, var_6, var_3)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True

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



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'key2'
    var_5 = 4
    var_6 = 7
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'value1'
    var_9 = 8
    var_10 = 13
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'value2'
    var_13 = 14
    var_14 = 19
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = (var_3, var_11)
    var_17 = (var_7, var_15)
    var_18 = {var_16, var_17}
    var_19 = 'key1value1key2value2'
    var_20 = [var_18, var_1, var_14, var_19]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'key1': var_3, 'key2': var_7})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'key1': var_11, 'key2': var_15})
    assert var_26 is True



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'token1'
    var_3 = 'token2'
    var_4 = 0
    var_5 = 10
    var_6 = 'content'
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'key1'
    var_10 = 'key2'



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'Hello World'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'Hello World'



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = [var_9, var_1, var_4, var_0]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = var_12._value
    var_14 = bool(var_12._value == var_9)
    assert var_14 is True
    var_15 = var_12._start_index
    var_16 = bool(var_12._start_index == var_1)
    assert var_16 is True
    var_17 = var_12._end_index
    var_18 = bool(var_12._end_index == var_4)
    assert var_18 is True
    var_19 = var_12._content
    var_20 = bool(var_12._content == var_0)
    assert var_20 is True
    var_21 = var_12._child_keys
    var_22 = bool(var_12._child_keys == {})
    assert var_22 is True
    var_23 = var_12._child_tokens
    var_24 = bool(var_12._child_tokens == {})
    assert var_24 is True



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 5
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_7, var_1, var_2, var_0]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_7)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_1)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_2)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_0)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens



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
    var_6 = module_0.Token(var_4, var_1, var_5, var_4)
    var_7 = 'key2'
    var_8 = module_0.Token(var_7, var_1, var_2, var_7)
    var_9 = 'value2'
    var_10 = module_0.Token(var_9, var_1, var_5, var_9)
    var_11 = {var_3: var_6, var_8: var_10}
    var_12 = ''
    var_13 = [var_11, var_1, var_1, var_12]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._child_keys
    var_17 = bool(var_15._child_keys == {var_3._value: var_3, var_8._value: var_8})
    assert var_17 is True
    var_18 = var_15._child_tokens
    var_19 = bool(var_15._child_tokens == {var_3._value: var_6, var_8._value: var_10})
    assert var_19 is True



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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
    var_21 = bool(var_10._child_tokens == var_4)
    assert var_21 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 4
    var_8 = var_4._content
    assert var_8 == 'content'



# Parsed testcases at query #25
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



# Parsed testcases at query #26
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
    var_5 = 'fail'
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'test content'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_7, var_0, var_1, var_2]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_7)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_0)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_1)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_2)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = 5
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_7, var_1, var_2, var_0]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._value
    var_12 = bool(var_10._value == var_7)
    assert var_12 is True
    var_13 = var_10._start_index
    var_14 = bool(var_10._start_index == var_1)
    assert var_14 is True
    var_15 = var_10._end_index
    var_16 = bool(var_10._end_index == var_2)
    assert var_16 is True
    var_17 = var_10._content
    var_18 = bool(var_10._content == var_0)
    assert var_18 is True
    var_19 = var_10._child_keys
    var_20 = var_10._child_tokens



# Parsed testcases at query #31
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
    var_17 = 'key1:value1,key2:value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = module_0.Token(var_0, var_1, var_2, var_0)
    var_22 = module_0.Token(var_4, var_5, var_6, var_4)
    var_23 = {var_0: var_21, var_4: var_22}
    var_24 = var_20._child_keys
    var_25 = bool(var_20._child_keys == var_23)
    assert var_25 is True
    var_26 = module_0.Token(var_8, var_9, var_10, var_8)
    var_27 = module_0.Token(var_12, var_13, var_14, var_12)
    var_28 = {var_0: var_26, var_4: var_27}
    var_29 = var_20._child_tokens
    var_30 = bool(var_20._child_tokens == var_28)
    assert var_30 is True



# Parsed testcases at query #32
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
    var_8 = {var_0: var_4}
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



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = 'content_string'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 4
    var_8 = var_4._content
    assert var_8 == 'content_string'



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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



# Parsed testcases at query #38
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



# Parsed testcases at query #39
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



# Parsed testcases at query #40
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



# Parsed testcases at query #41
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



# Parsed testcases at query #42
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 26/27 statements.


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
    var_15 = 21
    var_16 = module_0.Token(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1: value1, key2: value2'
    var_25 = [var_23, var_1, var_15, var_24]
    var_26 = {}
    var_27 = module_0.DictToken(*var_25, **var_26)
    var_28 = var_27._child_keys
    var_29 = bool(var_27._child_keys == {'key1': var_8[0], 'key2': var_8[1]})
    assert var_29 is True
    var_30 = var_27._child_tokens
    var_31 = bool(var_27._child_tokens == {'key1': var_17[0], 'key2': var_17[1]})
    assert var_31 is True



# Parsed testcases at query #44
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'content'
    var_7 = '_child_keys'
    var_8 = '_child_tokens'



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
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



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 2
    var_3 = 'content1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 7/34 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'some content'



# Parsed testcases at query #54
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    var_6 = bool(var_4._start_index == var_1)
    assert var_6 is True



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 7/34 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'content'



# Parsed testcases at query #57
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



# Parsed testcases at query #58
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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 8/23 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 'test'
    var_5 = 0
    var_6 = 5
    var_7 = 'content'



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_dicttoken_initialization. Retrieved 21/23 statements.


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
    var_17 = 'key1:value1,key2:value2'
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_16, var_20: var_1, var_21: var_14, var_22: var_17}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = var_24._child_tokens



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 20/28 statements.


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
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys



# Parsed testcases at query #68
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 5
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #69
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



# Parsed testcases at query #70
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



# Parsed testcases at query #71
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 10



# Parsed testcases at query #72
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_15 = 21
    var_16 = module_0.Token(var_13, var_14, var_15, var_13)
    var_17 = [var_12, var_16]
    var_18 = var_8[var_1]
    var_19 = 1
    var_20 = var_8[var_19]
    var_21 = var_17[var_1]
    var_22 = var_17[var_19]
    var_23 = {var_18: var_21, var_20: var_22}
    var_24 = 'key1:value1,key2:value2'
    var_25 = [var_23, var_1, var_15, var_24]
    var_26 = {}
    var_27 = module_0.DictToken(*var_25, **var_26)
    var_28 = var_27._child_keys
    var_29 = bool(var_27._child_keys == {'key1': var_8[0], 'key2': var_8[1]})
    assert var_29 is True
    var_30 = var_27._child_tokens
    var_31 = bool(var_27._child_tokens == {'key1': var_17[0], 'key2': var_17[1]})
    assert var_31 is True



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 != 'not a token')
    assert var_5 is True



# Parsed testcases at query #4
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
    var_19 = 'key1:value1,key2:value2'
    var_20 = [var_18, var_1, var_14, var_19]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'key1': var_3, 'key2': var_11})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'key1': var_7, 'key2': var_15})
    assert var_26 is True



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 5
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = [var_9, var_1, var_7, var_3]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = var_12._child_keys
    var_14 = bool(var_12._child_keys == {'key': var_4})
    assert var_14 is True
    var_15 = var_12._child_tokens
    var_16 = bool(var_12._child_tokens == {'key': var_8})
    assert var_16 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_2, var_2, var_3)
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 2
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'b'
    var_5 = 1
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'value'
    var_10 = 'start_index'
    var_11 = 'end_index'
    var_12 = 'content'
    var_13 = {var_9: var_7, var_10: var_1, var_11: var_5, var_12: var_2}
    var_14 = module_0.DictToken(*var_8, **var_13)
    var_15 = '_child_keys'
    var_16 = hasattr(var_14, var_15)
    var_17 = bool(var_16)
    assert var_17 is True
    var_18 = var_14._child_keys



# Parsed testcases at query #10
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
    var_7 = 'mock content'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = var_10._child_keys
    var_12 = var_10._child_tokens



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is False



# Parsed testcases at query #12
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
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = '_child_keys'
    var_22 = hasattr(var_20, var_21)
    var_23 = bool(var_22)
    assert var_23 is True
    var_24 = '_child_tokens'
    var_25 = hasattr(var_20, var_24)
    var_26 = bool(var_25)
    assert var_26 is True
    var_27 = module_0.Token(var_0, var_1, var_2, var_0)
    var_28 = module_0.Token(var_4, var_5, var_6, var_4)
    var_29 = {var_0: var_27, var_4: var_28}
    var_30 = var_20._child_keys
    var_31 = bool(var_20._child_keys == var_29)
    assert var_31 is True
    var_32 = module_0.Token(var_8, var_9, var_10, var_8)
    var_33 = module_0.Token(var_12, var_13, var_14, var_12)
    var_34 = {var_0: var_32, var_4: var_33}
    var_35 = var_20._child_tokens
    var_36 = bool(var_20._child_tokens == var_34)
    assert var_36 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key2: value2'
    var_13 = module_0.Token(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.Token(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = 'key1: value1\nkey2: value2'
    var_20 = []
    var_21 = 'value'
    var_22 = 'start_index'
    var_23 = 'end_index'
    var_24 = 'content'
    var_25 = {var_21: var_18, var_22: var_1, var_23: var_16, var_24: var_19}
    var_26 = module_0.DictToken(*var_20, **var_25)
    var_27 = var_26._child_keys
    var_28 = bool(var_26._child_keys == {'key1': var_4, 'key2': var_13})
    assert var_28 is True
    var_29 = var_26._child_tokens
    var_30 = bool(var_26._child_tokens == {'key1': var_8, 'key2': var_17})
    assert var_30 is True



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_22 = bool(var_20._child_keys == {var_3._value: var_3, var_7._value: var_7})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {var_3._value: var_11, var_7._value: var_15})
    assert var_24 is True



# Parsed testcases at query #19
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 9/11 statements.


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
    var_21 = bool(var_10._child_tokens == var_4)
    assert var_21 is True



# Parsed testcases at query #25
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
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
    var_22 = bool(var_20._child_keys == {var_3._value: var_3, var_11._value: var_11})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {var_3._value: var_7, var_11._value: var_15})
    assert var_24 is True



# Parsed testcases at query #28
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



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dict_token_initialization_with_valid_args. Retrieved 34/40 statements.


def test_case_0():
    var_0 = 'MockToken'
    var_1 = ()
    var_2 = '_value'
    var_3 = 'key1'
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = ()
    var_8 = 'value1'
    var_9 = {var_2: var_8}
    var_10 = type(var_0, var_7, var_9)
    var_11 = var_10()
    var_12 = ()
    var_13 = 'key2'
    var_14 = {var_2: var_13}
    var_15 = type(var_0, var_12, var_14)
    var_16 = var_15()
    var_17 = ()
    var_18 = 'value2'
    var_19 = {var_2: var_18}
    var_20 = type(var_0, var_17, var_19)
    var_21 = var_20()
    var_22 = {var_6: var_11, var_16: var_21}
    var_23 = ()
    var_24 = '_start_index'
    var_25 = '_end_index'
    var_26 = '_content'
    var_27 = 0
    var_28 = 10
    var_29 = 'mock content'
    var_30 = {var_2: var_22, var_24: var_27, var_25: var_28, var_26: var_29}
    var_31 = type(var_0, var_23, var_30)
    var_32 = var_31()
    var_33 = 'mock_value'



# Parsed testcases at query #37
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



# Parsed testcases at query #38
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



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = -1
    var_2 = 10
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1



# Parsed testcases at query #40
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



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = 'some content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == -1



# Parsed testcases at query #43
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



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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
    var_0 = 'test1'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'test2'
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



# Parsed testcases at query #47
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
    var_14 = 21
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_11, var_7: var_15}
    var_17 = 'key1value1key2value2'
    var_18 = [var_16, var_1, var_14, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {var_3._value: var_3, var_7._value: var_7})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {var_3._value: var_11, var_7._value: var_15})
    assert var_24 is True



# Parsed testcases at query #48
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 10



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_dicttoken_init_creates_child_keys_and_tokens. Retrieved 9/34 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = ''
    var_6 = '_child_keys'
    var_7 = '_child_tokens'
    var_8 = 1



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_dict_token_init_sets_child_keys_and_tokens. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'content'



# Parsed testcases at query #54
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 2
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 2
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 2
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 3
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 2
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = 10
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 5



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



# Parsed testcases at query #60
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = -1
    var_2 = 10
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    var_6 = bool(var_4._start_index < 0)
    assert var_6 is True



# Parsed testcases at query #61
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 'b'
    var_4 = 1
    var_5 = module_0.Token(var_3, var_4, var_4, var_3)
    var_6 = [var_2, var_5]
    var_7 = 2
    var_8 = '1'
    var_9 = module_0.Token(var_4, var_7, var_7, var_8)
    var_10 = 3
    var_11 = '2'
    var_12 = module_0.Token(var_7, var_10, var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = zip(var_6, var_13)
    var_15 = dict(var_14)
    var_16 = 4
    var_17 = 'a:1,b:2'
    var_18 = [var_15, var_1, var_16, var_17]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._child_keys
    var_22 = bool(var_20._child_keys == {'a': var_6[0], 'b': var_6[1]})
    assert var_22 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = 0
    var_5 = ''



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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
    var_22 = bool(var_20._child_keys == {var_3._value: var_3, var_7._value: var_7})
    assert var_22 is True
    var_23 = var_20._child_tokens
    var_24 = bool(var_20._child_tokens == {var_3._value: var_11, var_7._value: var_15})
    assert var_24 is True



# Parsed testcases at query #69
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #70
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



# Parsed testcases at query #71
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 10



# Parsed testcases at query #72
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



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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



