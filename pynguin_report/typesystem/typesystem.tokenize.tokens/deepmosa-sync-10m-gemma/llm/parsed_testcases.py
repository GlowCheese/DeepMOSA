####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v1'
    var_5 = 2
    var_6 = 3
    var_7 = 'kv1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 5
    var_10 = 'k2'
    var_11 = module_0.Token(var_5, var_9, var_9, var_10)
    var_12 = 'v2'
    var_13 = 7
    var_14 = 8
    var_15 = 'kv2'
    var_16 = module_0.Token(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_8, var_11: var_16}
    var_18 = 'k1v1k2v2'
    var_19 = [var_17, var_1, var_14, var_18]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._value
    var_23 = bool(var_21._value == var_17)
    assert var_23 is True
    var_24 = var_21._start_index
    assert var_24 == 0
    var_25 = var_21._end_index
    assert var_25 == 8
    var_26 = var_21._content
    assert var_26 == 'k1v1k2v2'
    var_27 = var_21._child_keys
    var_28 = bool(var_21._child_keys == {1: var_3, 2: var_11})
    assert var_28 is True
    var_29 = var_21._child_tokens
    var_30 = bool(var_21._child_tokens == {var_3: var_8, var_11: var_16})
    assert var_30 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dict_token_init_logic. Retrieved 10/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = '0'
    var_5 = module_0.Token(var_1, var_1, var_1, var_4)
    var_6 = module_0.Token(var_0, var_0, var_0, var_2)
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = '01'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_6, var_12: var_18}
    var_25 = '{"key1": "val1", "key2": "val2"}'
    var_26 = {var_5: var_11, var_17: var_23}
    var_27 = len(var_25)
    var_28 = 1
    var_29 = var_27 - var_28
    var_30 = []
    var_31 = 'value'
    var_32 = 'start_index'
    var_33 = 'end_index'
    var_34 = 'content'
    var_35 = {var_31: var_26, var_32: var_1, var_33: var_29, var_34: var_25}
    var_36 = module_0.DictToken(*var_30, **var_35)
    var_37 = var_36._value
    var_38 = bool(var_36._value == var_24)
    assert var_38 is True
    var_39 = var_36._start_index
    assert var_39 == 0
    var_40 = len(var_25)
    var_41 = var_40 - var_28
    var_42 = var_36._end_index
    var_43 = bool(var_36._end_index == var_41)
    assert var_43 is True
    var_44 = var_36._content
    var_45 = bool(var_36._content == var_25)
    assert var_45 is True
    var_46 = var_36._child_keys['key1']
    var_47 = bool(var_36._child_keys['key1'] == var_5)
    assert var_47 is True
    var_48 = var_36._child_tokens['key1']
    var_49 = bool(var_36._child_tokens['key1'] == var_11)
    assert var_49 is True
    var_50 = var_36._child_keys['key2']
    var_51 = bool(var_36._child_keys['key2'] == var_17)
    assert var_51 is True
    var_52 = var_36._child_tokens['key2']
    var_53 = bool(var_36._child_tokens['key2'] == var_23)
    assert var_53 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_token_eq_success. Retrieved 4/13 statements.
# Partially parsed test_token_eq_failure_different_value. Retrieved 5/16 statements.
# Partially parsed test_token_eq_failure_different_start_index. Retrieved 5/14 statements.
# Partially parsed test_token_eq_failure_different_end_index. Retrieved 5/14 statements.
# Partially parsed test_token_eq_failure_different_type. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = 20

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = 1

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = 4

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = 'not a token'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 15/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = 'a: 1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = 'a: 1, b: 2'
    var_10 = module_0.Token(var_7, var_8, var_8, var_9)
    var_11 = 8
    var_12 = module_0.Token(var_4, var_11, var_11, var_9)
    var_13 = {var_2: var_6, var_10: var_12}
    var_14 = []
    var_15 = 'value'
    var_16 = 'start_index'
    var_17 = 'end_index'
    var_18 = 'content'
    var_19 = {var_15: var_13, var_16: var_1, var_17: var_11, var_18: var_9}
    var_20 = module_0.DictToken(*var_14, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_13)
    assert var_22 is True
    var_23 = var_20._start_index
    assert var_23 == 0
    var_24 = var_20._end_index
    assert var_24 == 8
    var_25 = var_20._content
    assert var_25 == 'a: 1, b: 2'
    var_26 = var_20._child_keys
    var_27 = bool(var_20._child_keys == {'a': var_2, 'b': var_10})
    assert var_27 is True
    var_28 = var_20._child_tokens
    var_29 = bool(var_20._child_tokens == {'a': var_6, 'b': var_12})
    assert var_29 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 0
    var_3 = ''
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dict_token_init_initializes_attributes. Retrieved 15/28 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'k'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'key1'
    var_5 = 'v'
    var_6 = module_0.Token(var_0, var_1, var_1, var_5)
    var_7 = 'val1'
    var_8 = module_0.Token(var_0, var_1, var_1, var_2)
    var_9 = 'key2'
    var_10 = module_0.Token(var_0, var_1, var_1, var_5)
    var_11 = 'val2'
    var_12 = {var_3: var_6, var_8: var_10}
    var_13 = 5
    var_14 = 'content'



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_6, var_12: var_18}
    var_25 = {var_5: var_11, var_17: var_23}
    var_26 = 'key1: val1, key2: val2'
    var_27 = []
    var_28 = 'value'
    var_29 = 'start_index'
    var_30 = 'end_index'
    var_31 = 'content'
    var_32 = {var_28: var_25, var_29: var_1, var_30: var_20, var_31: var_26}
    var_33 = module_0.DictToken(*var_27, **var_32)
    var_34 = var_33._value
    var_35 = bool(var_33._value == var_24)
    assert var_35 is True
    var_36 = var_33._start_index
    assert var_36 == 0
    var_37 = var_33._end_index
    assert var_37 == 22
    var_38 = var_33._content
    assert var_38 == 'key1: val1, key2: val2'
    var_39 = var_33._child_keys['key1']
    var_40 = bool(var_33._child_keys['key1'] == var_5)
    assert var_40 is True
    var_41 = var_33._child_keys['key2']
    var_42 = bool(var_33._child_keys['key2'] == var_17)
    assert var_42 is True
    var_43 = var_33._child_tokens['key1']
    var_44 = bool(var_33._child_tokens['key1'] == var_11)
    assert var_44 is True
    var_45 = var_33._child_tokens['key2']
    var_46 = bool(var_33._child_tokens['key2'] == var_23)
    assert var_46 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'v'
    var_7 = 3
    var_8 = 4
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = {var_5: var_11}
    var_13 = 'k: v'
    var_14 = []
    var_15 = 'value'
    var_16 = 'start_index'
    var_17 = 'end_index'
    var_18 = 'content'
    var_19 = {var_15: var_12, var_16: var_1, var_17: var_8, var_18: var_13}
    var_20 = module_0.DictToken(*var_14, **var_19)
    var_21 = var_20.string
    assert var_21 == 'k: v'



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 2
    var_3 = 'xabcy'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'ab'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = repr(var_3)
    assert var_4 == "Token('val')"



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample content'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample content'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)
    var_5 = var_4.string
    assert var_5 == 'hello'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = "'a'"
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = repr(var_3)
    assert var_4 == 'Token("\'a\'")'



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_11, var_12: var_23}
    var_25 = {var_5: var_11, var_17: var_23}
    var_26 = 'key1:val1,key2:val2'
    var_27 = [var_25, var_1, var_20, var_26]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = var_29._value
    var_31 = bool(var_29._value == {'key1': 'val1', 'key2': 'val2'})
    assert var_31 is True
    var_32 = var_29._start_index
    assert var_32 == 0
    var_33 = var_29._end_index
    assert var_33 == 22
    var_34 = var_29._content
    assert var_34 == 'key1:val1,key2:val2'
    var_35 = var_29._child_keys['key1']
    var_36 = bool(var_29._child_keys['key1'] == var_5)
    assert var_36 is True
    var_37 = var_29._child_tokens['key1']
    var_38 = bool(var_29._child_tokens['key1'] == var_11)
    assert var_38 is True
    var_39 = var_29._child_keys['key2']
    var_40 = bool(var_29._child_keys['key2'] == var_17)
    assert var_40 is True
    var_41 = var_29._child_tokens['key2']
    var_42 = bool(var_29._child_tokens['key2'] == var_23)
    assert var_42 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 20/37 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_5: var_11, var_17: var_23}
    var_25 = 30
    var_26 = 'key1: val1, key2: val2'
    var_27 = [var_24, var_1, var_25, var_26]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = var_29._value
    var_31 = bool(var_29._value == var_24)
    assert var_31 is True
    var_32 = var_29._start_index
    assert var_32 == 0
    var_33 = var_29._end_index
    assert var_33 == 30
    var_34 = var_29._content
    assert var_34 == 'key1: val1, key2: val2'
    var_35 = var_29._child_keys['key1']
    var_36 = bool(var_29._child_keys['key1'] == var_5)
    assert var_36 is True
    var_37 = var_29._child_tokens['key1']
    var_38 = bool(var_29._child_tokens['key1'] == var_11)
    assert var_38 is True
    var_39 = var_29._child_keys['key2']
    var_40 = bool(var_29._child_keys['key2'] == var_17)
    assert var_40 is True
    var_41 = var_29._child_tokens['key2']
    var_42 = bool(var_29._child_tokens['key2'] == var_23)
    assert var_42 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'v'
    var_7 = [var_6, var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_0.DictToken(*var_7, **var_8)
    var_10 = {var_5: var_9}
    var_11 = 5
    var_12 = 'k: v'
    var_13 = [var_10, var_1, var_11, var_12]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15.string
    assert var_16 == 'k: v'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'v'
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = [var_0, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = [var_6, var_7, var_8, var_6]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = {var_5: var_11}
    var_19 = 5
    var_20 = 'k: v'
    var_21 = [var_18, var_1, var_19, var_20]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_14: var_17}
    var_25 = [var_24, var_1, var_19, var_20]
    var_26 = {}
    var_27 = module_0.DictToken(*var_25, **var_26)
    var_28 = bool(var_23 == var_27)
    assert var_28 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 3
    var_3 = '012345'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == '123'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 17/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = 'key1: val1, key2: val2'



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 31/88 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = 'k1'
    var_25 = 'v1'
    var_26 = 'k2'
    var_27 = 'v2'
    var_28 = 2
    var_29 = [var_24, var_1, var_28, var_24]
    var_30 = {}
    var_31 = module_0.DictToken(*var_29, **var_30)
    var_32 = 3
    var_33 = 5
    var_34 = [var_25, var_32, var_33, var_25]
    var_35 = {}
    var_36 = module_0.DictToken(*var_34, **var_35)
    var_37 = 8
    var_38 = [var_26, var_7, var_37, var_26]
    var_39 = {}
    var_40 = module_0.DictToken(*var_38, **var_39)
    var_41 = 9
    var_42 = 11
    var_43 = [var_27, var_41, var_42, var_27]
    var_44 = {}
    var_45 = module_0.DictToken(*var_43, **var_44)
    var_46 = 'dummy'



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = '10'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 2
    var_6 = ' [10]'
    var_7 = module_0.ListToken(var_4, var_1, var_5, var_6)
    var_8 = module_0.Token(var_0, var_1, var_1, var_2)
    var_9 = [var_8]
    var_10 = var_7._value
    var_11 = bool(var_7._value == var_9)
    assert var_11 is True
    var_12 = var_7._start_index
    assert var_12 == 0
    var_13 = var_7._end_index
    assert var_13 == 2
    var_14 = var_7._content
    assert var_14 == ' [10]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'val'
    var_2 = module_0.Token(var_0, var_0, var_0, var_1)
    var_3 = [var_2]
    var_4 = 0
    var_5 = 4
    var_6 = ' [val]'
    var_7 = module_0.ListToken(var_3, var_4, var_5, var_6)
    var_8 = var_7.string
    assert var_8 == '[val]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = module_0.ListToken(var_4, var_1, var_1, var_2)
    var_6 = module_0.Token(var_0, var_1, var_1, var_2)
    var_7 = [var_6]
    var_8 = module_0.ListToken(var_7, var_1, var_1, var_2)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dict_token_init_logic. Retrieved 5/44 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'test'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 24/34 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = 'k1'
    var_25 = 'v1'
    var_26 = 2
    var_27 = [var_25, var_1, var_26, var_25]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = 'k2'
    var_31 = 'v2'
    var_32 = [var_31, var_2, var_7, var_31]
    var_33 = {}
    var_34 = module_0.DictToken(*var_32, **var_33)
    var_35 = 'k1: v1, k2: v2'



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: 1, b: 2'
    var_16 = [var_13, var_1, var_14, var_15]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_13)
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 7
    var_23 = var_18._content
    assert var_23 == 'a: 1, b: 2'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_27 is True
    var_28 = var_18.string
    assert var_28 == 'a: 1, b: 2'



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 18/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_5: var_11, var_17: var_23}
    var_25 = 'key1: val1, key2: val2'



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 10
    var_3 = '01234567890'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 5



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = '0123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == '1234'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dict_token_init_logic. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 'abcde'



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a: 1, b: 2'
    var_15 = 9
    var_16 = [var_13, var_1, var_15, var_14]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == {'a': 1, 'b': 2})
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 9
    var_23 = var_18._content
    assert var_23 == 'a: 1, b: 2'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 'v'
    var_4 = 2
    var_5 = module_0.Token(var_3, var_4, var_4, var_3)
    var_6 = {var_2: var_5}
    var_7 = 'k: v'
    var_8 = [var_6, var_1, var_4, var_7]
    var_9 = {}
    var_10 = module_0.DictToken(*var_8, **var_9)
    var_11 = module_0.Token(var_0, var_1, var_1, var_0)
    var_12 = module_0.Token(var_3, var_4, var_4, var_3)
    var_13 = {var_11: var_12}
    var_14 = [var_13, var_1, var_4, var_7]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = bool(var_10 == var_16)
    assert var_17 is True



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: 1, b: 2'
    var_16 = []
    var_17 = 'value'
    var_18 = 'start_index'
    var_19 = 'end_index'
    var_20 = 'content'
    var_21 = {var_17: var_13, var_18: var_1, var_19: var_14, var_20: var_15}
    var_22 = module_0.DictToken(*var_16, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == {'a': 1, 'b': 2})
    assert var_24 is True
    var_25 = var_22._start_index
    assert var_25 == 0
    var_26 = var_22._end_index
    assert var_26 == 7
    var_27 = var_22._content
    assert var_27 == 'a: 1, b: 2'
    var_28 = var_22._child_keys['a']
    var_29 = bool(var_22._child_keys['a'] == var_2)
    assert var_29 is True
    var_30 = var_22._child_keys['b']
    var_31 = bool(var_22._child_keys['b'] == var_9)
    assert var_31 is True
    var_32 = var_22._child_tokens['a']
    var_33 = bool(var_22._child_tokens['a'] == var_6)
    assert var_33 is True
    var_34 = var_22._child_tokens['b']
    var_35 = bool(var_22._child_tokens['b'] == var_12)
    assert var_35 is True



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 2
    var_8 = var_4._content
    assert var_8 == '123'



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = []
    var_4 = 'value'
    var_5 = 'start_index'
    var_6 = 'end_index'
    var_7 = 'content'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_0}
    var_9 = module_0.DictToken(*var_3, **var_8)
    var_10 = 'val1'
    var_11 = 6
    var_12 = 9
    var_13 = []
    var_14 = 'value'
    var_15 = 'start_index'
    var_16 = 'end_index'
    var_17 = 'content'
    var_18 = {var_14: var_10, var_15: var_11, var_16: var_12, var_17: var_10}
    var_19 = module_0.DictToken(*var_13, **var_18)
    var_20 = 'key2'
    var_21 = 11
    var_22 = 14
    var_23 = []
    var_24 = 'value'
    var_25 = 'start_index'
    var_26 = 'end_index'
    var_27 = 'content'
    var_28 = {var_24: var_20, var_25: var_21, var_26: var_22, var_27: var_20}
    var_29 = module_0.DictToken(*var_23, **var_28)
    var_30 = 'val2'
    var_31 = 16
    var_32 = 19
    var_33 = []
    var_34 = 'value'
    var_35 = 'start_index'
    var_36 = 'end_index'
    var_37 = 'content'
    var_38 = {var_34: var_30, var_35: var_31, var_36: var_32, var_37: var_30}
    var_39 = module_0.DictToken(*var_33, **var_38)
    var_40 = "{'key1': 'val1', 'key2': 'val2'}"
    var_41 = {var_9: var_19, var_29: var_39}
    var_42 = len(var_40)
    var_43 = 1
    var_44 = var_42 - var_43
    var_45 = []
    var_46 = 'value'
    var_47 = 'start_index'
    var_48 = 'end_index'
    var_49 = 'content'
    var_50 = {var_46: var_41, var_47: var_1, var_48: var_44, var_49: var_40}
    var_51 = module_0.DictToken(*var_45, **var_50)
    var_52 = var_51._value
    var_53 = bool(var_51._value == {'key1': 'val1', 'key2': 'val2'})
    assert var_53 is True
    var_54 = var_51._start_index
    assert var_54 == 0
    var_55 = len(var_40)
    var_56 = var_55 - var_43
    var_57 = var_51._end_index
    var_58 = bool(var_51._end_index == var_56)
    assert var_58 is True
    var_59 = var_51._content
    var_60 = bool(var_51._content == var_40)
    assert var_60 is True
    var_61 = var_51._child_keys
    var_62 = bool(var_51._child_keys == {'key1': var_9, 'key2': var_29})
    assert var_62 is True
    var_63 = var_51._child_tokens
    var_64 = bool(var_51._child_tokens == {'key1': var_19, 'key2': var_39})
    assert var_64 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 5
    var_3 = 'data'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 0
    var_3 = module_0.ListToken(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 3
    var_3 = 'abcde'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'bcd'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ListToken('abc')"



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_token_eq_same_attributes. Retrieved 4/15 statements.
# Partially parsed test_token_eq_different_values. Retrieved 5/19 statements.
# Partially parsed test_token_eq_different_indices. Retrieved 6/18 statements.
# Partially parsed test_token_eq_different_types. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = 456

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = 6
    var_5 = 1

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: 1, b: 2'
    var_16 = [var_13, var_1, var_14, var_15]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_13)
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 7
    var_23 = var_18._content
    assert var_23 == 'a: 1, b: 2'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_27 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: 1, b: 2'
    var_16 = [var_13, var_1, var_14, var_15]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_13)
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 7
    var_23 = var_18._content
    assert var_23 == 'a: 1, b: 2'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 100
    var_5 = 5
    var_6 = 7
    var_7 = '100'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = 'key: 100'
    var_11 = [var_9, var_1, var_6, var_10]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = var_13.value
    var_15 = bool(var_13.value == {'key': 100})
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'k: 1'
    var_9 = [var_7, var_1, var_4, var_8]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = var_11.string
    assert var_12 == 'k: 1'



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 2
    var_8 = var_4._content
    assert var_8 == '123'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 5
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'abc'
    var_5 = var_3._start_index
    assert var_5 == 5



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 20/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 5
    var_6 = 9
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 11
    var_10 = 13
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 15
    var_14 = 16
    var_15 = '30'
    var_16 = module_0.Token(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = 'name: Alice, age: 30'
    var_19 = []
    var_20 = 'value'
    var_21 = 'start_index'
    var_22 = 'end_index'
    var_23 = 'content'
    var_24 = {var_20: var_17, var_21: var_1, var_22: var_14, var_23: var_18}
    var_25 = module_0.DictToken(*var_19, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == var_17)
    assert var_27 is True
    var_28 = var_25._start_index
    assert var_28 == 0
    var_29 = var_25._end_index
    assert var_29 == 16
    var_30 = var_25._content
    assert var_30 == 'name: Alice, age: 30'
    var_31 = var_25._child_keys
    var_32 = bool(var_25._child_keys == {'name': var_3, 'age': var_11})
    assert var_32 is True
    var_33 = var_25._child_tokens
    var_34 = bool(var_25._child_tokens == {'name': var_7, 'age': var_16})
    assert var_34 is True
    var_35 = var_25.value
    var_36 = bool(var_25.value == {'name': 'Alice', 'age': 30})
    assert var_36 is True

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
    var_14 = var_9.value
    var_15 = bool(var_9.value == {})
    assert var_15 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dict_token_initialization_success. Retrieved 16/30 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'k'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'key1'
    var_5 = 'v'
    var_6 = module_0.Token(var_0, var_1, var_1, var_5)
    var_7 = 'val1'
    var_8 = module_0.Token(var_0, var_1, var_1, var_2)
    var_9 = 'key2'
    var_10 = module_0.Token(var_0, var_1, var_1, var_5)
    var_11 = 'val2'
    var_12 = {var_4: var_6, var_9: var_10}
    var_13 = {var_4: var_6, var_9: var_10}
    var_14 = 5
    var_15 = 'key1: val1, key2: val2'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 29/43 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_6, var_12: var_18}
    var_25 = '{"key1": "val1", "key2": "val2"}'
    var_26 = len(var_25)
    var_27 = 1
    var_28 = var_26 - var_27
    var_29 = 'a'
    var_30 = [var_29, var_1, var_1, var_29]
    var_31 = {}
    var_32 = module_0.DictToken(*var_30, **var_31)
    var_33 = 'b'
    var_34 = 2
    var_35 = [var_33, var_34, var_34, var_33]
    var_36 = {}
    var_37 = module_0.DictToken(*var_35, **var_36)
    var_38 = {var_32: var_37}
    var_39 = 5
    var_40 = '{"a":"b"}'



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = [var_0, var_1, var_1, var_0]
    var_3 = {}
    var_4 = module_0.DictToken(*var_2, **var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = [var_5, var_6, var_6, var_5]
    var_8 = {}
    var_9 = module_0.DictToken(*var_7, **var_8)
    var_10 = 'c'
    var_11 = 4
    var_12 = [var_10, var_11, var_11, var_10]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = 'd'
    var_16 = 6
    var_17 = [var_15, var_16, var_16, var_15]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = {var_0: var_5, var_10: var_15}
    var_21 = {var_4: var_9, var_14: var_19}
    var_22 = 7
    var_23 = 'a: b, c: d'
    var_24 = [var_21, var_1, var_22, var_23]
    var_25 = {}
    var_26 = module_0.DictToken(*var_24, **var_25)
    var_27 = var_26._value
    var_28 = bool(var_26._value == var_20)
    assert var_28 is True
    var_29 = var_26._start_index
    assert var_29 == 0
    var_30 = var_26._end_index
    assert var_30 == 7
    var_31 = var_26._content
    assert var_31 == 'a: b, c: d'
    var_32 = var_26._child_keys['a']
    var_33 = bool(var_26._child_keys['a'] == var_4)
    assert var_33 is True
    var_34 = var_26._child_tokens['a']
    var_35 = bool(var_26._child_tokens['a'] == var_9)
    assert var_35 is True
    var_36 = var_26._child_keys['c']
    var_37 = bool(var_26._child_keys['c'] == var_14)
    assert var_37 is True
    var_38 = var_26._child_tokens['c']
    var_39 = bool(var_26._child_tokens['c'] == var_19)
    assert var_39 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_maps. Retrieved 15/35 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 'key1'
    var_4 = 3
    var_5 = 'val1'
    var_6 = 5
    var_7 = 8
    var_8 = 'key2'
    var_9 = 10
    var_10 = 13
    var_11 = 'val2'
    var_12 = 15
    var_13 = 18
    var_14 = 'content'



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v1'
    var_5 = 2
    var_6 = 3
    var_7 = 'k1:v1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 5
    var_10 = 'k2'
    var_11 = module_0.Token(var_5, var_9, var_9, var_10)
    var_12 = 'v2'
    var_13 = 7
    var_14 = 8
    var_15 = 'k2:v2'
    var_16 = module_0.Token(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_8, var_11: var_16}
    var_18 = 'k1:v1, k2:v2'
    var_19 = 12
    var_20 = [var_17, var_1, var_19, var_18]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_17)
    assert var_24 is True
    var_25 = var_22._start_index
    assert var_25 == 0
    var_26 = var_22._end_index
    assert var_26 == 12
    var_27 = var_22._content
    var_28 = bool(var_22._content == var_18)
    assert var_28 is True
    var_29 = var_22._child_keys[1]
    var_30 = bool(var_22._child_keys[1] == var_3)
    assert var_30 is True
    var_31 = var_22._child_keys[2]
    var_32 = bool(var_22._child_keys[2] == var_11)
    assert var_32 is True
    var_33 = var_22._child_tokens[1]
    var_34 = bool(var_22._child_tokens[1] == var_8)
    assert var_34 is True
    var_35 = var_22._child_tokens[2]
    var_36 = bool(var_22._child_tokens[2] == var_16)
    assert var_36 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v1'
    var_5 = 2
    var_6 = 3
    var_7 = 'k1:v1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = [var_9, var_1, var_6, var_7]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = var_12.string
    assert var_13 == 'k1:v1'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v1'
    var_5 = 2
    var_6 = 3
    var_7 = 'k1:v1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = [var_9, var_1, var_6, var_7]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = [var_9, var_1, var_6, var_7]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = bool(var_12 == var_15)
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_maps. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = 'content'



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_6, var_12: var_18}
    var_25 = {var_5: var_11, var_17: var_23}
    var_26 = 'key1: val1, key2: val2'
    var_27 = [var_24, var_1, var_20, var_26]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = var_29._value
    var_31 = bool(var_29._value == var_24)
    assert var_31 is True
    var_32 = var_29._start_index
    assert var_32 == 0
    var_33 = var_29._end_index
    assert var_33 == 22
    var_34 = var_29._content
    assert var_34 == 'key1: val1, key2: val2'
    var_35 = var_29._child_keys['key1']
    var_36 = bool(var_29._child_keys['key1'] == var_5)
    assert var_36 is True
    var_37 = var_29._child_tokens['key1']
    var_38 = bool(var_29._child_tokens['key1'] == var_11)
    assert var_38 is True
    var_39 = var_29._child_keys['key2']
    var_40 = bool(var_29._child_keys['key2'] == var_17)
    assert var_40 is True
    var_41 = var_29._child_tokens['key2']
    var_42 = bool(var_29._child_tokens['key2'] == var_23)
    assert var_42 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_5: var_11, var_17: var_23}
    var_25 = 'key1: val1, key2: val2'
    var_26 = []
    var_27 = 'value'
    var_28 = 'start_index'
    var_29 = 'end_index'
    var_30 = 'content'
    var_31 = {var_27: var_24, var_28: var_1, var_29: var_20, var_30: var_25}
    var_32 = module_0.DictToken(*var_26, **var_31)
    var_33 = var_32._value
    var_34 = bool(var_32._value == {'key1': 'val1', 'key2': 'val2'})
    assert var_34 is True
    var_35 = var_32._start_index
    assert var_35 == 0
    var_36 = var_32._end_index
    assert var_36 == 22
    var_37 = var_32._content
    assert var_37 == 'key1: val1, key2: val2'
    var_38 = var_32._child_keys['key1']
    var_39 = bool(var_32._child_keys['key1'] == var_5)
    assert var_39 is True
    var_40 = var_32._child_tokens['key1']
    var_41 = bool(var_32._child_tokens['key1'] == var_11)
    assert var_41 is True
    var_42 = var_32._child_keys['key2']
    var_43 = bool(var_32._child_keys['key2'] == var_17)
    assert var_43 is True
    var_44 = var_32._child_tokens['key2']
    var_45 = bool(var_32._child_tokens['key2'] == var_23)
    assert var_45 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_11, var_12: var_23}
    var_25 = 'key1: val1, key2: val2'
    var_26 = [var_24, var_1, var_20, var_25]
    var_27 = {}
    var_28 = module_0.DictToken(*var_26, **var_27)
    var_29 = var_28._value
    var_30 = bool(var_28._value == var_24)
    assert var_30 is True
    var_31 = var_28._start_index
    assert var_31 == 0
    var_32 = var_28._end_index
    assert var_32 == 22
    var_33 = var_28._content
    assert var_33 == 'key1: val1, key2: val2'
    var_34 = var_28._child_keys
    var_35 = bool(var_28._child_keys == {'key1': var_5, 'key2': var_17})
    assert var_35 is True
    var_36 = var_28._child_tokens
    var_37 = bool(var_28._child_tokens == {'key1': var_11, 'key2': var_23})
    assert var_37 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'v'
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = {var_5: var_11}
    var_13 = 'k: v'
    var_14 = [var_12, var_1, var_8, var_13]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16.string
    assert var_17 == 'k: v'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'v'
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 5
    var_13 = 6
    var_14 = [var_0, var_12, var_13, var_0]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = 7
    var_18 = 8
    var_19 = [var_6, var_17, var_18, var_6]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = {var_5: var_11}
    var_23 = 'k: v'
    var_24 = [var_22, var_1, var_8, var_23]
    var_25 = {}
    var_26 = module_0.DictToken(*var_24, **var_25)
    var_27 = {var_16: var_21}
    var_28 = [var_27, var_12, var_18, var_23]
    var_29 = {}
    var_30 = module_0.DictToken(*var_28, **var_29)
    var_31 = bool(var_26 != var_30)
    assert var_31 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dict_token_initialization_and_structure. Retrieved 7/27 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 0
    var_5 = 5
    var_6 = 'abcde'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_structures. Retrieved 17/30 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.Token(var_4, var_10, var_10, var_11)
    var_13 = {var_2: var_6, var_9: var_12}
    var_14 = 'a: 1, b: 2'
    var_15 = 10
    var_16 = [var_13, var_1, var_15, var_14]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._child_keys
    var_20 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_20 is True
    var_21 = var_18._child_tokens
    var_22 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_22 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dict_token_init_initializes_attributes. Retrieved 29/89 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'a'
    var_5 = 2
    var_6 = module_0.Token(var_4, var_5, var_5, var_4)
    var_7 = 4
    var_8 = '2'
    var_9 = module_0.Token(var_5, var_7, var_7, var_8)
    var_10 = 'b'
    var_11 = 6
    var_12 = module_0.Token(var_10, var_11, var_11, var_10)
    var_13 = {var_3: var_6, var_9: var_12}
    var_14 = module_0.Token(var_0, var_1, var_1, var_2)
    var_15 = module_0.Token(var_0, var_1, var_1, var_2)
    var_16 = module_0.Token(var_4, var_5, var_5, var_4)
    var_17 = 10
    var_18 = '10'
    var_19 = 'val'
    var_20 = module_0.Token(var_19, var_5, var_7, var_19)
    var_21 = 20
    var_22 = 5
    var_23 = '20'
    var_24 = 'other'
    var_25 = 7
    var_26 = 11
    var_27 = module_0.Token(var_24, var_25, var_26, var_24)
    var_28 = ''



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = '2'
    var_6 = module_0.Token(var_4, var_4, var_4, var_5)
    var_7 = 3
    var_8 = 4
    var_9 = '3'
    var_10 = module_0.Token(var_7, var_8, var_8, var_9)
    var_11 = 6
    var_12 = '4'
    var_13 = module_0.Token(var_8, var_11, var_11, var_12)
    var_14 = {var_3: var_6, var_10: var_13}
    var_15 = '1: 2, 3: 4'
    var_16 = 9
    var_17 = [var_14, var_1, var_16, var_15]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == {1: 2, 3: 4})
    assert var_21 is True
    var_22 = var_19._start_index
    assert var_22 == 0
    var_23 = var_19._end_index
    assert var_23 == 9
    var_24 = var_19._content
    assert var_24 == '1: 2, 3: 4'
    var_25 = var_19._child_keys[1]
    var_26 = bool(var_19._child_keys[1] == var_3)
    assert var_26 is True
    var_27 = var_19._child_keys[3]
    var_28 = bool(var_19._child_keys[3] == var_10)
    assert var_28 is True
    var_29 = var_19._child_tokens[1]
    var_30 = bool(var_19._child_tokens[1] == var_6)
    assert var_30 is True
    var_31 = var_19._child_tokens[3]
    var_32 = bool(var_19._child_tokens[3] == var_13)
    assert var_32 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = 'key2'
    var_13 = 12
    var_14 = 16
    var_15 = [var_12, var_13, var_14, var_12]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'val2'
    var_19 = 18
    var_20 = 22
    var_21 = [var_18, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = {var_0: var_11, var_12: var_23}
    var_25 = {var_5: var_11, var_17: var_23}
    var_26 = 'key1: val1, key2: val2'
    var_27 = [var_25, var_1, var_20, var_26]
    var_28 = {}
    var_29 = module_0.DictToken(*var_27, **var_28)
    var_30 = var_29._value
    var_31 = bool(var_29._value == var_25)
    assert var_31 is True
    var_32 = var_29._start_index
    assert var_32 == 0
    var_33 = var_29._end_index
    assert var_33 == 22
    var_34 = var_29._content
    assert var_34 == 'key1: val1, key2: val2'
    var_35 = var_29._child_keys['key1']
    var_36 = bool(var_29._child_keys['key1'] == var_5)
    assert var_36 is True
    var_37 = var_29._child_keys['key2']
    var_38 = bool(var_29._child_keys['key2'] == var_17)
    assert var_38 is True
    var_39 = var_29._child_tokens['key1']
    var_40 = bool(var_29._child_tokens['key1'] == var_11)
    assert var_40 is True
    var_41 = var_29._child_tokens['key2']
    var_42 = bool(var_29._child_tokens['key2'] == var_23)
    assert var_42 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val1'
    var_7 = 6
    var_8 = 10
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = {var_5: var_11}
    var_13 = 'key1: val1'
    var_14 = [var_12, var_1, var_8, var_13]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16.string
    assert var_17 == 'key1: val'



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'Alice'
    var_5 = 5
    var_6 = 9
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'age'
    var_9 = 11
    var_10 = 13
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 30
    var_13 = 15
    var_14 = 16
    var_15 = '30'
    var_16 = module_0.Token(var_12, var_13, var_14, var_15)
    var_17 = {var_3: var_7, var_11: var_16}
    var_18 = "name: 'Alice', age: 30"
    var_19 = [var_17, var_1, var_14, var_18]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._value
    var_23 = bool(var_21._value == {'name': 'Alice', 'age': 30})
    assert var_23 is True
    var_24 = var_21._start_index
    assert var_24 == 0
    var_25 = var_21._end_index
    assert var_25 == 16
    var_26 = var_21._content
    assert var_26 == "name: 'Alice', age: 30"
    var_27 = var_21.string
    assert var_27 == "name: 'Alice', age: 30"
    var_28 = var_21._child_keys['name']
    var_29 = bool(var_21._child_keys['name'] == var_3)
    assert var_29 is True
    var_30 = var_21._child_keys['age']
    var_31 = bool(var_21._child_keys['age'] == var_11)
    assert var_31 is True
    var_32 = var_21._child_tokens['name']
    var_33 = bool(var_21._child_tokens['name'] == var_7)
    assert var_33 is True
    var_34 = var_21._child_tokens['age']
    var_35 = bool(var_21._child_tokens['age'] == var_16)
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = -1
    var_3 = ''
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._child_keys
    var_10 = bool(var_6._child_keys == {})
    assert var_10 is True
    var_11 = var_6._child_tokens
    var_12 = bool(var_6._child_tokens == {})
    assert var_12 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == var_0)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dict_token_init_initializes_attributes. Retrieved 13/25 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'key1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'k1'
    var_5 = 'v1'
    var_6 = 2
    var_7 = 'k2'
    var_8 = 4
    var_9 = 'v2'
    var_10 = 6
    var_11 = 10
    var_12 = 'k1:v1, k2:v2'



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data_payload'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 10
    var_3 = '01234567890'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    assert var_5 == 5



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abcdefg'
    var_1 = 'abc'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = var_4.string
    assert var_5 == 'bcd'



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == var_0)
    assert var_5 is True
    var_6 = var_3._start_index
    var_7 = bool(var_3._start_index == var_1)
    assert var_7 is True
    var_8 = var_3._end_index
    var_9 = bool(var_3._end_index == var_2)
    assert var_9 is True
    var_10 = var_3._content
    assert var_10 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)
    var_5 = var_4.string
    assert var_5 == 'hello'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = "'1'"
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = repr(var_3)
    assert var_4 == 'Token("\'1\'")'



