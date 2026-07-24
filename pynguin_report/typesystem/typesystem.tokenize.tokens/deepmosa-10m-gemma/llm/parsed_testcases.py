####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = [var_0, var_1, var_2, var_0]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = 'val'
    var_7 = 5
    var_8 = 8
    var_9 = [var_6, var_7, var_8, var_6]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = {var_0: var_6}
    var_13 = 'key: val'
    var_14 = [var_12, var_1, var_8, var_13]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_12)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 8
    var_21 = var_16._content
    assert var_21 == 'key: val'
    var_22 = 'key'
    var_23 = bool('key' in var_16._child_keys)
    assert var_23 is True
    var_24 = 'key'
    var_25 = bool('key' in var_16._child_tokens)
    assert var_25 is True
    var_26 = var_16._child_keys['key']
    var_27 = bool(var_16._child_keys['key'] == var_5)
    assert var_27 is True
    var_28 = var_16._child_tokens['key']
    var_29 = bool(var_16._child_tokens['key'] == var_11)
    assert var_29 is True



# Parsed testcases at query #3
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
    var_9 = var_4.string
    assert var_9 == '123'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''
    var_5 = var_3.string
    assert var_5 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 4
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'cde'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_token_eq_success. Retrieved 5/16 statements.
# Partially parsed test_token_eq_failure_different_value. Retrieved 5/19 statements.
# Partially parsed test_token_eq_failure_different_start_index. Retrieved 5/16 statements.
# Partially parsed test_token_eq_failure_different_end_index. Retrieved 5/16 statements.
# Partially parsed test_token_eq_different_type. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = 'world'

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
    var_4 = 1

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = 4

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = 'not a token'



# Parsed testcases at query #5
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
    var_48 = var_36._child_keys['key2']
    var_49 = bool(var_36._child_keys['key2'] == var_17)
    assert var_49 is True
    var_50 = var_36._child_tokens['key1']
    var_51 = bool(var_36._child_tokens['key1'] == var_11)
    assert var_51 is True
    var_52 = var_36._child_tokens['key2']
    var_53 = bool(var_36._child_tokens['key2'] == var_23)
    assert var_53 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 19/38 statements.


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
    var_22 = 15
    var_23 = []
    var_24 = 'value'
    var_25 = 'start_index'
    var_26 = 'end_index'
    var_27 = 'content'
    var_28 = {var_24: var_20, var_25: var_21, var_26: var_22, var_27: var_20}
    var_29 = module_0.DictToken(*var_23, **var_28)
    var_30 = 'val2'
    var_31 = 17
    var_32 = 20
    var_33 = []
    var_34 = 'value'
    var_35 = 'start_index'
    var_36 = 'end_index'
    var_37 = 'content'
    var_38 = {var_34: var_30, var_35: var_31, var_36: var_32, var_37: var_30}
    var_39 = module_0.DictToken(*var_33, **var_38)
    var_40 = {var_9: var_19, var_29: var_39}
    var_41 = 'key1: val1, key2: val2'
    var_42 = []
    var_43 = 'value'
    var_44 = 'start_index'
    var_45 = 'end_index'
    var_46 = 'content'
    var_47 = {var_43: var_40, var_44: var_1, var_45: var_32, var_46: var_41}
    var_48 = module_0.DictToken(*var_42, **var_47)
    var_49 = var_48._child_keys['key1']
    var_50 = bool(var_48._child_keys['key1'] == var_9)
    assert var_50 is True
    var_51 = var_48._child_tokens['key1']
    var_52 = bool(var_48._child_tokens['key1'] == var_19)
    assert var_52 is True
    var_53 = var_48._child_keys['key2']
    var_54 = bool(var_48._child_keys['key2'] == var_29)
    assert var_54 is True
    var_55 = var_48._child_tokens['key2']
    var_56 = bool(var_48._child_tokens['key2'] == var_39)
    assert var_56 is True
    var_57 = var_48.string
    assert var_57 == 'key1: val1, key2: val2'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dict_token_init_raises_error_if_value_not_dict. Retrieved 7/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = '123'
    var_2 = 'key'
    var_3 = 0
    var_4 = 3
    var_5 = module_0.Token(var_2, var_3, var_4, var_2)
    var_6 = 'val'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3.string
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'hello'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = repr(var_3)
    assert var_4 == "Token('abc')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = module_0.Token(var_0, var_1, var_1, var_0)
    var_4 = 'b'
    var_5 = module_0.Token(var_4, var_1, var_1, var_4)
    var_6 = bool(var_2 == var_3)
    assert var_6 is True
    var_7 = bool(var_2 != var_5)
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 24/39 statements.


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
    var_25 = [var_0, var_1, var_2, var_0]
    var_26 = {}
    var_27 = module_0.DictToken(*var_25, **var_26)
    var_28 = [var_6, var_7, var_8, var_6]
    var_29 = {}
    var_30 = module_0.DictToken(*var_28, **var_29)
    var_31 = [var_12, var_13, var_14, var_12]
    var_32 = {}
    var_33 = module_0.DictToken(*var_31, **var_32)
    var_34 = [var_18, var_19, var_20, var_18]
    var_35 = {}
    var_36 = module_0.DictToken(*var_34, **var_35)
    var_37 = 'mock_val1'
    var_38 = 'mock_val2'
    var_39 = 'key1: val1, key2: val2'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
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
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #4
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
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dict_token_init_child_keys_assignment. Retrieved 9/28 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v1'
    var_5 = 'k2'
    var_6 = 'v2'
    var_7 = 5
    var_8 = 'k1:v1, k2:v2'
    var_9 = 'k1'



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'val1'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = []
    var_5 = 1
    var_6 = 'val2'
    var_7 = module_0.ListToken(var_4, var_5, var_5, var_6)
    var_8 = [var_3, var_7]
    var_9 = 5
    var_10 = 'val1, val2'
    var_11 = module_0.ListToken(var_8, var_1, var_9, var_10)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [var_3, var_7])
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 0
    var_15 = var_11._end_index
    assert var_15 == 5
    var_16 = var_11._content
    assert var_16 == 'val1, val2'
    var_17 = var_11.string
    assert var_17 == 'val1, val'



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'

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
    var_1 = 0
    var_2 = 4
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'abcde'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 3
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'bcd'



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'

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
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = '012345'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == '123'



# Parsed testcases at query #9
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
    var_26 = [var_24, var_1, var_20, var_25]
    var_27 = {}
    var_28 = module_0.DictToken(*var_26, **var_27)
    var_29 = var_28._value
    var_30 = bool(var_28._value == {'key1': 'val1', 'key2': 'val2'})
    assert var_30 is True
    var_31 = var_28._start_index
    assert var_31 == 0
    var_32 = var_28._end_index
    assert var_32 == 22
    var_33 = var_28._content
    assert var_33 == 'key1: val1, key2: val2'
    var_34 = var_28._child_keys['key1']
    var_35 = bool(var_28._child_keys['key1'] == var_5)
    assert var_35 is True
    var_36 = var_28._child_keys['key2']
    var_37 = bool(var_28._child_keys['key2'] == var_17)
    assert var_37 is True
    var_38 = var_28._child_tokens['key1']
    var_39 = bool(var_28._child_tokens['key1'] == var_11)
    assert var_39 is True
    var_40 = var_28._child_tokens['key2']
    var_41 = bool(var_28._child_tokens['key2'] == var_23)
    assert var_41 is True
    var_42 = var_28.string
    assert var_42 == 'key1: val1, key2: val2'



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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
    var_25 = 'key1: val1, key2: val2'
    var_26 = {var_5: var_11, var_17: var_23}
    var_27 = []
    var_28 = 'value'
    var_29 = 'start_index'
    var_30 = 'end_index'
    var_31 = 'content'
    var_32 = {var_28: var_26, var_29: var_1, var_30: var_20, var_31: var_25}
    var_33 = module_0.DictToken(*var_27, **var_32)
    var_34 = var_33._value
    var_35 = bool(var_33._value == var_24)
    assert var_35 is True
    var_36 = var_33._start_index
    assert var_36 == 0
    var_37 = var_33._end_index
    assert var_37 == 22
    var_38 = var_33._content
    var_39 = bool(var_33._content == var_25)
    assert var_39 is True
    var_40 = var_33._child_keys
    var_41 = bool(var_33._child_keys == {'key1': var_5, 'key2': var_17})
    assert var_41 is True
    var_42 = var_33._child_tokens
    var_43 = bool(var_33._child_tokens == {'key1': var_11, 'key2': var_23})
    assert var_43 is True
    var_44 = var_33.string
    assert var_44 == 'key1: val1, key2: val2'



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_maps. Retrieved 17/21 statements.


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
    var_19 = var_18._child_keys
    var_20 = bool(var_18._child_keys == {'a': var_2, 'b': var_9})
    assert var_20 is True
    var_21 = var_18._child_tokens
    var_22 = bool(var_18._child_tokens == {'a': var_6, 'b': var_12})
    assert var_22 is True



