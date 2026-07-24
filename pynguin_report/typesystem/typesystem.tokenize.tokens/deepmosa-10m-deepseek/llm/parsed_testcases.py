####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'sample'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = -5
    var_2 = -1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == -5
    var_8 = var_4._end_index
    assert var_8 == -1
    var_9 = var_4._content
    assert var_9 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 1000
    var_2 = 2000
    var_3 = 'x'
    var_4 = 3000
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 1000
    var_10 = var_6._end_index
    assert var_10 == 2000
    var_11 = var_6._content
    var_12 = bool(var_6._content == 'x' * 3000)
    assert var_12 is True



# Parsed testcases at query #2
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = -5
    var_2 = -1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == -5
    var_8 = var_4._end_index
    assert var_8 == -1
    var_9 = var_4._content
    assert var_9 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 100
    var_2 = 200
    var_3 = 'x'
    var_4 = 300
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 100
    var_10 = var_6._end_index
    assert var_10 == 200
    var_11 = var_6._content
    var_12 = bool(var_6._content == 'x' * 300)
    assert var_12 is True



# Parsed testcases at query #3
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True

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
    var_10 = 'key: value'
    var_11 = []
    var_12 = 'value'
    var_13 = 'start_index'
    var_14 = 'end_index'
    var_15 = 'content'
    var_16 = {var_12: var_9, var_13: var_1, var_14: var_7, var_15: var_10}
    var_17 = module_0.DictToken(*var_11, **var_16)
    var_18 = var_17._value
    var_19 = bool(var_17._value == var_9)
    assert var_19 is True
    var_20 = var_17._start_index
    assert var_20 == 0
    var_21 = var_17._end_index
    assert var_21 == 9
    var_22 = var_17._content
    var_23 = bool(var_17._content == var_10)
    assert var_23 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = -1
    var_3 = ''
    var_4 = []
    var_5 = 'value'
    var_6 = 'start_index'
    var_7 = 'end_index'
    var_8 = 'content'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.DictToken(*var_4, **var_9)
    var_11 = var_10._child_keys
    var_12 = bool(var_10._child_keys == {})
    assert var_12 is True
    var_13 = var_10._child_tokens
    var_14 = bool(var_10._child_tokens == {})
    assert var_14 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: val1, key2: val2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val1'
    var_6 = 7
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'val2'
    var_14 = 20
    var_15 = 23
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_17, var_20: var_1, var_21: var_15, var_22: var_3}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = bool(var_24._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_26 is True
    var_27 = var_24._child_tokens
    var_28 = bool(var_24._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_28 is True



# Parsed testcases at query #4
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20
    var_3 = 'some content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 10
    var_7 = var_4._end_index
    assert var_7 == 20
    var_8 = var_4._content
    assert var_8 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = var_2._value
    assert var_3 == ''
    var_4 = var_2._start_index
    assert var_4 == 0
    var_5 = var_2._end_index
    assert var_5 == 0
    var_6 = var_2._content
    assert var_6 == ''



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = 7
    var_3 = 1
    var_4 = module_0.Token(var_3, var_3, var_3, var_0)
    var_5 = 2
    var_6 = 4
    var_7 = module_0.Token(var_5, var_6, var_6, var_0)
    var_8 = 3
    var_9 = 7
    var_10 = module_0.Token(var_8, var_9, var_9, var_0)
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.ListToken(var_11, var_1, var_2, var_0)
    var_13 = var_12._value
    var_14 = bool(var_12._value == var_11)
    assert var_14 is True
    var_15 = var_12._start_index
    var_16 = bool(var_12._start_index == var_1)
    assert var_16 is True
    var_17 = var_12._end_index
    var_18 = bool(var_12._end_index == var_2)
    assert var_18 is True
    var_19 = var_12._content
    var_20 = bool(var_12._content == var_0)
    assert var_20 is True



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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True

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
    var_10 = var_9._value
    var_11 = bool(var_9._value == var_0)
    assert var_11 is True
    var_12 = var_9._start_index
    assert var_12 == 0
    var_13 = var_9._end_index
    assert var_13 == 0
    var_14 = var_9._content
    assert var_14 == ''
    var_15 = var_9._child_keys
    var_16 = bool(var_9._child_keys == {})
    assert var_16 is True
    var_17 = var_9._child_tokens
    var_18 = bool(var_9._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'a:1,b:2'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = 6
    var_11 = module_0.Token(var_5, var_10, var_10, var_2)
    var_12 = {var_3: var_6, var_9: var_11}
    var_13 = []
    var_14 = 'value'
    var_15 = 'start_index'
    var_16 = 'end_index'
    var_17 = 'content'
    var_18 = {var_14: var_12, var_15: var_1, var_16: var_10, var_17: var_2}
    var_19 = module_0.DictToken(*var_13, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == var_12)
    assert var_21 is True
    var_22 = var_19._start_index
    assert var_22 == 0
    var_23 = var_19._end_index
    assert var_23 == 6
    var_24 = var_19._content
    assert var_24 == 'a:1,b:2'
    var_25 = var_19._child_keys
    var_26 = bool(var_19._child_keys == {'a': var_3, 'b': var_9})
    assert var_26 is True
    var_27 = var_19._child_tokens
    var_28 = bool(var_19._child_tokens == {'a': var_6, 'b': var_11})
    assert var_28 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'value'
    var_9 = 7
    var_10 = 13
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = [var_12, var_1, var_4, var_0]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_12)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_1)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_4)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_0)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key': var_7})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key': var_11})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'a'
    var_6 = 2
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 5
    var_9 = module_0.Token(var_3, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 8
    var_12 = 9
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 12
    var_15 = module_0.Token(var_6, var_14, var_14, var_0)
    var_16 = {var_7: var_9, var_13: var_15}
    var_17 = [var_16, var_1, var_4, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == var_16)
    assert var_21 is True
    var_22 = var_19._start_index
    var_23 = bool(var_19._start_index == var_1)
    assert var_23 is True
    var_24 = var_19._end_index
    var_25 = bool(var_19._end_index == var_4)
    assert var_25 is True
    var_26 = var_19._content
    var_27 = bool(var_19._content == var_0)
    assert var_27 is True
    var_28 = var_19._child_keys
    var_29 = bool(var_19._child_keys == {'a': var_7, 'b': var_13})
    assert var_29 is True
    var_30 = var_19._child_tokens
    var_31 = bool(var_19._child_tokens == {'a': var_9, 'b': var_15})
    assert var_31 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'first'
    var_9 = 7
    var_10 = 13
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 16
    var_13 = 19
    var_14 = module_0.Token(var_5, var_12, var_13, var_0)
    var_15 = 'second'
    var_16 = 22
    var_17 = 28
    var_18 = module_0.Token(var_15, var_16, var_17, var_0)
    var_19 = {var_7: var_11, var_14: var_18}
    var_20 = [var_19, var_1, var_4, var_0]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_19)
    assert var_24 is True
    var_25 = var_22._start_index
    var_26 = bool(var_22._start_index == var_1)
    assert var_26 is True
    var_27 = var_22._end_index
    var_28 = bool(var_22._end_index == var_4)
    assert var_28 is True
    var_29 = var_22._content
    var_30 = bool(var_22._content == var_0)
    assert var_30 is True
    var_31 = var_22._child_keys
    var_32 = bool(var_22._child_keys == {'key': var_14})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'key': var_18})
    assert var_34 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "{1: 'one', 2: 'two'}"
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = module_0.Token(var_3, var_3, var_3, var_0)
    var_6 = 'one'
    var_7 = 5
    var_8 = 9
    var_9 = module_0.Token(var_6, var_7, var_8, var_0)
    var_10 = 2
    var_11 = 12
    var_12 = module_0.Token(var_10, var_11, var_11, var_0)
    var_13 = 'two'
    var_14 = 16
    var_15 = 20
    var_16 = module_0.Token(var_13, var_14, var_15, var_0)
    var_17 = {var_5: var_9, var_12: var_16}
    var_18 = [var_17, var_1, var_4, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_17)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_1)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_4)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_0)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {1: var_5, 2: var_12})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {1: var_9, 2: var_16})
    assert var_32 is True



# Parsed testcases at query #8
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 1
    var_7 = var_4._end_index
    assert var_7 == 2
    var_8 = var_4._content
    assert var_8 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = -5
    var_2 = -1
    var_3 = 'negative'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == -5
    var_8 = var_4._end_index
    assert var_8 == -1
    var_9 = var_4._content
    assert var_9 == 'negative'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = var_2._value
    assert var_3 == ''
    var_4 = var_2._start_index
    assert var_4 == 0
    var_5 = var_2._end_index
    assert var_5 == 0
    var_6 = var_2._content
    assert var_6 == ''



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 15
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == var_11)
    assert var_16 is True
    var_17 = var_14._start_index
    var_18 = bool(var_14._start_index == var_1)
    assert var_18 is True
    var_19 = var_14._end_index
    var_20 = bool(var_14._end_index == var_2)
    assert var_20 is True
    var_21 = var_14._content
    var_22 = bool(var_14._content == var_0)
    assert var_22 is True
    var_23 = var_14._child_keys
    var_24 = bool(var_14._child_keys == {'key': var_6})
    assert var_24 is True
    var_25 = var_14._child_tokens
    var_26 = bool(var_14._child_tokens == {'key': var_10})
    assert var_26 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 16
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = 15
    var_15 = module_0.Token(var_5, var_13, var_14, var_0)
    var_16 = {var_6: var_8, var_12: var_15}
    var_17 = [var_16, var_1, var_2, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == var_16)
    assert var_21 is True
    var_22 = var_19._start_index
    var_23 = bool(var_19._start_index == var_1)
    assert var_23 is True
    var_24 = var_19._end_index
    var_25 = bool(var_19._end_index == var_2)
    assert var_25 is True
    var_26 = var_19._content
    var_27 = bool(var_19._content == var_0)
    assert var_27 is True
    var_28 = var_19._child_keys
    var_29 = bool(var_19._child_keys == {'a': var_6, 'b': var_12})
    assert var_29 is True
    var_30 = var_19._child_tokens
    var_31 = bool(var_19._child_tokens == {'a': var_8, 'b': var_15})
    assert var_31 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 3}}'
    var_1 = 0
    var_2 = 24
    var_3 = 'outer'
    var_4 = 1
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 11
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 3
    var_12 = 20
    var_13 = module_0.Token(var_11, var_12, var_12, var_0)
    var_14 = {var_10: var_13}
    var_15 = 10
    var_16 = 21
    var_17 = [var_14, var_15, var_16, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = {var_6: var_19}
    var_21 = [var_20, var_1, var_2, var_0]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_20)
    assert var_25 is True
    var_26 = var_23._start_index
    var_27 = bool(var_23._start_index == var_1)
    assert var_27 is True
    var_28 = var_23._end_index
    var_29 = bool(var_23._end_index == var_2)
    assert var_29 is True
    var_30 = var_23._content
    var_31 = bool(var_23._content == var_0)
    assert var_31 is True
    var_32 = var_23._child_keys
    var_33 = bool(var_23._child_keys == {'outer': var_6})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'outer': var_19})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = 32
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'first'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 17
    var_12 = 20
    var_13 = module_0.Token(var_3, var_11, var_12, var_0)
    var_14 = 'second'
    var_15 = 24
    var_16 = 31
    var_17 = module_0.Token(var_14, var_15, var_16, var_0)
    var_18 = {var_6: var_10, var_13: var_17}
    var_19 = [var_18, var_1, var_2, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._value
    var_23 = bool(var_21._value == var_18)
    assert var_23 is True
    var_24 = var_21._start_index
    var_25 = bool(var_21._start_index == var_1)
    assert var_25 is True
    var_26 = var_21._end_index
    var_27 = bool(var_21._end_index == var_2)
    assert var_27 is True
    var_28 = var_21._content
    var_29 = bool(var_21._content == var_0)
    assert var_29 is True
    var_30 = var_21._child_keys
    var_31 = bool(var_21._child_keys == {'key': var_13})
    assert var_31 is True
    var_32 = var_21._child_tokens
    var_33 = bool(var_21._child_tokens == {'key': var_17})
    assert var_33 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_8)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 7
    var_21 = var_16._content
    assert var_21 == '"key": 1'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_7})
    assert var_25 is True



# Parsed testcases at query #12
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #13
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 20
    var_3 = 'some text'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 10
    var_7 = var_4._end_index
    assert var_7 == 20
    var_8 = var_4._content
    assert var_8 == 'some text'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 0
    var_1 = -1
    var_2 = ''
    var_3 = module_0.Token(var_0, var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 0
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == -1
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #14
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 1
    var_7 = var_4._end_index
    assert var_7 == 2
    var_8 = var_4._content
    assert var_8 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = var_2._value
    assert var_3 == ''
    var_4 = var_2._start_index
    assert var_4 == 0
    var_5 = var_2._end_index
    assert var_5 == 0
    var_6 = var_2._content
    assert var_6 == ''



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 1
    var_11 = var_6._content
    assert var_11 == '{}'
    var_12 = var_6._child_keys
    var_13 = bool(var_6._child_keys == {})
    assert var_13 is True
    var_14 = var_6._child_tokens
    var_15 = bool(var_6._child_tokens == {})
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 15
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == {var_6: var_10})
    assert var_16 is True
    var_17 = var_14._start_index
    assert var_17 == 0
    var_18 = var_14._end_index
    assert var_18 == 15
    var_19 = var_14._content
    assert var_19 == '{"key": "value"}'
    var_20 = var_14._child_keys
    var_21 = bool(var_14._child_keys == {'key': var_6})
    assert var_21 is True
    var_22 = var_14._child_tokens
    var_23 = bool(var_14._child_tokens == {'key': var_10})
    assert var_23 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 16
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = 15
    var_15 = module_0.Token(var_5, var_13, var_14, var_0)
    var_16 = {var_6: var_8, var_12: var_15}
    var_17 = [var_16, var_1, var_2, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == {var_6: var_8, var_12: var_15})
    assert var_21 is True
    var_22 = var_19._start_index
    assert var_22 == 0
    var_23 = var_19._end_index
    assert var_23 == 16
    var_24 = var_19._content
    assert var_24 == '{"a": 1, "b": 2}'
    var_25 = var_19._child_keys
    var_26 = bool(var_19._child_keys == {'a': var_6, 'b': var_12})
    assert var_26 is True
    var_27 = var_19._child_tokens
    var_28 = bool(var_19._child_tokens == {'a': var_8, 'b': var_15})
    assert var_28 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"nested": {}}'
    var_1 = 0
    var_2 = 13
    var_3 = 'nested'
    var_4 = 1
    var_5 = 7
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = {}
    var_8 = 10
    var_9 = 11
    var_10 = [var_7, var_8, var_9, var_0]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = {var_6: var_12}
    var_14 = [var_13, var_1, var_2, var_0]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == {var_6: var_12})
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 13
    var_21 = var_16._content
    assert var_21 == '{"nested": {}}'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'nested': var_6})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'nested': var_12})
    assert var_25 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = 30
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'first'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 17
    var_12 = 20
    var_13 = module_0.Token(var_3, var_11, var_12, var_0)
    var_14 = 'second'
    var_15 = 24
    var_16 = 30
    var_17 = module_0.Token(var_14, var_15, var_16, var_0)
    var_18 = {var_6: var_10, var_13: var_17}
    var_19 = [var_18, var_1, var_2, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._value
    var_23 = bool(var_21._value == {var_6: var_10, var_13: var_17})
    assert var_23 is True
    var_24 = var_21._start_index
    assert var_24 == 0
    var_25 = var_21._end_index
    assert var_25 == 30
    var_26 = var_21._content
    assert var_26 == '{"key": "first", "key": "second"}'
    var_27 = var_21._child_keys
    var_28 = bool(var_21._child_keys == {'key': var_13})
    assert var_28 is True
    var_29 = var_21._child_tokens
    var_30 = bool(var_21._child_tokens == {'key': var_17})
    assert var_30 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": "value"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #17
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #18
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 1
    var_11 = var_6._content
    assert var_11 == '{}'
    var_12 = var_6._child_keys
    var_13 = bool(var_6._child_keys == {})
    assert var_13 is True
    var_14 = var_6._child_tokens
    var_15 = bool(var_6._child_tokens == {})
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 15
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == {var_6: var_10})
    assert var_16 is True
    var_17 = var_14._start_index
    assert var_17 == 0
    var_18 = var_14._end_index
    assert var_18 == 15
    var_19 = var_14._content
    assert var_19 == '{"key": "value"}'
    var_20 = var_14._child_keys
    var_21 = bool(var_14._child_keys == {'key': var_6})
    assert var_21 is True
    var_22 = var_14._child_tokens
    var_23 = bool(var_14._child_tokens == {'key': var_10})
    assert var_23 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 16
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = module_0.Token(var_5, var_13, var_13, var_0)
    var_15 = {var_6: var_8, var_12: var_14}
    var_16 = [var_15, var_1, var_2, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == {var_6: var_8, var_12: var_14})
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 16
    var_23 = var_18._content
    assert var_23 == '{"a": 1, "b": 2}'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_6, 'b': var_12})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_8, 'b': var_14})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 3}}'
    var_1 = 0
    var_2 = 24
    var_3 = 'outer'
    var_4 = 1
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 11
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 3
    var_12 = 20
    var_13 = module_0.Token(var_11, var_12, var_12, var_0)
    var_14 = {var_10: var_13}
    var_15 = 10
    var_16 = 22
    var_17 = [var_14, var_15, var_16, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = {var_6: var_19}
    var_21 = [var_20, var_1, var_2, var_0]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == {var_6: var_19})
    assert var_25 is True
    var_26 = var_23._start_index
    assert var_26 == 0
    var_27 = var_23._end_index
    assert var_27 == 24
    var_28 = var_23._content
    assert var_28 == '{"outer": {"inner": 3}}'
    var_29 = var_23._child_keys
    var_30 = bool(var_23._child_keys == {'outer': var_6})
    assert var_30 is True
    var_31 = var_23._child_tokens
    var_32 = bool(var_23._child_tokens == {'outer': var_19})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = {}
    var_3 = [var_2, var_0, var_1]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = var_5._value
    var_7 = bool(var_5._value == {})
    assert var_7 is True
    var_8 = var_5._start_index
    assert var_8 == 5
    var_9 = var_5._end_index
    assert var_9 == 10
    var_10 = var_5._content
    assert var_10 == ''
    var_11 = var_5._child_keys
    var_12 = bool(var_5._child_keys == {})
    assert var_12 is True
    var_13 = var_5._child_tokens
    var_14 = bool(var_5._child_tokens == {})
    assert var_14 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"x": 10, "y": 20}'
    var_1 = 0
    var_2 = 18
    var_3 = 'x'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 10
    var_8 = 6
    var_9 = 7
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 'y'
    var_12 = 11
    var_13 = module_0.Token(var_11, var_7, var_12, var_0)
    var_14 = 20
    var_15 = 15
    var_16 = 16
    var_17 = module_0.Token(var_14, var_15, var_16, var_0)
    var_18 = {var_6: var_10, var_13: var_17}
    var_19 = [var_18, var_1, var_2, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._child_keys['x']
    var_23 = bool(var_21._child_keys['x'] == var_6)
    assert var_23 is True
    var_24 = var_21._child_keys['y']
    var_25 = bool(var_21._child_keys['y'] == var_13)
    assert var_25 is True
    var_26 = var_21._child_tokens['x']
    var_27 = bool(var_21._child_tokens['x'] == var_10)
    assert var_27 is True
    var_28 = var_21._child_tokens['y']
    var_29 = bool(var_21._child_tokens['y'] == var_17)
    assert var_29 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'sample'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = -2
    var_2 = -1
    var_3 = 'ab'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == -2
    var_8 = var_4._end_index
    assert var_8 == -1
    var_9 = var_4._content
    assert var_9 == 'ab'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 100
    var_2 = 200
    var_3 = 'x'
    var_4 = 201
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 100
    var_10 = var_6._end_index
    assert var_10 == 200
    var_11 = var_6._content
    var_12 = bool(var_6._content == 'x' * 201)
    assert var_12 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #23
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'sample'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = 6
    var_3 = 'abcdefg'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'cdefg'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 3
    var_2 = 'hello'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3.string
    assert var_4 == 'l'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = var_2.value
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = 10
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.start
    var_6 = var_5.line_no
    assert var_6 == 2
    var_7 = var_5.column_no
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 5

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 12
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.end
    var_6 = var_5.line_no
    assert var_6 == 3
    var_7 = var_5.column_no
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 12

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 0
    var_4 = [var_3]
    var_5 = var_2.lookup(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 0
    var_4 = [var_3]
    var_5 = var_2.lookup_key(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 3
    var_3 = 'abcd'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('bcd')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 100
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 200
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = bool(var_4 == var_5)
    assert var_8 is True
    var_9 = bool(not var_4 == var_7)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = bool(not var_2 == 'not a token')
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -1
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = -5
    var_2 = 0
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 6
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key: value'
    var_10 = 0
    var_11 = 10
    var_12 = {var_4: var_8}
    var_13 = [var_12, var_10, var_11, var_9]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_12)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_10)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_11)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_9)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key': var_4})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key': var_8})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = 4
    var_3 = 'key1: val1, key2: val2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val1'
    var_6 = 7
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'val2'
    var_14 = 19
    var_15 = 22
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = 'key1: val1, key2: val2'
    var_18 = 0
    var_19 = 22
    var_20 = {var_4: var_8, var_12: var_16}
    var_21 = [var_20, var_18, var_19, var_17]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_20)
    assert var_25 is True
    var_26 = var_23._start_index
    var_27 = bool(var_23._start_index == var_18)
    assert var_27 is True
    var_28 = var_23._end_index
    var_29 = bool(var_23._end_index == var_19)
    assert var_29 is True
    var_30 = var_23._content
    var_31 = bool(var_23._content == var_17)
    assert var_31 is True
    var_32 = var_23._child_keys
    var_33 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 8
    var_2 = 12
    var_3 = 'outer: {inner: val}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val'
    var_6 = 15
    var_7 = 17
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 7
    var_11 = 18
    var_12 = [var_9, var_10, var_11, var_3]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = 'outer'
    var_16 = 0
    var_17 = 4
    var_18 = module_0.Token(var_15, var_16, var_17, var_3)
    var_19 = 'outer: {inner: val}'
    var_20 = 0
    var_21 = 19
    var_22 = {var_18: var_14}
    var_23 = [var_22, var_20, var_21, var_19]
    var_24 = {}
    var_25 = module_0.DictToken(*var_23, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == var_22)
    assert var_27 is True
    var_28 = var_25._start_index
    var_29 = bool(var_25._start_index == var_20)
    assert var_29 is True
    var_30 = var_25._end_index
    var_31 = bool(var_25._end_index == var_21)
    assert var_31 is True
    var_32 = var_25._content
    var_33 = bool(var_25._content == var_19)
    assert var_33 is True
    var_34 = var_25._child_keys
    var_35 = bool(var_25._child_keys == {'outer': var_18})
    assert var_35 is True
    var_36 = var_25._child_tokens
    var_37 = bool(var_25._child_tokens == {'outer': var_14})
    assert var_37 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = 'key: val1, key: val2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val1'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 12
    var_10 = 14
    var_11 = module_0.Token(var_0, var_9, var_10, var_3)
    var_12 = 'val2'
    var_13 = 17
    var_14 = 20
    var_15 = module_0.Token(var_12, var_13, var_14, var_3)
    var_16 = 'key: val1, key: val2'
    var_17 = 0
    var_18 = 20
    var_19 = {var_4: var_8, var_11: var_15}
    var_20 = [var_19, var_17, var_18, var_16]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_19)
    assert var_24 is True
    var_25 = var_22._start_index
    var_26 = bool(var_22._start_index == var_17)
    assert var_26 is True
    var_27 = var_22._end_index
    var_28 = bool(var_22._end_index == var_18)
    assert var_28 is True
    var_29 = var_22._content
    var_30 = bool(var_22._content == var_16)
    assert var_30 is True
    var_31 = var_22._child_keys
    var_32 = bool(var_22._child_keys == {'key': var_11})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'key': var_15})
    assert var_34 is True



# Parsed testcases at query #27
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{"key": "value"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = '{"key": "value"}'
    var_10 = 0
    var_11 = 15
    var_12 = {var_4: var_8}
    var_13 = [var_12, var_10, var_11, var_9]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_12)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_10)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_11)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_9)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key': var_4})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key': var_8})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = 4
    var_3 = '{"key1": "value1", "key2": "value2"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 8
    var_7 = 15
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 18
    var_11 = 21
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'value2'
    var_14 = 25
    var_15 = 32
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = '{"key1": "value1", "key2": "value2"}'
    var_18 = 0
    var_19 = 34
    var_20 = {var_4: var_8, var_12: var_16}
    var_21 = [var_20, var_18, var_19, var_17]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_20)
    assert var_25 is True
    var_26 = var_23._start_index
    var_27 = bool(var_23._start_index == var_18)
    assert var_27 is True
    var_28 = var_23._end_index
    var_29 = bool(var_23._end_index == var_19)
    assert var_29 is True
    var_30 = var_23._content
    var_31 = bool(var_23._content == var_17)
    assert var_31 is True
    var_32 = var_23._child_keys
    var_33 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'nested_key'
    var_1 = 10
    var_2 = 19
    var_3 = '{"key": {"nested_key": "nested_value"}}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'nested_value'
    var_6 = 23
    var_7 = 35
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 8
    var_11 = 36
    var_12 = [var_9, var_10, var_11, var_3]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = 'key'
    var_16 = 1
    var_17 = 3
    var_18 = module_0.Token(var_15, var_16, var_17, var_3)
    var_19 = '{"key": {"nested_key": "nested_value"}}'
    var_20 = 0
    var_21 = 38
    var_22 = {var_18: var_14}
    var_23 = [var_22, var_20, var_21, var_19]
    var_24 = {}
    var_25 = module_0.DictToken(*var_23, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == var_22)
    assert var_27 is True
    var_28 = var_25._start_index
    var_29 = bool(var_25._start_index == var_20)
    assert var_29 is True
    var_30 = var_25._end_index
    var_31 = bool(var_25._end_index == var_21)
    assert var_31 is True
    var_32 = var_25._content
    var_33 = bool(var_25._content == var_19)
    assert var_33 is True
    var_34 = var_25._child_keys
    var_35 = bool(var_25._child_keys == {'key': var_18})
    assert var_35 is True
    var_36 = var_25._child_tokens
    var_37 = bool(var_25._child_tokens == {'key': var_14})
    assert var_37 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{"key": "value1", "key": "value2"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'value2'
    var_10 = 20
    var_11 = 26
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = '{"key": "value1", "key": "value2"}'
    var_14 = 0
    var_15 = 28
    var_16 = {var_4: var_8}
    var_17 = [var_16, var_14, var_15, var_13]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == var_16)
    assert var_21 is True
    var_22 = var_19._start_index
    var_23 = bool(var_19._start_index == var_14)
    assert var_23 is True
    var_24 = var_19._end_index
    var_25 = bool(var_19._end_index == var_15)
    assert var_25 is True
    var_26 = var_19._content
    var_27 = bool(var_19._content == var_13)
    assert var_27 is True
    var_28 = var_19._child_keys
    var_29 = bool(var_19._child_keys == {'key': var_4})
    assert var_29 is True
    var_30 = var_19._child_tokens
    var_31 = bool(var_19._child_tokens == {'key': var_8})
    assert var_31 is True



# Parsed testcases at query #29
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



# Parsed testcases at query #30
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 1
    var_11 = var_6._content
    assert var_11 == '{}'
    var_12 = var_6._child_keys
    var_13 = bool(var_6._child_keys == {})
    assert var_13 is True
    var_14 = var_6._child_tokens
    var_15 = bool(var_6._child_tokens == {})
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 15
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == {var_6: var_10})
    assert var_16 is True
    var_17 = var_14._start_index
    assert var_17 == 0
    var_18 = var_14._end_index
    assert var_18 == 15
    var_19 = var_14._content
    assert var_19 == '{"key": "value"}'
    var_20 = var_14._child_keys
    var_21 = bool(var_14._child_keys == {'key': var_6})
    assert var_21 is True
    var_22 = var_14._child_tokens
    var_23 = bool(var_14._child_tokens == {'key': var_10})
    assert var_23 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 16
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = 15
    var_15 = module_0.Token(var_5, var_13, var_14, var_0)
    var_16 = {var_6: var_8, var_12: var_15}
    var_17 = [var_16, var_1, var_2, var_0]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == {var_6: var_8, var_12: var_15})
    assert var_21 is True
    var_22 = var_19._start_index
    assert var_22 == 0
    var_23 = var_19._end_index
    assert var_23 == 16
    var_24 = var_19._content
    assert var_24 == '{"a": 1, "b": 2}'
    var_25 = var_19._child_keys
    var_26 = bool(var_19._child_keys == {'a': var_6, 'b': var_12})
    assert var_26 is True
    var_27 = var_19._child_tokens
    var_28 = bool(var_19._child_tokens == {'a': var_8, 'b': var_15})
    assert var_28 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = 25
    var_3 = 'outer'
    var_4 = 1
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 11
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 42
    var_12 = 20
    var_13 = 22
    var_14 = module_0.Token(var_11, var_12, var_13, var_0)
    var_15 = {var_10: var_14}
    var_16 = 10
    var_17 = 23
    var_18 = [var_15, var_16, var_17, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = {var_6: var_20}
    var_22 = [var_21, var_1, var_2, var_0]
    var_23 = {}
    var_24 = module_0.DictToken(*var_22, **var_23)
    var_25 = var_24._value
    var_26 = bool(var_24._value == {var_6: var_20})
    assert var_26 is True
    var_27 = var_24._start_index
    assert var_27 == 0
    var_28 = var_24._end_index
    assert var_28 == 25
    var_29 = var_24._content
    assert var_29 == '{"outer": {"inner": 42}}'
    var_30 = var_24._child_keys
    var_31 = bool(var_24._child_keys == {'outer': var_6})
    assert var_31 is True
    var_32 = var_24._child_tokens
    var_33 = bool(var_24._child_tokens == {'outer': var_20})
    assert var_33 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "{1: 'one', True: 'yes'}"
    var_1 = 0
    var_2 = 22
    var_3 = 1
    var_4 = module_0.Token(var_3, var_3, var_3, var_0)
    var_5 = 'one'
    var_6 = 5
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = True
    var_10 = 12
    var_11 = 15
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 'yes'
    var_14 = 19
    var_15 = 22
    var_16 = module_0.Token(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = [var_17, var_1, var_2, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == {var_4: var_8, var_12: var_16})
    assert var_22 is True
    var_23 = var_20._start_index
    assert var_23 == 0
    var_24 = var_20._end_index
    assert var_24 == 22
    var_25 = var_20._content
    assert var_25 == "{1: 'one', True: 'yes'}"
    var_26 = var_20._child_keys
    var_27 = bool(var_20._child_keys == {1: var_4, True: var_12})
    assert var_27 is True
    var_28 = var_20._child_tokens
    var_29 = bool(var_20._child_tokens == {1: var_8, True: var_16})
    assert var_29 is True



# Parsed testcases at query #2
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 10
    var_7 = var_4._end_index
    assert var_7 == 15
    var_8 = var_4._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 2
    var_6 = var_3._end_index
    assert var_6 == 6
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #3
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 14
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 13
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == var_11)
    assert var_16 is True
    var_17 = var_14._start_index
    var_18 = bool(var_14._start_index == var_1)
    assert var_18 is True
    var_19 = var_14._end_index
    var_20 = bool(var_14._end_index == var_2)
    assert var_20 is True
    var_21 = var_14._content
    var_22 = bool(var_14._content == var_0)
    assert var_22 is True
    var_23 = var_14._child_keys
    var_24 = bool(var_14._child_keys == {'key': var_6})
    assert var_24 is True
    var_25 = var_14._child_tokens
    var_26 = bool(var_14._child_tokens == {'key': var_10})
    assert var_26 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 1}}'
    var_1 = 0
    var_2 = 23
    var_3 = 'outer'
    var_4 = 1
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 11
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 19
    var_12 = module_0.Token(var_4, var_11, var_11, var_0)
    var_13 = {var_10: var_12}
    var_14 = 10
    var_15 = 21
    var_16 = [var_13, var_14, var_15, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = {var_6: var_18}
    var_20 = [var_19, var_1, var_2, var_0]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_19)
    assert var_24 is True
    var_25 = var_22._start_index
    var_26 = bool(var_22._start_index == var_1)
    assert var_26 is True
    var_27 = var_22._end_index
    var_28 = bool(var_22._end_index == var_2)
    assert var_28 is True
    var_29 = var_22._content
    var_30 = bool(var_22._content == var_0)
    assert var_30 is True
    var_31 = var_22._child_keys
    var_32 = bool(var_22._child_keys == {'outer': var_6})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'outer': var_18})
    assert var_34 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 14
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = module_0.Token(var_5, var_13, var_13, var_0)
    var_15 = {var_6: var_8, var_12: var_14}
    var_16 = [var_15, var_1, var_2, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_15)
    assert var_20 is True
    var_21 = var_18._start_index
    var_22 = bool(var_18._start_index == var_1)
    assert var_22 is True
    var_23 = var_18._end_index
    var_24 = bool(var_18._end_index == var_2)
    assert var_24 is True
    var_25 = var_18._content
    var_26 = bool(var_18._content == var_0)
    assert var_26 is True
    var_27 = var_18._child_keys
    var_28 = bool(var_18._child_keys == {'a': var_6, 'b': var_12})
    assert var_28 is True
    var_29 = var_18._child_tokens
    var_30 = bool(var_18._child_tokens == {'a': var_8, 'b': var_14})
    assert var_30 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = {}
    var_3 = [var_2, var_0, var_1]
    var_4 = {}
    var_5 = module_0.DictToken(*var_3, **var_4)
    var_6 = var_5._value
    var_7 = bool(var_5._value == var_2)
    assert var_7 is True
    var_8 = var_5._start_index
    var_9 = bool(var_5._start_index == var_0)
    assert var_9 is True
    var_10 = var_5._end_index
    var_11 = bool(var_5._end_index == var_1)
    assert var_11 is True
    var_12 = var_5._content
    assert var_12 == ''
    var_13 = var_5._child_keys
    var_14 = bool(var_5._child_keys == {})
    assert var_14 is True
    var_15 = var_5._child_tokens
    var_16 = bool(var_5._child_tokens == {})
    assert var_16 is True



# Parsed testcases at query #2
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1, key2: value2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 15
    var_11 = 18
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'value2'
    var_14 = 22
    var_15 = 27
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_17, var_20: var_1, var_21: var_15, var_22: var_3}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = bool(var_24._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_26 is True
    var_27 = var_24._child_tokens
    var_28 = bool(var_24._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_28 is True

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
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value1, key: value2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 14
    var_10 = 16
    var_11 = module_0.Token(var_0, var_9, var_10, var_3)
    var_12 = 'value2'
    var_13 = 19
    var_14 = 24
    var_15 = module_0.Token(var_12, var_13, var_14, var_3)
    var_16 = {var_4: var_8, var_11: var_15}
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_16, var_19: var_1, var_20: var_14, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'key': var_11})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'key': var_15})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 5
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {123: var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {123: var_8})
    assert var_20 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == 'test'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 10
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = var_4._start_index
    assert var_6 == 5
    var_7 = var_4._end_index
    assert var_7 == 10
    var_8 = var_4._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = var_2._value
    assert var_3 is None
    var_4 = var_2._start_index
    assert var_4 == 2
    var_5 = var_2._end_index
    assert var_5 == 2
    var_6 = var_2._content
    assert var_6 == ''



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 3
    var_7 = var_3._content
    assert var_7 == 'test'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = 10
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 is None
    var_6 = var_4._start_index
    assert var_6 == 5
    var_7 = var_4._end_index
    assert var_7 == 10
    var_8 = var_4._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 123
    var_5 = var_3._start_index
    assert var_5 == 2
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'sample'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = -5
    var_2 = -1
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == -5
    var_8 = var_4._end_index
    assert var_8 == -1
    var_9 = var_4._content
    assert var_9 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 100
    var_2 = 200
    var_3 = 'x'
    var_4 = 300
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 100
    var_10 = var_6._end_index
    assert var_10 == 200
    var_11 = var_6._content
    var_12 = bool(var_6._content == 'x' * 300)
    assert var_12 is True



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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 5
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True
    var_21 = var_16._value
    var_22 = bool(var_16._value == var_8)
    assert var_22 is True
    var_23 = var_16._start_index
    assert var_23 == 0
    var_24 = var_16._end_index
    assert var_24 == 7
    var_25 = var_16._content
    assert var_25 == '"key": 1'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = []
    var_5 = 'value'
    var_6 = 'start_index'
    var_7 = 'end_index'
    var_8 = 'content'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.DictToken(*var_4, **var_9)
    var_11 = var_10._child_keys
    var_12 = bool(var_10._child_keys == {})
    assert var_12 is True
    var_13 = var_10._child_tokens
    var_14 = bool(var_10._child_tokens == {})
    assert var_14 is True
    var_15 = var_10._value
    var_16 = bool(var_10._value == {})
    assert var_16 is True
    var_17 = var_10._start_index
    assert var_17 == 0
    var_18 = var_10._end_index
    assert var_18 == 1
    var_19 = var_10._content
    assert var_19 == '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 3
    var_3 = '{"a": 1, "b": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 'b'
    var_8 = 9
    var_9 = 11
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 14
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = {var_4: var_6, var_10: var_13}
    var_15 = 0
    var_16 = 15
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_14, var_19: var_15, var_20: var_16, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'a': var_4, 'b': var_10})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'a': var_6, 'b': var_13})
    assert var_27 is True
    var_28 = var_23._value
    var_29 = bool(var_23._value == var_14)
    assert var_29 is True
    var_30 = var_23._start_index
    assert var_30 == 0
    var_31 = var_23._end_index
    assert var_31 == 15
    var_32 = var_23._content
    assert var_32 == '{"a": 1, "b": 2}'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_eq_with_token_subclass. Retrieved 5/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4 == var_4
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 10
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 3
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = var_4 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #9
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'test'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 4
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = 6
    var_3 = 'abcdefg'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'cdefg'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3.string
    assert var_4 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = var_2.value
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = 10
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.start
    var_6 = var_5.line_no
    assert var_6 == 2
    var_7 = var_5.column_no
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 5

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 12
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.end
    var_6 = var_5.line_no
    assert var_6 == 3
    var_7 = var_5.column_no
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 12

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 0
    var_4 = [var_3]
    var_5 = var_2.lookup(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 0
    var_4 = [var_3]
    var_5 = var_2.lookup_key(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('hello')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 100
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 200
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = bool(var_4 == var_5)
    assert var_8 is True
    var_9 = bool(not var_4 == var_7)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = bool(not var_2 == 'not a token')
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'value'
    var_9 = 7
    var_10 = 11
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = [var_12, var_1, var_4, var_0]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_12)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_1)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_4)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_0)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key': var_7})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key': var_11})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'a'
    var_6 = module_0.Token(var_5, var_3, var_3, var_0)
    var_7 = 5
    var_8 = module_0.Token(var_3, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = module_0.Token(var_9, var_10, var_10, var_0)
    var_12 = 2
    var_13 = 13
    var_14 = module_0.Token(var_12, var_13, var_13, var_0)
    var_15 = {var_6: var_8, var_11: var_14}
    var_16 = [var_15, var_1, var_4, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_15)
    assert var_20 is True
    var_21 = var_18._start_index
    var_22 = bool(var_18._start_index == var_1)
    assert var_22 is True
    var_23 = var_18._end_index
    var_24 = bool(var_18._end_index == var_4)
    assert var_24 is True
    var_25 = var_18._content
    var_26 = bool(var_18._content == var_0)
    assert var_26 is True
    var_27 = var_18._child_keys
    var_28 = bool(var_18._child_keys == {'a': var_6, 'b': var_11})
    assert var_28 is True
    var_29 = var_18._child_tokens
    var_30 = bool(var_18._child_tokens == {'a': var_8, 'b': var_14})
    assert var_30 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'inner'
    var_6 = 11
    var_7 = 15
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = 42
    var_10 = 18
    var_11 = 19
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = 10
    var_15 = 20
    var_16 = [var_13, var_14, var_15, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = 'outer'
    var_20 = 5
    var_21 = module_0.Token(var_19, var_3, var_20, var_0)
    var_22 = {var_21: var_18}
    var_23 = [var_22, var_1, var_4, var_0]
    var_24 = {}
    var_25 = module_0.DictToken(*var_23, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == var_22)
    assert var_27 is True
    var_28 = var_25._start_index
    var_29 = bool(var_25._start_index == var_1)
    assert var_29 is True
    var_30 = var_25._end_index
    var_31 = bool(var_25._end_index == var_4)
    assert var_31 is True
    var_32 = var_25._content
    var_33 = bool(var_25._content == var_0)
    assert var_33 is True
    var_34 = var_25._child_keys
    var_35 = bool(var_25._child_keys == {'outer': var_21})
    assert var_35 is True
    var_36 = var_25._child_tokens
    var_37 = bool(var_25._child_tokens == {'outer': var_18})
    assert var_37 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'first'
    var_9 = 7
    var_10 = 11
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 15
    var_13 = 17
    var_14 = module_0.Token(var_5, var_12, var_13, var_0)
    var_15 = 'second'
    var_16 = 21
    var_17 = 26
    var_18 = module_0.Token(var_15, var_16, var_17, var_0)
    var_19 = {var_7: var_11, var_14: var_18}
    var_20 = [var_19, var_1, var_4, var_0]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_19)
    assert var_24 is True
    var_25 = var_22._start_index
    var_26 = bool(var_22._start_index == var_1)
    assert var_26 is True
    var_27 = var_22._end_index
    var_28 = bool(var_22._end_index == var_4)
    assert var_28 is True
    var_29 = var_22._content
    var_30 = bool(var_22._content == var_0)
    assert var_30 is True
    var_31 = var_22._child_keys
    var_32 = bool(var_22._child_keys == {'key': var_14})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'key': var_18})
    assert var_34 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 8
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #12
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_9)
    assert var_18 is True
    var_19 = var_16._start_index
    assert var_19 == 0
    var_20 = var_16._end_index
    assert var_20 == 9
    var_21 = var_16._content
    assert var_21 == 'key: value'
    var_22 = var_16._child_keys
    var_23 = bool(var_16._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_16._child_tokens
    var_25 = bool(var_16._child_tokens == {'key': var_8})
    assert var_25 is True



# Parsed testcases at query #13
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1, key2: value2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 15
    var_11 = 18
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'value2'
    var_14 = 22
    var_15 = 27
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_17, var_20: var_1, var_21: var_15, var_22: var_3}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = bool(var_24._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_26 is True
    var_27 = var_24._child_tokens
    var_28 = bool(var_24._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_28 is True

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
    var_10 = var_9._child_keys
    var_11 = bool(var_9._child_keys == {})
    assert var_11 is True
    var_12 = var_9._child_tokens
    var_13 = bool(var_9._child_tokens == {})
    assert var_13 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = ' key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 123
    var_6 = 6
    var_7 = 8
    var_8 = ' key: 123'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = {var_4: var_9}
    var_11 = 0
    var_12 = []
    var_13 = 'value'
    var_14 = 'start_index'
    var_15 = 'end_index'
    var_16 = 'content'
    var_17 = {var_13: var_10, var_14: var_11, var_15: var_7, var_16: var_8}
    var_18 = module_0.DictToken(*var_12, **var_17)
    var_19 = var_18._start_index
    assert var_19 == 0
    var_20 = var_18._end_index
    assert var_20 == 8
    var_21 = var_18._content
    assert var_21 == ' key: 123'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value1, key: value2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 13
    var_10 = 15
    var_11 = module_0.Token(var_0, var_9, var_10, var_3)
    var_12 = 'value2'
    var_13 = 18
    var_14 = 23
    var_15 = module_0.Token(var_12, var_13, var_14, var_3)
    var_16 = {var_4: var_8, var_11: var_15}
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_16, var_19: var_1, var_20: var_14, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'key': var_11})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'key': var_15})
    assert var_27 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": "value"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 6
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #15
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
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_8, var_12: var_1, var_13: var_9, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_7})
    assert var_20 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    var_10 = bool(var_6._start_index == var_1)
    assert var_10 is True
    var_11 = var_6._end_index
    var_12 = bool(var_6._end_index == var_2)
    assert var_12 is True
    var_13 = var_6._content
    var_14 = bool(var_6._content == var_0)
    assert var_14 is True
    var_15 = var_6._child_keys
    var_16 = bool(var_6._child_keys == {})
    assert var_16 is True
    var_17 = var_6._child_tokens
    var_18 = bool(var_6._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 15
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'value'
    var_8 = 8
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = [var_11, var_1, var_2, var_0]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == var_11)
    assert var_16 is True
    var_17 = var_14._start_index
    var_18 = bool(var_14._start_index == var_1)
    assert var_18 is True
    var_19 = var_14._end_index
    var_20 = bool(var_14._end_index == var_2)
    assert var_20 is True
    var_21 = var_14._content
    var_22 = bool(var_14._content == var_0)
    assert var_22 is True
    var_23 = var_14._child_keys
    var_24 = bool(var_14._child_keys == {'key': var_6})
    assert var_24 is True
    var_25 = var_14._child_tokens
    var_26 = bool(var_14._child_tokens == {'key': var_10})
    assert var_26 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 16
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 14
    var_14 = module_0.Token(var_5, var_13, var_13, var_0)
    var_15 = {var_6: var_8, var_12: var_14}
    var_16 = [var_15, var_1, var_2, var_0]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == var_15)
    assert var_20 is True
    var_21 = var_18._start_index
    var_22 = bool(var_18._start_index == var_1)
    assert var_22 is True
    var_23 = var_18._end_index
    var_24 = bool(var_18._end_index == var_2)
    assert var_24 is True
    var_25 = var_18._content
    var_26 = bool(var_18._content == var_0)
    assert var_26 is True
    var_27 = var_18._child_keys
    var_28 = bool(var_18._child_keys == {'a': var_6, 'b': var_12})
    assert var_28 is True
    var_29 = var_18._child_tokens
    var_30 = bool(var_18._child_tokens == {'a': var_8, 'b': var_14})
    assert var_30 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = 25
    var_3 = 'outer'
    var_4 = 1
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 11
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 42
    var_12 = 20
    var_13 = 21
    var_14 = module_0.Token(var_11, var_12, var_13, var_0)
    var_15 = {var_10: var_14}
    var_16 = 10
    var_17 = 22
    var_18 = [var_15, var_16, var_17, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = {var_6: var_20}
    var_22 = [var_21, var_1, var_2, var_0]
    var_23 = {}
    var_24 = module_0.DictToken(*var_22, **var_23)
    var_25 = var_24._value
    var_26 = bool(var_24._value == var_21)
    assert var_26 is True
    var_27 = var_24._start_index
    var_28 = bool(var_24._start_index == var_1)
    assert var_28 is True
    var_29 = var_24._end_index
    var_30 = bool(var_24._end_index == var_2)
    assert var_30 is True
    var_31 = var_24._content
    var_32 = bool(var_24._content == var_0)
    assert var_32 is True
    var_33 = var_24._child_keys
    var_34 = bool(var_24._child_keys == {'outer': var_6})
    assert var_34 is True
    var_35 = var_24._child_tokens
    var_36 = bool(var_24._child_tokens == {'outer': var_20})
    assert var_36 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = 32
    var_3 = 'key'
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'first'
    var_8 = 8
    var_9 = 13
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 17
    var_12 = 20
    var_13 = module_0.Token(var_3, var_11, var_12, var_0)
    var_14 = 'second'
    var_15 = 24
    var_16 = 31
    var_17 = module_0.Token(var_14, var_15, var_16, var_0)
    var_18 = {var_6: var_10, var_13: var_17}
    var_19 = [var_18, var_1, var_2, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = var_21._value
    var_23 = bool(var_21._value == var_18)
    assert var_23 is True
    var_24 = var_21._start_index
    var_25 = bool(var_21._start_index == var_1)
    assert var_25 is True
    var_26 = var_21._end_index
    var_27 = bool(var_21._end_index == var_2)
    assert var_27 is True
    var_28 = var_21._content
    var_29 = bool(var_21._content == var_0)
    assert var_29 is True
    var_30 = var_21._child_keys
    var_31 = bool(var_21._child_keys == {'key': var_13})
    assert var_31 is True
    var_32 = var_21._child_tokens
    var_33 = bool(var_21._child_tokens == {'key': var_17})
    assert var_33 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = '"key": "value"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = []
    var_11 = 'value'
    var_12 = 'start_index'
    var_13 = 'end_index'
    var_14 = 'content'
    var_15 = {var_11: var_9, var_12: var_1, var_13: var_7, var_14: var_3}
    var_16 = module_0.DictToken(*var_10, **var_15)
    var_17 = var_16._child_keys
    var_18 = bool(var_16._child_keys == {'key': var_4})
    assert var_18 is True
    var_19 = var_16._child_tokens
    var_20 = bool(var_16._child_tokens == {'key': var_8})
    assert var_20 is True



