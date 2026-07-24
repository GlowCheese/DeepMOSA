####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 'a'
    var_1 = 0
    var_2 = 'a: 1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = {var_3: var_6}
    var_8 = 'a: 1'
    var_9 = []
    var_10 = 'value'
    var_11 = 'start_index'
    var_12 = 'end_index'
    var_13 = 'content'
    var_14 = {var_10: var_7, var_11: var_1, var_12: var_5, var_13: var_8}
    var_15 = module_0.DictToken(*var_9, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_7)
    assert var_17 is True
    var_18 = var_15._start_index
    assert var_18 == 0
    var_19 = var_15._end_index
    assert var_19 == 3
    var_20 = var_15._content
    var_21 = bool(var_15._content == var_8)
    assert var_21 is True

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
    var_0 = 'x'
    var_1 = 0
    var_2 = 'x: 1, y: 2'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'y'
    var_8 = 6
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = 2
    var_11 = 9
    var_12 = module_0.Token(var_10, var_11, var_11, var_2)
    var_13 = {var_3: var_6, var_9: var_12}
    var_14 = 10
    var_15 = []
    var_16 = 'value'
    var_17 = 'start_index'
    var_18 = 'end_index'
    var_19 = 'content'
    var_20 = {var_16: var_13, var_17: var_1, var_18: var_14, var_19: var_2}
    var_21 = module_0.DictToken(*var_15, **var_20)
    var_22 = var_21._child_keys
    var_23 = bool(var_21._child_keys == {'x': var_3, 'y': var_9})
    assert var_23 is True
    var_24 = var_21._child_tokens
    var_25 = bool(var_21._child_tokens == {'x': var_6, 'y': var_12})
    assert var_25 is True



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



# Parsed testcases at query #3
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
    var_6 = var_5.line
    assert var_6 == 2
    var_7 = var_5.column
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 5

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.end
    var_6 = var_5.line
    assert var_6 == 1
    var_7 = var_5.column
    assert var_7 == 5
    var_8 = var_5.index
    assert var_8 == 4

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
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('est')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 20
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = var_4 == var_5
    assert var_6 is False



# Parsed testcases at query #4
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
    assert var_6 is False



# Parsed testcases at query #5
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
    var_11 = var_10._value
    var_12 = bool(var_10._value == {})
    assert var_12 is True
    var_13 = var_10._start_index
    assert var_13 == 0
    var_14 = var_10._end_index
    assert var_14 == 1
    var_15 = var_10._content
    assert var_15 == '{}'
    var_16 = var_10._child_keys
    var_17 = bool(var_10._child_keys == {})
    assert var_17 is True
    var_18 = var_10._child_tokens
    var_19 = bool(var_10._child_tokens == {})
    assert var_19 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 3
    var_3 = '{"a": 1, "b": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 7
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 'b'
    var_8 = 11
    var_9 = 13
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 17
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = {var_4: var_6, var_10: var_13}
    var_15 = 0
    var_16 = 18
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_14, var_19: var_15, var_20: var_16, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_14)
    assert var_25 is True
    var_26 = var_23._start_index
    assert var_26 == 0
    var_27 = var_23._end_index
    assert var_27 == 18
    var_28 = var_23._content
    assert var_28 == '{"a": 1, "b": 2}'
    var_29 = var_23._child_keys
    var_30 = bool(var_23._child_keys == {'a': var_4, 'b': var_10})
    assert var_30 is True
    var_31 = var_23._child_tokens
    var_32 = bool(var_23._child_tokens == {'a': var_6, 'b': var_13})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123: "val"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val'
    var_6 = 6
    var_7 = 10
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 5
    var_2 = 6
    var_3 = '  {}'
    var_4 = []
    var_5 = 'value'
    var_6 = 'start_index'
    var_7 = 'end_index'
    var_8 = 'content'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.DictToken(*var_4, **var_9)
    var_11 = var_10._start_index
    assert var_11 == 5
    var_12 = var_10._end_index
    assert var_12 == 6



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = var_3 == var_4
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test2'
    var_5 = module_0.Token(var_4, var_1, var_2, var_4)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 1
    var_5 = module_0.Token(var_0, var_4, var_2, var_0)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 2
    var_5 = module_0.Token(var_0, var_1, var_4, var_0)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'content1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'content2'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_4 == var_6
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_eq_false_when_values_differ. Retrieved 4/10 statements.
# Partially parsed test_eq_false_when_start_indices_differ. Retrieved 3/9 statements.
# Partially parsed test_eq_false_when_end_indices_differ. Retrieved 3/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = 2

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_eq_false_when_values_differ. Retrieved 4/14 statements.
# Partially parsed test_eq_false_when_start_indices_differ. Retrieved 3/13 statements.
# Partially parsed test_eq_false_when_end_indices_differ. Retrieved 3/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = 2

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = var_3 == var_4
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test2'
    var_5 = module_0.Token(var_4, var_1, var_2, var_4)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 1
    var_5 = module_0.Token(var_0, var_4, var_2, var_0)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 2
    var_5 = module_0.Token(var_0, var_1, var_4, var_0)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3 == var_3
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'value2'
    var_5 = module_0.Token(var_4, var_1, var_1, var_2)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'same'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'same'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False



# Parsed testcases at query #12
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
    var_0 = '{"a": {"b": 1}}'
    var_1 = 0
    var_2 = 15
    var_3 = 'b'
    var_4 = 7
    var_5 = 8
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 1
    var_8 = 11
    var_9 = 12
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = {var_6: var_10}
    var_12 = 5
    var_13 = 13
    var_14 = [var_11, var_12, var_13, var_0]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = 'a'
    var_18 = 2
    var_19 = module_0.Token(var_17, var_7, var_18, var_0)
    var_20 = {var_19: var_16}
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
    var_33 = bool(var_23._child_keys == {'a': var_19})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'a': var_16})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"x": 10, "y": 20}'
    var_1 = 0
    var_2 = 19
    var_3 = 'x'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 10
    var_8 = 6
    var_9 = 8
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 'y'
    var_12 = 11
    var_13 = 12
    var_14 = module_0.Token(var_11, var_12, var_13, var_0)
    var_15 = 20
    var_16 = 16
    var_17 = 18
    var_18 = module_0.Token(var_15, var_16, var_17, var_0)
    var_19 = {var_6: var_10, var_14: var_18}
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
    var_32 = bool(var_22._child_keys == {'x': var_6, 'y': var_14})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'x': var_10, 'y': var_18})
    assert var_34 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 0
    var_3 = {}
    var_4 = [var_3, var_1, var_2]
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



# Parsed testcases at query #13
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
    var_5 = 7
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 12
    var_9 = 18
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 3
    var_12 = 21
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
    var_2 = 31
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



# Parsed testcases at query #14
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
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = 7
    var_3 = []
    var_4 = module_0.ListToken(var_3, var_1, var_2, var_0)
    var_5 = var_4._value
    var_6 = bool(var_4._value == var_3)
    assert var_6 is True
    var_7 = var_4._start_index
    var_8 = bool(var_4._start_index == var_1)
    assert var_8 is True
    var_9 = var_4._end_index
    var_10 = bool(var_4._end_index == var_2)
    assert var_10 is True
    var_11 = var_4._content
    var_12 = bool(var_4._content == var_0)
    assert var_12 is True



# Parsed testcases at query #17
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
    var_5 = 'outer'
    var_6 = 5
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'inner'
    var_9 = 11
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 42
    var_13 = 18
    var_14 = 19
    var_15 = module_0.Token(var_12, var_13, var_14, var_0)
    var_16 = {var_11: var_15}
    var_17 = 9
    var_18 = 20
    var_19 = [var_16, var_17, var_18, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = {var_7: var_21}
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
    var_35 = bool(var_25._child_keys == {'outer': var_7})
    assert var_35 is True
    var_36 = var_25._child_tokens
    var_37 = bool(var_25._child_tokens == {'outer': var_21})
    assert var_37 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value1", "key": "value2"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'value1'
    var_9 = 7
    var_10 = 12
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 16
    var_13 = 18
    var_14 = module_0.Token(var_5, var_12, var_13, var_0)
    var_15 = 'value2'
    var_16 = 22
    var_17 = 27
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



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = 6
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



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._content
    var_5 = 'wrong'
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._content
    var_6 = ''
    var_7 = var_5 == var_6
    var_8 = bool(not var_7)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = 5
    var_5 = '[1, 2]'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._content
    var_8 = '[1,2]'
    var_9 = var_7 == var_8
    var_10 = bool(not var_9)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = ' a '
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._content
    var_5 = var_4 == var_0
    var_6 = bool(not var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._content
    var_6 = var_5 is var_0
    var_7 = bool(not var_6)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 9
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3._value
    assert var_4 == 'test_value'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 9
    var_7 = var_3._content
    assert var_7 == 'test_value'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'value'
    var_5 = var_3._start_index
    assert var_5 == 1
    var_6 = var_3._end_index
    assert var_6 == 5
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'child'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = [var_3]
    var_5 = module_0.ListToken(var_4, var_1, var_2, var_0)
    var_6 = var_5._value
    var_7 = bool(var_5._value == [var_3])
    assert var_7 is True
    var_8 = var_5._start_index
    assert var_8 == 0
    var_9 = var_5._end_index
    assert var_9 == 4
    var_10 = var_5._content
    assert var_10 == 'child'



# Parsed testcases at query #21
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
    var_9 = {var_4: var_8}
    var_10 = '{key: value}'
    var_11 = 0
    var_12 = 11
    var_13 = [var_9, var_11, var_12, var_10]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_9)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_11)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_12)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_10)
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
    var_3 = '{key1: val1, key2: val2}'
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
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = '{key1: val1, key2: val2}'
    var_19 = 0
    var_20 = 23
    var_21 = [var_17, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_17)
    assert var_25 is True
    var_26 = var_23._start_index
    var_27 = bool(var_23._start_index == var_19)
    assert var_27 is True
    var_28 = var_23._end_index
    var_29 = bool(var_23._end_index == var_20)
    assert var_29 is True
    var_30 = var_23._content
    var_31 = bool(var_23._content == var_18)
    assert var_31 is True
    var_32 = var_23._child_keys
    var_33 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{key: val1, key: val2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'val1'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'val2'
    var_10 = 16
    var_11 = 19
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = {var_4: var_8}
    var_14 = '{key: val1, key: val2}'
    var_15 = 0
    var_16 = 21
    var_17 = [var_13, var_15, var_16, var_14]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == var_13)
    assert var_21 is True
    var_22 = var_19._start_index
    var_23 = bool(var_19._start_index == var_15)
    assert var_23 is True
    var_24 = var_19._end_index
    var_25 = bool(var_19._end_index == var_16)
    assert var_25 is True
    var_26 = var_19._content
    var_27 = bool(var_19._content == var_14)
    assert var_27 is True
    var_28 = var_19._child_keys
    var_29 = bool(var_19._child_keys == {'key': var_4})
    assert var_29 is True
    var_30 = var_19._child_tokens
    var_31 = bool(var_19._child_tokens == {'key': var_8})
    assert var_31 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{a: 1}'
    var_1 = 0
    var_2 = 5
    var_3 = 'a'
    var_4 = 1
    var_5 = module_0.Token(var_3, var_4, var_4, var_0)
    var_6 = 4
    var_7 = module_0.Token(var_4, var_6, var_6, var_0)
    var_8 = {var_5: var_7}
    var_9 = [var_8, var_1, var_2, var_0]
    var_10 = {}
    var_11 = module_0.DictToken(*var_9, **var_10)
    var_12 = var_11._value
    var_13 = bool(var_11._value == var_8)
    assert var_13 is True
    var_14 = var_11._start_index
    var_15 = bool(var_11._start_index == var_1)
    assert var_15 is True
    var_16 = var_11._end_index
    var_17 = bool(var_11._end_index == var_2)
    assert var_17 is True
    var_18 = var_11._content
    var_19 = bool(var_11._content == var_0)
    assert var_19 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_eq_false_when_values_differ. Retrieved 4/13 statements.
# Partially parsed test_eq_false_when_start_indices_differ. Retrieved 5/11 statements.
# Partially parsed test_eq_false_when_end_indices_differ. Retrieved 5/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 5
    var_3 = 'content'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = 1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = 6



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
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
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 3
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'world'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True



# Parsed testcases at query #24
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
    var_21 = var_16._value
    var_22 = bool(var_16._value == var_9)
    assert var_22 is True
    var_23 = var_16._start_index
    assert var_23 == 0
    var_24 = var_16._end_index
    assert var_24 == 9
    var_25 = var_16._content
    assert var_25 == 'key: value'

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
    var_14 = var_9._value
    var_15 = bool(var_9._value == {})
    assert var_15 is True
    var_16 = var_9._start_index
    assert var_16 == 0
    var_17 = var_9._end_index
    assert var_17 == 0
    var_18 = var_9._content
    assert var_18 == '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 7
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = 'key2'
    var_9 = 11
    var_10 = 14
    var_11 = module_0.Token(var_8, var_9, var_10, var_3)
    var_12 = 2
    var_13 = 18
    var_14 = module_0.Token(var_12, var_13, var_13, var_3)
    var_15 = {var_4: var_7, var_11: var_14}
    var_16 = 20
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_15, var_19: var_1, var_20: var_16, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_11})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'key1': var_7, 'key2': var_14})
    assert var_27 is True
    var_28 = var_23._value
    var_29 = bool(var_23._value == var_15)
    assert var_29 is True
    var_30 = var_23._start_index
    assert var_30 == 0
    var_31 = var_23._end_index
    assert var_31 == 20
    var_32 = var_23._content
    assert var_32 == '{"key1": 1, "key2": 2}'



# Parsed testcases at query #25
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
    var_0 = 'key1'
    var_1 = 1
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = '{key1: value1}'
    var_11 = 0
    var_12 = 14
    var_13 = [var_9, var_11, var_12, var_10]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_9)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_11)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_12)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_10)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key1': var_4})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key1': var_8})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = 3
    var_3 = '{key1: value1, key2: value2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 15
    var_11 = 17
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 'value2'
    var_14 = 21
    var_15 = 26
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = '{key1: value1, key2: value2}'
    var_19 = 0
    var_20 = 28
    var_21 = [var_17, var_19, var_20, var_18]
    var_22 = {}
    var_23 = module_0.DictToken(*var_21, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_17)
    assert var_25 is True
    var_26 = var_23._start_index
    var_27 = bool(var_23._start_index == var_19)
    assert var_27 is True
    var_28 = var_23._end_index
    var_29 = bool(var_23._end_index == var_20)
    assert var_29 is True
    var_30 = var_23._content
    var_31 = bool(var_23._content == var_18)
    assert var_31 is True
    var_32 = var_23._child_keys
    var_33 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_12})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'key1': var_8, 'key2': var_16})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{key: value1, key: value2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 15
    var_10 = 17
    var_11 = module_0.Token(var_0, var_9, var_10, var_3)
    var_12 = 'value2'
    var_13 = 21
    var_14 = 26
    var_15 = module_0.Token(var_12, var_13, var_14, var_3)
    var_16 = {var_4: var_8, var_11: var_15}
    var_17 = '{key: value1, key: value2}'
    var_18 = 0
    var_19 = 28
    var_20 = [var_16, var_18, var_19, var_17]
    var_21 = {}
    var_22 = module_0.DictToken(*var_20, **var_21)
    var_23 = var_22._value
    var_24 = bool(var_22._value == var_16)
    assert var_24 is True
    var_25 = var_22._start_index
    var_26 = bool(var_22._start_index == var_18)
    assert var_26 is True
    var_27 = var_22._end_index
    var_28 = bool(var_22._end_index == var_19)
    assert var_28 is True
    var_29 = var_22._content
    var_30 = bool(var_22._content == var_17)
    assert var_30 is True
    var_31 = var_22._child_keys
    var_32 = bool(var_22._child_keys == {'key': var_11})
    assert var_32 is True
    var_33 = var_22._child_tokens
    var_34 = bool(var_22._child_tokens == {'key': var_15})
    assert var_34 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 1
    var_2 = 3
    var_3 = '{123: value}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = '{123: value}'
    var_11 = 0
    var_12 = 12
    var_13 = [var_9, var_11, var_12, var_10]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_9)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_11)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_12)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_10)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {123: var_4})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {123: var_8})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = ''
    var_11 = 0
    var_12 = 0
    var_13 = [var_9, var_11, var_12, var_10]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_9)
    assert var_17 is True
    var_18 = var_15._start_index
    var_19 = bool(var_15._start_index == var_11)
    assert var_19 is True
    var_20 = var_15._end_index
    var_21 = bool(var_15._end_index == var_12)
    assert var_21 is True
    var_22 = var_15._content
    var_23 = bool(var_15._content == var_10)
    assert var_23 is True
    var_24 = var_15._child_keys
    var_25 = bool(var_15._child_keys == {'key': var_4})
    assert var_25 is True
    var_26 = var_15._child_tokens
    var_27 = bool(var_15._child_tokens == {'key': var_8})
    assert var_27 is True



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
    var_5 = 3
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 7
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 10
    var_11 = 12
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 2
    var_14 = 16
    var_15 = module_0.Token(var_13, var_14, var_14, var_0)
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
    var_0 = '{"outer": {"inner": 3}}'
    var_1 = 0
    var_2 = 24
    var_3 = 'outer'
    var_4 = 1
    var_5 = 7
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 12
    var_9 = 17
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 3
    var_12 = 21
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
    var_0 = '{"key": "first", "key": "second"}'
    var_1 = 0
    var_2 = 31
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
    var_23 = bool(var_21._value == {var_6: var_10, var_13: var_17})
    assert var_23 is True
    var_24 = var_21._start_index
    assert var_24 == 0
    var_25 = var_21._end_index
    assert var_25 == 31
    var_26 = var_21._content
    assert var_26 == '{"key": "first", "key": "second"}'
    var_27 = var_21._child_keys
    var_28 = bool(var_21._child_keys == {'key': var_13})
    assert var_28 is True
    var_29 = var_21._child_tokens
    var_30 = bool(var_21._child_tokens == {'key': var_17})
    assert var_30 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
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
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 3
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'world'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True



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
    var_3 = '"key": "value"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = '"key": "value"'
    var_10 = 0
    var_11 = 13
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
    var_2 = 5
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 9
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 'key2'
    var_8 = 13
    var_9 = 17
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 21
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = '{"key1": 1, "key2": 2}'
    var_15 = 0
    var_16 = 22
    var_17 = {var_4: var_6, var_10: var_13}
    var_18 = [var_17, var_15, var_16, var_14]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_17)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_15)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_16)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_14)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {'key1': var_4, 'key2': var_10})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {'key1': var_6, 'key2': var_13})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{"key": 1, "key": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 7
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 2
    var_8 = 17
    var_9 = module_0.Token(var_7, var_8, var_8, var_3)
    var_10 = '{"key": 1, "key": 2}'
    var_11 = 0
    var_12 = 20
    var_13 = {var_4: var_6}
    var_14 = [var_13, var_11, var_12, var_10]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_13)
    assert var_18 is True
    var_19 = var_16._start_index
    var_20 = bool(var_16._start_index == var_11)
    assert var_20 is True
    var_21 = var_16._end_index
    var_22 = bool(var_16._end_index == var_12)
    assert var_22 is True
    var_23 = var_16._content
    var_24 = bool(var_16._content == var_10)
    assert var_24 is True
    var_25 = var_16._child_keys
    var_26 = bool(var_16._child_keys == {'key': var_4})
    assert var_26 is True
    var_27 = var_16._child_tokens
    var_28 = bool(var_16._child_tokens == {'key': var_6})
    assert var_28 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 0
    var_10 = 13
    var_11 = {var_4: var_8}
    var_12 = [var_11, var_9, var_10]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == var_11)
    assert var_16 is True
    var_17 = var_14._start_index
    var_18 = bool(var_14._start_index == var_9)
    assert var_18 is True
    var_19 = var_14._end_index
    var_20 = bool(var_14._end_index == var_10)
    assert var_20 is True
    var_21 = var_14._content
    assert var_21 == ''
    var_22 = var_14._child_keys
    var_23 = bool(var_14._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_14._child_tokens
    var_25 = bool(var_14._child_tokens == {'key': var_8})
    assert var_25 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._content
    assert var_4 == ''



# Parsed testcases at query #30
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
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'outer'
    var_6 = 6
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'inner'
    var_9 = 11
    var_10 = 16
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 42
    var_13 = 18
    var_14 = 19
    var_15 = module_0.Token(var_12, var_13, var_14, var_0)
    var_16 = {var_11: var_15}
    var_17 = 9
    var_18 = 20
    var_19 = [var_16, var_17, var_18, var_0]
    var_20 = {}
    var_21 = module_0.DictToken(*var_19, **var_20)
    var_22 = {var_7: var_21}
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
    var_35 = bool(var_25._child_keys == {'outer': var_7})
    assert var_35 is True
    var_36 = var_25._child_tokens
    var_37 = bool(var_25._child_tokens == {'outer': var_21})
    assert var_37 is True

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
    var_11 = 9
    var_12 = 10
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 13
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_6 = var_5.line
    assert var_6 == 2
    var_7 = var_5.column
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 5

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 8
    var_3 = 'line1\nline2'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4.end
    var_6 = var_5.line
    assert var_6 == 2
    var_7 = var_5.column
    assert var_7 == 3
    var_8 = var_5.index
    assert var_8 == 8

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
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'xyz'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'xyz'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 20
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = 'xyz'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True



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
    var_0 = 'a'
    var_1 = 0
    var_2 = 'a: 1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = {var_3: var_6}
    var_8 = []
    var_9 = 'value'
    var_10 = 'start_index'
    var_11 = 'end_index'
    var_12 = 'content'
    var_13 = {var_9: var_7, var_10: var_1, var_11: var_5, var_12: var_2}
    var_14 = module_0.DictToken(*var_8, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == var_7)
    assert var_16 is True
    var_17 = var_14._start_index
    assert var_17 == 0
    var_18 = var_14._end_index
    assert var_18 == 3
    var_19 = var_14._content
    assert var_19 == 'a: 1'

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
    var_0 = 'x'
    var_1 = 0
    var_2 = 'x: 1, y: 2'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'y'
    var_8 = 6
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = 2
    var_11 = 9
    var_12 = module_0.Token(var_10, var_11, var_11, var_2)
    var_13 = {var_3: var_6, var_9: var_12}
    var_14 = 11
    var_15 = []
    var_16 = 'value'
    var_17 = 'start_index'
    var_18 = 'end_index'
    var_19 = 'content'
    var_20 = {var_16: var_13, var_17: var_1, var_18: var_14, var_19: var_2}
    var_21 = module_0.DictToken(*var_15, **var_20)
    var_22 = var_21._child_keys
    var_23 = bool(var_21._child_keys == {'x': var_3, 'y': var_9})
    assert var_23 is True
    var_24 = var_21._child_tokens
    var_25 = bool(var_21._child_tokens == {'x': var_6, 'y': var_12})
    assert var_25 is True



# Parsed testcases at query #4
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
    var_21 = var_16._value
    var_22 = bool(var_16._value == var_9)
    assert var_22 is True
    var_23 = var_16._start_index
    assert var_23 == 0
    var_24 = var_16._end_index
    assert var_24 == 9
    var_25 = var_16._content
    assert var_25 == 'key: value'

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
    var_14 = var_9._value
    var_15 = bool(var_9._value == {})
    assert var_15 is True
    var_16 = var_9._start_index
    assert var_16 == 0
    var_17 = var_9._end_index
    assert var_17 == 0
    var_18 = var_9._content
    assert var_18 == '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 7
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = 'key2'
    var_9 = 11
    var_10 = 14
    var_11 = module_0.Token(var_8, var_9, var_10, var_3)
    var_12 = 2
    var_13 = 18
    var_14 = module_0.Token(var_12, var_13, var_13, var_3)
    var_15 = {var_4: var_7, var_11: var_14}
    var_16 = 20
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_15, var_19: var_1, var_20: var_16, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._child_keys
    var_25 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_11})
    assert var_25 is True
    var_26 = var_23._child_tokens
    var_27 = bool(var_23._child_tokens == {'key1': var_7, 'key2': var_14})
    assert var_27 is True
    var_28 = var_23._value
    var_29 = bool(var_23._value == var_15)
    assert var_29 is True
    var_30 = var_23._start_index
    assert var_30 == 0
    var_31 = var_23._end_index
    assert var_31 == 20
    var_32 = var_23._content
    assert var_32 == '{"key1": 1, "key2": 2}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = '{"k": "v"}'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'v'
    var_5 = 5
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = {var_3: var_6}
    var_8 = 8
    var_9 = []
    var_10 = 'value'
    var_11 = 'start_index'
    var_12 = 'end_index'
    var_13 = 'content'
    var_14 = {var_10: var_7, var_11: var_1, var_12: var_8, var_13: var_2}
    var_15 = module_0.DictToken(*var_9, **var_14)
    var_16 = var_15._value
    var_17 = bool(var_15._value == var_7)
    assert var_17 is True
    var_18 = var_15._start_index
    assert var_18 == 0
    var_19 = var_15._end_index
    assert var_19 == 8
    var_20 = var_15._content
    assert var_20 == '{"k": "v"}'



# Parsed testcases at query #5
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
    var_0 = 'a'
    var_1 = 5
    var_2 = 'xyz a uvw'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    var_5 = var_3._end_index
    var_6 = var_4 == var_5
    var_7 = bool(not var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 10
    var_2 = 2
    var_3 = 'some content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index
    var_6 = var_4._end_index
    var_7 = var_5 == var_6
    var_8 = bool(not var_7)
    assert var_8 is True



# Parsed testcases at query #6
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
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 15
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = module_0.Token(var_4, var_7, var_7, var_0)
    var_9 = 'b'
    var_10 = 10
    var_11 = 11
    var_12 = module_0.Token(var_9, var_10, var_11, var_0)
    var_13 = 15
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
    var_2 = 24
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
    var_2 = 30
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
    var_16 = 29
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
    var_12 = 19
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
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 15
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = 7
    var_9 = module_0.Token(var_4, var_7, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 11
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 14
    var_15 = 15
    var_16 = module_0.Token(var_5, var_14, var_15, var_0)
    var_17 = {var_6: var_9, var_13: var_16}
    var_18 = [var_17, var_1, var_2, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == {var_6: var_9, var_13: var_16})
    assert var_22 is True
    var_23 = var_20._start_index
    assert var_23 == 0
    var_24 = var_20._end_index
    assert var_24 == 15
    var_25 = var_20._content
    assert var_25 == '{"a": 1, "b": 2}'
    var_26 = var_20._child_keys
    var_27 = bool(var_20._child_keys == {'a': var_6, 'b': var_13})
    assert var_27 is True
    var_28 = var_20._child_tokens
    var_29 = bool(var_20._child_tokens == {'a': var_9, 'b': var_16})
    assert var_29 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{123: "number"}'
    var_1 = 0
    var_2 = 14
    var_3 = 123
    var_4 = 1
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'number'
    var_8 = 7
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
    assert var_18 == 14
    var_19 = var_14._content
    assert var_19 == '{123: "number"}'
    var_20 = var_14._child_keys
    var_21 = bool(var_14._child_keys == {123: var_6})
    assert var_21 is True
    var_22 = var_14._child_tokens
    var_23 = bool(var_14._child_tokens == {123: var_10})
    assert var_23 is True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_3 = '{"key": 1}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = '{"key": 1}'
    var_8 = 0
    var_9 = 9
    var_10 = {var_4: var_6}
    var_11 = [var_10, var_8, var_9, var_7]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = var_13._value
    var_15 = bool(var_13._value == var_10)
    assert var_15 is True
    var_16 = var_13._start_index
    var_17 = bool(var_13._start_index == var_8)
    assert var_17 is True
    var_18 = var_13._end_index
    var_19 = bool(var_13._end_index == var_9)
    assert var_19 is True
    var_20 = var_13._content
    var_21 = bool(var_13._content == var_7)
    assert var_21 is True
    var_22 = var_13._child_keys
    var_23 = bool(var_13._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_13._child_tokens
    var_25 = bool(var_13._child_tokens == {'key': var_6})
    assert var_25 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = 4
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 7
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 'key2'
    var_8 = 11
    var_9 = 14
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 17
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = '{"key1": 1, "key2": 2}'
    var_15 = 0
    var_16 = 22
    var_17 = {var_4: var_6, var_10: var_13}
    var_18 = [var_17, var_15, var_16, var_14]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_17)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_15)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_16)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_14)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {'key1': var_4, 'key2': var_10})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {'key1': var_6, 'key2': var_13})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 8
    var_2 = 12
    var_3 = '{"outer": {"inner": 1}}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 15
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = {var_4: var_7}
    var_9 = 7
    var_10 = 17
    var_11 = [var_8, var_9, var_10, var_3]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = 'outer'
    var_15 = 5
    var_16 = module_0.Token(var_14, var_5, var_15, var_3)
    var_17 = '{"outer": {"inner": 1}}'
    var_18 = 0
    var_19 = 22
    var_20 = {var_16: var_13}
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
    var_33 = bool(var_23._child_keys == {'outer': var_16})
    assert var_33 is True
    var_34 = var_23._child_tokens
    var_35 = bool(var_23._child_tokens == {'outer': var_13})
    assert var_35 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{"key": 1, "key": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 2
    var_8 = 15
    var_9 = module_0.Token(var_7, var_8, var_8, var_3)
    var_10 = '{"key": 1, "key": 2}'
    var_11 = 0
    var_12 = 20
    var_13 = {var_4: var_6, var_4: var_9}
    var_14 = [var_13, var_11, var_12, var_10]
    var_15 = {}
    var_16 = module_0.DictToken(*var_14, **var_15)
    var_17 = var_16._value
    var_18 = bool(var_16._value == var_13)
    assert var_18 is True
    var_19 = var_16._start_index
    var_20 = bool(var_16._start_index == var_11)
    assert var_20 is True
    var_21 = var_16._end_index
    var_22 = bool(var_16._end_index == var_12)
    assert var_22 is True
    var_23 = var_16._content
    var_24 = bool(var_16._content == var_10)
    assert var_24 is True
    var_25 = var_16._child_keys
    var_26 = bool(var_16._child_keys == {'key': var_4})
    assert var_26 is True
    var_27 = var_16._child_tokens
    var_28 = bool(var_16._child_tokens == {'key': var_9})
    assert var_28 is True



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
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = 25
    var_3 = 'outer'
    var_4 = 1
    var_5 = 7
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 12
    var_9 = 18
    var_10 = module_0.Token(var_7, var_8, var_9, var_0)
    var_11 = 42
    var_12 = 21
    var_13 = 23
    var_14 = module_0.Token(var_11, var_12, var_13, var_0)
    var_15 = {var_10: var_14}
    var_16 = 10
    var_17 = 24
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
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 15
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = 7
    var_9 = module_0.Token(var_4, var_7, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 11
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 15
    var_15 = 16
    var_16 = module_0.Token(var_5, var_14, var_15, var_0)
    var_17 = {var_6: var_9, var_13: var_16}
    var_18 = [var_17, var_1, var_2, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_17)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_1)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_2)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_0)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {'a': var_6, 'b': var_13})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {'a': var_9, 'b': var_16})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 0
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



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
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
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 3
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'world'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = bool(not var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == var_4)
    assert var_5 is True



# Parsed testcases at query #12
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
    var_9 = 8
    var_10 = 14
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
    var_8 = 6
    var_9 = module_0.Token(var_3, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 11
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 15
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
    var_0 = '{"test": true}'
    var_1 = 5
    var_2 = 10
    var_3 = 'test'
    var_4 = 6
    var_5 = 9
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = True
    var_8 = 12
    var_9 = 15
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"x": 100}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'x'
    var_6 = 2
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 100
    var_9 = 6
    var_10 = 8
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = [var_12, var_1, var_4, var_0]
    var_14 = {}
    var_15 = module_0.DictToken(*var_13, **var_14)
    var_16 = var_15._child_keys[var_7._value]
    var_17 = bool(var_15._child_keys[var_7._value] == var_7)
    assert var_17 is True
    var_18 = var_15._child_tokens[var_7._value]
    var_19 = bool(var_15._child_tokens[var_7._value] == var_11)
    assert var_19 is True



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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
    var_0 = '{"test": true}'
    var_1 = 5
    var_2 = 10
    var_3 = {}
    var_4 = [var_3, var_1, var_2, var_0]
    var_5 = {}
    var_6 = module_0.DictToken(*var_4, **var_5)
    var_7 = var_6._start_index
    var_8 = bool(var_6._start_index == var_1)
    assert var_8 is True
    var_9 = var_6._end_index
    var_10 = bool(var_6._end_index == var_2)
    assert var_10 is True
    var_11 = var_6._content
    var_12 = bool(var_6._content == var_0)
    assert var_12 is True
    var_13 = var_6._value
    var_14 = bool(var_6._value == var_3)
    assert var_14 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_eq_false_when_values_different. Retrieved 4/14 statements.
# Partially parsed test_eq_false_when_start_indices_different. Retrieved 3/13 statements.
# Partially parsed test_eq_false_when_end_indices_different. Retrieved 3/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'
    var_3 = 2

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'a'



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'value2'
    var_5 = module_0.Token(var_4, var_1, var_1, var_2)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = module_0.Token(var_4, var_1, var_1, var_2)
    var_6 = var_3 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = module_0.Token(var_0, var_0, var_1, var_2)
    var_5 = var_3 == var_4
    assert var_5 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = module_0.Token(var_0, var_1, var_0, var_2)
    var_5 = var_3 == var_4
    assert var_5 is False



# Parsed testcases at query #18
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
    var_21 = var_16._value
    var_22 = bool(var_16._value == var_9)
    assert var_22 is True
    var_23 = var_16._start_index
    assert var_23 == 0
    var_24 = var_16._end_index
    assert var_24 == 9
    var_25 = var_16._content
    assert var_25 == 'key: value'

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
    var_14 = var_9._value
    var_15 = bool(var_9._value == {})
    assert var_15 is True
    var_16 = var_9._start_index
    assert var_16 == 0
    var_17 = var_9._end_index
    assert var_17 == 0
    var_18 = var_9._content
    assert var_18 == '{}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'k1'
    var_1 = 0
    var_2 = 1
    var_3 = '{"k1": 1, "k2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = module_0.Token(var_2, var_5, var_5, var_3)
    var_7 = 'k2'
    var_8 = 9
    var_9 = 10
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 14
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = {var_4: var_6, var_10: var_13}
    var_15 = 16
    var_16 = []
    var_17 = 'value'
    var_18 = 'start_index'
    var_19 = 'end_index'
    var_20 = 'content'
    var_21 = {var_17: var_14, var_18: var_1, var_19: var_15, var_20: var_3}
    var_22 = module_0.DictToken(*var_16, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'k1': var_4, 'k2': var_10})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'k1': var_6, 'k2': var_13})
    assert var_26 is True
    var_27 = var_22._value
    var_28 = bool(var_22._value == var_14)
    assert var_28 is True
    var_29 = var_22._start_index
    assert var_29 == 0
    var_30 = var_22._end_index
    assert var_30 == 16
    var_31 = var_22._content
    assert var_31 == '{"k1": 1, "k2": 2}'



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = 7
    var_3 = []
    var_4 = module_0.ListToken(var_3, var_1, var_2, var_0)
    var_5 = var_4._value
    var_6 = bool(var_4._value == var_3)
    assert var_6 is True
    var_7 = var_4._start_index
    var_8 = bool(var_4._start_index == var_1)
    assert var_8 is True
    var_9 = var_4._end_index
    var_10 = bool(var_4._end_index == var_2)
    assert var_10 is True
    var_11 = var_4._content
    var_12 = bool(var_4._content == var_0)
    assert var_12 is True



# Parsed testcases at query #20
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
    var_0 = 'k1'
    var_1 = 0
    var_2 = 1
    var_3 = '{"k1":1,"k2":2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = module_0.Token(var_2, var_5, var_5, var_3)
    var_7 = 'k2'
    var_8 = 8
    var_9 = 9
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 13
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = {var_4: var_6, var_10: var_13}
    var_15 = 14
    var_16 = []
    var_17 = 'value'
    var_18 = 'start_index'
    var_19 = 'end_index'
    var_20 = 'content'
    var_21 = {var_17: var_14, var_18: var_1, var_19: var_15, var_20: var_3}
    var_22 = module_0.DictToken(*var_16, **var_21)
    var_23 = var_22._child_keys
    var_24 = bool(var_22._child_keys == {'k1': var_4, 'k2': var_10})
    assert var_24 is True
    var_25 = var_22._child_tokens
    var_26 = bool(var_22._child_tokens == {'k1': var_6, 'k2': var_13})
    assert var_26 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test2'
    var_5 = module_0.Token(var_4, var_1, var_2, var_4)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 1
    var_5 = module_0.Token(var_0, var_4, var_2, var_0)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 2
    var_5 = module_0.Token(var_0, var_1, var_4, var_0)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test'
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'different'
    var_5 = module_0.Token(var_0, var_1, var_2, var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 1
    var_2 = module_0.Token(var_1, var_1, var_1, var_0)
    var_3 = 2
    var_4 = 4
    var_5 = module_0.Token(var_3, var_4, var_4, var_0)
    var_6 = 3
    var_7 = 7
    var_8 = module_0.Token(var_6, var_7, var_7, var_0)
    var_9 = [var_2, var_5, var_8]
    var_10 = 0
    var_11 = 8
    var_12 = module_0.ListToken(var_9, var_10, var_11, var_0)
    var_13 = var_12._value
    var_14 = bool(var_12._value == var_9)
    assert var_14 is True
    var_15 = var_12._start_index
    var_16 = bool(var_12._start_index == var_10)
    assert var_16 is True
    var_17 = var_12._end_index
    var_18 = bool(var_12._end_index == var_11)
    assert var_18 is True
    var_19 = var_12._content
    var_20 = bool(var_12._content == var_0)
    assert var_20 is True



# Parsed testcases at query #23
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
    var_9 = 8
    var_10 = 14
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
    var_8 = 6
    var_9 = module_0.Token(var_3, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 9
    var_12 = 10
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 14
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
    var_0 = '{"outer": {"inner": 3}}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'inner'
    var_6 = 11
    var_7 = 16
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = 3
    var_10 = 19
    var_11 = module_0.Token(var_9, var_10, var_10, var_0)
    var_12 = {var_8: var_11}
    var_13 = 10
    var_14 = 20
    var_15 = [var_12, var_13, var_14, var_0]
    var_16 = {}
    var_17 = module_0.DictToken(*var_15, **var_16)
    var_18 = 'outer'
    var_19 = 6
    var_20 = module_0.Token(var_18, var_3, var_19, var_0)
    var_21 = {var_20: var_17}
    var_22 = [var_21, var_1, var_4, var_0]
    var_23 = {}
    var_24 = module_0.DictToken(*var_22, **var_23)
    var_25 = var_24._value
    var_26 = bool(var_24._value == var_21)
    assert var_26 is True
    var_27 = var_24._start_index
    var_28 = bool(var_24._start_index == var_1)
    assert var_28 is True
    var_29 = var_24._end_index
    var_30 = bool(var_24._end_index == var_4)
    assert var_30 is True
    var_31 = var_24._content
    var_32 = bool(var_24._content == var_0)
    assert var_32 is True
    var_33 = var_24._child_keys
    var_34 = bool(var_24._child_keys == {'outer': var_20})
    assert var_34 is True
    var_35 = var_24._child_tokens
    var_36 = bool(var_24._child_tokens == {'outer': var_17})
    assert var_36 is True

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
    var_9 = 8
    var_10 = 14
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 17
    var_13 = 20
    var_14 = module_0.Token(var_5, var_12, var_13, var_0)
    var_15 = 'second'
    var_16 = 24
    var_17 = 31
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



# Parsed testcases at query #24
#--------------------------




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



# Parsed testcases at query #25
#--------------------------




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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_eq_false_when_values_differ. Retrieved 3/12 statements.
# Partially parsed test_eq_false_when_start_indices_differ. Retrieved 4/10 statements.
# Partially parsed test_eq_false_when_end_indices_differ. Retrieved 4/10 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'not a token'
    var_5 = var_3 == var_4
    var_6 = bool(not var_5)
    assert var_6 is True

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = 1



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 5
    var_6 = 'sample'
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
    var_10 = var_9._value
    var_11 = bool(var_9._value == {})
    assert var_11 is True
    var_12 = var_9._start_index
    assert var_12 == 0
    var_13 = var_9._end_index
    assert var_13 == 0
    var_14 = var_9._content
    assert var_14 == '{}'
    var_15 = var_9._child_keys
    var_16 = bool(var_9._child_keys == {})
    assert var_16 is True
    var_17 = var_9._child_tokens
    var_18 = bool(var_9._child_tokens == {})
    assert var_18 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = 7
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = 'key2'
    var_9 = 11
    var_10 = 14
    var_11 = module_0.Token(var_8, var_9, var_10, var_3)
    var_12 = 2
    var_13 = 18
    var_14 = module_0.Token(var_12, var_13, var_13, var_3)
    var_15 = {var_4: var_7, var_11: var_14}
    var_16 = 20
    var_17 = []
    var_18 = 'value'
    var_19 = 'start_index'
    var_20 = 'end_index'
    var_21 = 'content'
    var_22 = {var_18: var_15, var_19: var_1, var_20: var_16, var_21: var_3}
    var_23 = module_0.DictToken(*var_17, **var_22)
    var_24 = var_23._value
    var_25 = bool(var_23._value == var_15)
    assert var_25 is True
    var_26 = var_23._start_index
    assert var_26 == 0
    var_27 = var_23._end_index
    assert var_27 == 20
    var_28 = var_23._content
    assert var_28 == '{"key1": 1, "key2": 2}'
    var_29 = var_23._child_keys
    var_30 = bool(var_23._child_keys == {'key1': var_4, 'key2': var_11})
    assert var_30 is True
    var_31 = var_23._child_tokens
    var_32 = bool(var_23._child_tokens == {'key1': var_7, 'key2': var_14})
    assert var_32 is True



# Parsed testcases at query #30
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
    var_3 = '{"key": 1}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 7
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = {var_4: var_6}
    var_8 = '{"key": 1}'
    var_9 = 0
    var_10 = 9
    var_11 = [var_7, var_9, var_10, var_8]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = var_13._value
    var_15 = bool(var_13._value == var_7)
    assert var_15 is True
    var_16 = var_13._start_index
    var_17 = bool(var_13._start_index == var_9)
    assert var_17 is True
    var_18 = var_13._end_index
    var_19 = bool(var_13._end_index == var_10)
    assert var_19 is True
    var_20 = var_13._content
    var_21 = bool(var_13._content == var_8)
    assert var_21 is True
    var_22 = var_13._child_keys
    var_23 = bool(var_13._child_keys == {'key': var_4})
    assert var_23 is True
    var_24 = var_13._child_tokens
    var_25 = bool(var_13._child_tokens == {'key': var_6})
    assert var_25 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = 5
    var_3 = '{"key1": 1, "key2": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 9
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 'key2'
    var_8 = 12
    var_9 = 16
    var_10 = module_0.Token(var_7, var_8, var_9, var_3)
    var_11 = 2
    var_12 = 20
    var_13 = module_0.Token(var_11, var_12, var_12, var_3)
    var_14 = {var_4: var_6, var_10: var_13}
    var_15 = '{"key1": 1, "key2": 2}'
    var_16 = 0
    var_17 = 23
    var_18 = [var_14, var_16, var_17, var_15]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_14)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_16)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_17)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_15)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {'key1': var_4, 'key2': var_10})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {'key1': var_6, 'key2': var_13})
    assert var_32 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 1
    var_2 = 3
    var_3 = '{123: 1}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = {var_4: var_6}
    var_8 = '{123: 1}'
    var_9 = 0
    var_10 = 7
    var_11 = [var_7, var_9, var_10, var_8]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = var_13._value
    var_15 = bool(var_13._value == var_7)
    assert var_15 is True
    var_16 = var_13._start_index
    var_17 = bool(var_13._start_index == var_9)
    assert var_17 is True
    var_18 = var_13._end_index
    var_19 = bool(var_13._end_index == var_10)
    assert var_19 is True
    var_20 = var_13._content
    var_21 = bool(var_13._content == var_8)
    assert var_21 is True
    var_22 = var_13._child_keys
    var_23 = bool(var_13._child_keys == {123: var_4})
    assert var_23 is True
    var_24 = var_13._child_tokens
    var_25 = bool(var_13._child_tokens == {123: var_6})
    assert var_25 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 7
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 9
    var_10 = [var_7, var_8, var_9]
    var_11 = {}
    var_12 = module_0.DictToken(*var_10, **var_11)
    var_13 = var_12._value
    var_14 = bool(var_12._value == var_7)
    assert var_14 is True
    var_15 = var_12._start_index
    var_16 = bool(var_12._start_index == var_8)
    assert var_16 is True
    var_17 = var_12._end_index
    var_18 = bool(var_12._end_index == var_9)
    assert var_18 is True
    var_19 = var_12._content
    assert var_19 == ''
    var_20 = var_12._child_keys
    var_21 = bool(var_12._child_keys == {'key': var_4})
    assert var_21 is True
    var_22 = var_12._child_tokens
    var_23 = bool(var_12._child_tokens == {'key': var_6})
    assert var_23 is True



# Parsed testcases at query #31
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
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '"key": 5'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = 6
    var_7 = module_0.Token(var_5, var_6, var_6, var_3)
    var_8 = '{"key": 5}'
    var_9 = 0
    var_10 = 9
    var_11 = {var_4: var_7}
    var_12 = [var_11, var_9, var_10, var_8]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = var_14._value
    var_16 = bool(var_14._value == {var_4: var_7})
    assert var_16 is True
    var_17 = var_14._start_index
    assert var_17 == 0
    var_18 = var_14._end_index
    assert var_18 == 9
    var_19 = var_14._content
    assert var_19 == '{"key": 5}'
    var_20 = var_14._child_keys
    var_21 = bool(var_14._child_keys == {'key': var_4})
    assert var_21 is True
    var_22 = var_14._child_tokens
    var_23 = bool(var_14._child_tokens == {'key': var_7})
    assert var_23 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = '{"a": 1, "b": 2}'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 5
    var_5 = module_0.Token(var_1, var_4, var_4, var_2)
    var_6 = 'b'
    var_7 = 9
    var_8 = module_0.Token(var_6, var_7, var_7, var_2)
    var_9 = 2
    var_10 = 13
    var_11 = module_0.Token(var_9, var_10, var_10, var_2)
    var_12 = '{"a": 1, "b": 2}'
    var_13 = 0
    var_14 = 15
    var_15 = {var_3: var_5, var_8: var_11}
    var_16 = [var_15, var_13, var_14, var_12]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == {var_3: var_5, var_8: var_11})
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 15
    var_23 = var_18._content
    assert var_23 == '{"a": 1, "b": 2}'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_3, 'b': var_8})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_5, 'b': var_11})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 8
    var_2 = 12
    var_3 = '{"outer": {"inner": 42}}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 42
    var_6 = 16
    var_7 = 17
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 7
    var_11 = 19
    var_12 = [var_9, var_10, var_11, var_3]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = 'outer'
    var_16 = 1
    var_17 = 5
    var_18 = module_0.Token(var_15, var_16, var_17, var_3)
    var_19 = '{"outer": {"inner": 42}}'
    var_20 = 0
    var_21 = 23
    var_22 = {var_18: var_14}
    var_23 = [var_22, var_20, var_21, var_19]
    var_24 = {}
    var_25 = module_0.DictToken(*var_23, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == {var_18: var_14})
    assert var_27 is True
    var_28 = var_25._start_index
    assert var_28 == 0
    var_29 = var_25._end_index
    assert var_29 == 23
    var_30 = var_25._content
    assert var_30 == '{"outer": {"inner": 42}}'
    var_31 = var_25._child_keys
    var_32 = bool(var_25._child_keys == {'outer': var_18})
    assert var_32 is True
    var_33 = var_25._child_tokens
    var_34 = bool(var_25._child_tokens == {'outer': var_14})
    assert var_34 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '{"key": 1, "key": 2}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = 10
    var_8 = 12
    var_9 = module_0.Token(var_0, var_7, var_8, var_3)
    var_10 = 2
    var_11 = 15
    var_12 = module_0.Token(var_10, var_11, var_11, var_3)
    var_13 = '{"key": 1, "key": 2}'
    var_14 = 0
    var_15 = 17
    var_16 = {var_4: var_6, var_9: var_12}
    var_17 = [var_16, var_14, var_15, var_13]
    var_18 = {}
    var_19 = module_0.DictToken(*var_17, **var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == {var_4: var_6, var_9: var_12})
    assert var_21 is True
    var_22 = var_19._start_index
    assert var_22 == 0
    var_23 = var_19._end_index
    assert var_23 == 17
    var_24 = var_19._content
    assert var_24 == '{"key": 1, "key": 2}'
    var_25 = var_19._child_keys
    var_26 = bool(var_19._child_keys == {'key': var_9})
    assert var_26 is True
    var_27 = var_19._child_tokens
    var_28 = bool(var_19._child_tokens == {'key': var_12})
    assert var_28 is True



# Parsed testcases at query #32
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
    var_0 = 'key'
    var_1 = 1
    var_2 = 3
    var_3 = '"key": 1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_1, var_5, var_5, var_3)
    var_7 = '{"key": 1}'
    var_8 = 0
    var_9 = 9
    var_10 = {var_4: var_6}
    var_11 = [var_10, var_8, var_9, var_7]
    var_12 = {}
    var_13 = module_0.DictToken(*var_11, **var_12)
    var_14 = var_13._value
    var_15 = bool(var_13._value == {var_4: var_6})
    assert var_15 is True
    var_16 = var_13._start_index
    assert var_16 == 0
    var_17 = var_13._end_index
    assert var_17 == 9
    var_18 = var_13._content
    assert var_18 == '{"key": 1}'
    var_19 = var_13._child_keys
    var_20 = bool(var_13._child_keys == {'key': var_4})
    assert var_20 is True
    var_21 = var_13._child_tokens
    var_22 = bool(var_13._child_tokens == {'key': var_6})
    assert var_22 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = '{"a": 1, "b": 2}'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 5
    var_5 = module_0.Token(var_1, var_4, var_4, var_2)
    var_6 = 'b'
    var_7 = 9
    var_8 = module_0.Token(var_6, var_7, var_7, var_2)
    var_9 = 2
    var_10 = 13
    var_11 = module_0.Token(var_9, var_10, var_10, var_2)
    var_12 = '{"a": 1, "b": 2}'
    var_13 = 0
    var_14 = 15
    var_15 = {var_3: var_5, var_8: var_11}
    var_16 = [var_15, var_13, var_14, var_12]
    var_17 = {}
    var_18 = module_0.DictToken(*var_16, **var_17)
    var_19 = var_18._value
    var_20 = bool(var_18._value == {var_3: var_5, var_8: var_11})
    assert var_20 is True
    var_21 = var_18._start_index
    assert var_21 == 0
    var_22 = var_18._end_index
    assert var_22 == 15
    var_23 = var_18._content
    assert var_23 == '{"a": 1, "b": 2}'
    var_24 = var_18._child_keys
    var_25 = bool(var_18._child_keys == {'a': var_3, 'b': var_8})
    assert var_25 is True
    var_26 = var_18._child_tokens
    var_27 = bool(var_18._child_tokens == {'a': var_5, 'b': var_11})
    assert var_27 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 8
    var_2 = 12
    var_3 = '{"outer": {"inner": 42}}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 42
    var_6 = 16
    var_7 = 17
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 7
    var_11 = 19
    var_12 = [var_9, var_10, var_11, var_3]
    var_13 = {}
    var_14 = module_0.DictToken(*var_12, **var_13)
    var_15 = 'outer'
    var_16 = 1
    var_17 = 5
    var_18 = module_0.Token(var_15, var_16, var_17, var_3)
    var_19 = '{"outer": {"inner": 42}}'
    var_20 = 0
    var_21 = 23
    var_22 = {var_18: var_14}
    var_23 = [var_22, var_20, var_21, var_19]
    var_24 = {}
    var_25 = module_0.DictToken(*var_23, **var_24)
    var_26 = var_25._value
    var_27 = bool(var_25._value == {var_18: var_14})
    assert var_27 is True
    var_28 = var_25._start_index
    assert var_28 == 0
    var_29 = var_25._end_index
    assert var_29 == 23
    var_30 = var_25._content
    assert var_30 == '{"outer": {"inner": 42}}'
    var_31 = var_25._child_keys
    var_32 = bool(var_25._child_keys == {'outer': var_18})
    assert var_32 is True
    var_33 = var_25._child_tokens
    var_34 = bool(var_25._child_tokens == {'outer': var_14})
    assert var_34 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_list_token_constructor_lookup. Retrieved 11/12 statements.
# Partially parsed test_list_token_constructor_get_child_token. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = module_0.ListToken(var_0, var_1, var_1)
    var_3 = var_2._value
    var_4 = bool(var_2._value == [])
    assert var_4 is True
    var_5 = var_2._start_index
    assert var_5 == 0
    var_6 = var_2._end_index
    assert var_6 == 0
    var_7 = var_2._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == 5
    var_8 = var_4._end_index
    assert var_8 == 10
    var_9 = var_4._content
    assert var_9 == 'test content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 1
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = [var_4]
    var_6 = 0
    var_7 = 3
    var_8 = '[42]'
    var_9 = module_0.ListToken(var_5, var_6, var_7, var_8)
    var_10 = var_9._value
    var_11 = bool(var_9._value == [var_4])
    assert var_11 is True
    var_12 = var_9._start_index
    assert var_12 == 0
    var_13 = var_9._end_index
    assert var_13 == 3
    var_14 = var_9._content
    assert var_14 == '[42]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = []
    var_5 = module_0.ListToken(var_4, var_1, var_1, var_2)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = module_0.ListToken(var_4, var_1, var_1, var_2)
    var_6 = []
    var_7 = module_0.ListToken(var_6, var_1, var_1, var_2)
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 0
    var_3 = ''
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = []
    var_6 = module_0.ListToken(var_5, var_2, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 1
    var_3 = ''
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = []
    var_6 = module_0.ListToken(var_5, var_1, var_1, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = bool(not var_3 == 'not a token')
    assert var_4 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = 5
    var_3 = 'abcdef'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.string
    assert var_5 == 'cdef'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = [var_3]
    var_5 = 6
    var_6 = '[inner]'
    var_7 = module_0.ListToken(var_4, var_1, var_5, var_6)
    var_8 = var_7.value
    var_9 = bool(var_7.value == ['inner'])
    assert var_9 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 10
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.start
    var_6 = var_5.line
    assert var_6 == 2
    var_7 = var_5.column
    assert var_7 == 1
    var_8 = var_5.index
    assert var_8 == 5

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 10
    var_3 = 'line1\nline2\nline3'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.end
    var_6 = var_5.line
    assert var_6 == 2
    var_7 = var_5.column
    assert var_7 == 6
    var_8 = var_5.index
    assert var_8 == 10

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 3
    var_3 = 'test'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ListToken('est')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 99
    var_1 = 1
    var_2 = 2
    var_3 = 'xyz'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = [var_4]
    var_6 = 0
    var_7 = 3
    var_8 = '[99]'
    var_9 = module_0.ListToken(var_5, var_6, var_7, var_8)
    var_10 = [var_6]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 99
    var_1 = 1
    var_2 = 2
    var_3 = 'xyz'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = [var_4]
    var_6 = 0
    var_7 = 3
    var_8 = '[99]'
    var_9 = module_0.ListToken(var_5, var_6, var_7, var_8)



# Parsed testcases at query #34
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
    var_2 = 31
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dict_token_init_with_non_token_keys. Retrieved 3/10 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 0



# Parsed testcases at query #36
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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 2
    var_3 = '"a": true'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = 6
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
    assert var_21 == '"a": true'

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 3
    var_3 = '{"x": 10, "y": 20}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 10
    var_6 = 7
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'y'
    var_10 = 12
    var_11 = 14
    var_12 = module_0.Token(var_9, var_10, var_11, var_3)
    var_13 = 20
    var_14 = 18
    var_15 = 19
    var_16 = module_0.Token(var_13, var_14, var_15, var_3)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = []
    var_20 = 'value'
    var_21 = 'start_index'
    var_22 = 'end_index'
    var_23 = 'content'
    var_24 = {var_20: var_17, var_21: var_18, var_22: var_13, var_23: var_3}
    var_25 = module_0.DictToken(*var_19, **var_24)
    var_26 = var_25._child_keys
    var_27 = bool(var_25._child_keys == {'x': var_4, 'y': var_12})
    assert var_27 is True
    var_28 = var_25._child_tokens
    var_29 = bool(var_25._child_tokens == {'x': var_8, 'y': var_16})
    assert var_29 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 0
    var_2 = 3
    var_3 = '"id": 5'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 8
    var_6 = 11
    var_7 = '"id": 5, "id": 10'
    var_8 = module_0.Token(var_0, var_5, var_6, var_7)
    var_9 = 5
    var_10 = 6
    var_11 = module_0.Token(var_9, var_10, var_10, var_7)
    var_12 = 10
    var_13 = 15
    var_14 = 16
    var_15 = module_0.Token(var_12, var_13, var_14, var_7)
    var_16 = {var_4: var_11, var_8: var_15}
    var_17 = 17
    var_18 = []
    var_19 = 'value'
    var_20 = 'start_index'
    var_21 = 'end_index'
    var_22 = 'content'
    var_23 = {var_19: var_16, var_20: var_1, var_21: var_17, var_22: var_7}
    var_24 = module_0.DictToken(*var_18, **var_23)
    var_25 = var_24._child_keys
    var_26 = bool(var_24._child_keys == {'id': var_8})
    assert var_26 is True
    var_27 = var_24._child_tokens
    var_28 = bool(var_24._child_tokens == {'id': var_15})
    assert var_28 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_eq_returns_false_when_get_value_differs. Retrieved 8/10 statements.
# Partially parsed test_eq_returns_false_when_start_index_differs. Retrieved 7/9 statements.
# Partially parsed test_eq_returns_false_when_end_index_differs. Retrieved 8/10 statements.
# Partially parsed test_eq_returns_false_when_other_is_not_token. Retrieved 7/8 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 2
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_0, var_2, var_3)
    var_6 = var_4 == var_5
    assert var_6 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 6
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = var_4 == var_6
    assert var_7 is False

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 5
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'not a token'
    var_6 = var_4 == var_5
    assert var_6 is False



# Parsed testcases at query #38
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
    var_0 = '{"outer": {"inner": 42}}'
    var_1 = 0
    var_2 = 25
    var_3 = 'outer'
    var_4 = 1
    var_5 = 7
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'inner'
    var_8 = 12
    var_9 = 17
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
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = 15
    var_3 = 'a'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 6
    var_8 = 7
    var_9 = module_0.Token(var_4, var_7, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = 11
    var_13 = module_0.Token(var_10, var_11, var_12, var_0)
    var_14 = 15
    var_15 = 16
    var_16 = module_0.Token(var_5, var_14, var_15, var_0)
    var_17 = {var_6: var_9, var_13: var_16}
    var_18 = [var_17, var_1, var_2, var_0]
    var_19 = {}
    var_20 = module_0.DictToken(*var_18, **var_19)
    var_21 = var_20._value
    var_22 = bool(var_20._value == var_17)
    assert var_22 is True
    var_23 = var_20._start_index
    var_24 = bool(var_20._start_index == var_1)
    assert var_24 is True
    var_25 = var_20._end_index
    var_26 = bool(var_20._end_index == var_2)
    assert var_26 is True
    var_27 = var_20._content
    var_28 = bool(var_20._content == var_0)
    assert var_28 is True
    var_29 = var_20._child_keys
    var_30 = bool(var_20._child_keys == {'a': var_6, 'b': var_13})
    assert var_30 is True
    var_31 = var_20._child_tokens
    var_32 = bool(var_20._child_tokens == {'a': var_9, 'b': var_16})
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



