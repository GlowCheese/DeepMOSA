####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/23 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 14/32 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 'key2'
    var_7 = 12
    var_8 = 15
    var_9 = 'value2'
    var_10 = 17
    var_11 = 22
    var_12 = 25
    var_13 = 'key1value1key2value2'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 20/27 statements.


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
    var_18 = [var_16, var_17]
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'test2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == None)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == {'value': 'test'})
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 20/23 statements.


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
    var_18 = [var_16, var_17]
    var_19 = 'key1:value1,key2:value2'
    var_20 = 'key1'
    var_21 = 'key2'
    var_22 = 'key1'
    var_23 = 'key2'



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'some content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2, var_3, var_4)
    var_6 = var_5._value
    var_7 = bool(var_5._value == var_2)
    assert var_7 is True
    var_8 = var_5._start_index
    assert var_8 == 0
    var_9 = var_5._end_index
    assert var_9 == 10
    var_10 = var_5._content
    assert var_10 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 'v'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0
    var_6 = var_3._value
    assert var_6 == 'val'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 14/31 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 'key2'
    var_7 = 12
    var_8 = 15
    var_9 = 'value2'
    var_10 = 17
    var_11 = 22
    var_12 = 23
    var_13 = 'key1: value1, key2: value2'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_eq_predicate_line_3_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = 'value2'



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'b'
    var_5 = 1
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'c'
    var_8 = 2
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.ListToken(var_10, var_1, var_8, var_2)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [var_3, var_6, var_9])
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 0
    var_15 = var_11._end_index
    assert var_15 == 2
    var_16 = var_11._content
    assert var_16 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = [var_2]
    var_4 = module_0.ListToken(var_3, var_1, var_1)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [var_2])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == 0
    var_8 = var_4._end_index
    assert var_8 == 0
    var_9 = var_4._content
    assert var_9 == ''



# Parsed testcases at query #11
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
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'x'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == [1, 2, 3])
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'test'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {'key': 'value'})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'test'



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = 'test1 content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'test2'
    var_6 = 'test2 content'
    var_7 = module_0.Token(var_5, var_1, var_2, var_6)
    var_8 = bool(not var_4 == var_7)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
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
    var_2 = 4
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == None)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == {'value': 'test'})
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 0
    var_3 = 10
    var_4 = 'test_content'
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 'key1'
    var_8 = 'key2'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'hello world'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 18/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1=value1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 'key2=value2'
    var_11 = module_0.Token(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.Token(var_12, var_6, var_7, var_10)
    var_14 = (var_4, var_8)
    var_15 = (var_11, var_13)
    var_16 = [var_14, var_15]
    var_17 = 'test_content'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key1: value1'
    var_4 = 'value1'
    var_5 = 6
    var_6 = 11
    var_7 = 'key2'
    var_8 = 'key2: value2'
    var_9 = 'value2'
    var_10 = 23
    var_11 = 'key1: value1, key2: value2'



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 15
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 15
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'x'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 10
    var_5 = 'test content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'test content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 20
    var_6 = 'some content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 0
    var_11 = var_7._end_index
    assert var_11 == 20
    var_12 = var_7._content
    assert var_12 == 'some content'



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'test'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'test'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = var_2._value
    assert var_3 == 'x'
    var_4 = var_2._start_index
    assert var_4 == 0
    var_5 = var_2._end_index
    assert var_5 == 0
    var_6 = var_2._content
    assert var_6 == 'x'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'large'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'a'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._value
    assert var_7 == 'large'
    var_8 = var_6._start_index
    assert var_8 == 1000
    var_9 = var_6._end_index
    assert var_9 == 2000
    var_10 = var_6._content
    var_11 = len(var_10)
    assert var_11 == 2001



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'x'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 'key1: value1'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = []



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'some_content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == [1, 2, 3])
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2, var_3, var_4)
    var_6 = var_5._value
    var_7 = bool(var_5._value == {'key': 'value'})
    assert var_7 is True
    var_8 = var_5._start_index
    assert var_8 == 0
    var_9 = var_5._end_index
    assert var_9 == 10
    var_10 = var_5._content
    assert var_10 == ''



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 18/19 statements.


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



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
    var_5 = var_3._start_index
    assert var_5 == 10
    var_6 = var_3._end_index
    assert var_6 == 20
    var_7 = var_3._content
    assert var_7 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 123
    var_6 = 1
    var_7 = 3
    var_8 = [var_6, var_2, var_7]
    var_9 = 9
    var_10 = '[1, 2, 3]'
    var_11 = module_0.Token(var_8, var_1, var_9, var_10)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [1, 2, 3])
    assert var_13 is True
    var_14 = 'key'
    var_15 = 'val'
    var_16 = {var_14: var_15}
    var_17 = 15
    var_18 = '{"key": "val"}'
    var_19 = module_0.Token(var_16, var_1, var_17, var_18)
    var_20 = var_19._value
    var_21 = bool(var_19._value == {'key': 'val'})
    assert var_21 is True
    var_22 = None
    var_23 = 4
    var_24 = 'None'
    var_25 = module_0.Token(var_22, var_1, var_23, var_24)
    var_26 = var_25._value
    assert var_26 is None



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 10
    var_6 = 'some_content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == [1, 2, 3])
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 10
    var_12 = var_7._content
    assert var_12 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = 'dictionary_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == {'key': 'value'})
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'dictionary_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 'v'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'x'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 11/28 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key1: value1'
    var_4 = 'value1'
    var_5 = 6
    var_6 = 11
    var_7 = 'key2'
    var_8 = 'key2: value2'
    var_9 = 'value2'
    var_10 = 'key1: value1, key2: value2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 16/18 statements.


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
    var_10 = 'key2: value2'
    var_11 = module_0.Token(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.Token(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'key1: value1, key2: value2'
    var_16 = [var_14, var_1, var_7, var_15]
    var_17 = 'key1'
    var_18 = 'key2'
    var_19 = 'key1'
    var_20 = 'key2'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 13/30 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key2'
    var_4 = 6
    var_5 = 10
    var_6 = 'val1'
    var_7 = 12
    var_8 = 15
    var_9 = 'val2'
    var_10 = 17
    var_11 = 20
    var_12 = 'key1:val1,key2:val2'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/23 statements.


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



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'b'
    var_5 = 1
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'c'
    var_8 = 2
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.ListToken(var_10, var_1, var_8, var_2)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [var_3, var_6, var_9])
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 0
    var_15 = var_11._end_index
    assert var_15 == 2
    var_16 = var_11._content
    assert var_16 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = [var_2]
    var_4 = module_0.ListToken(var_3, var_1, var_1)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [var_2])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == 0
    var_8 = var_4._end_index
    assert var_8 == 0
    var_9 = var_4._content
    assert var_9 == ''



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 'key2value'
    var_11 = module_0.Token(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.Token(var_12, var_6, var_7, var_10)
    var_14 = {var_4: var_8, var_11: var_13}
    var_15 = 'test_content'
    var_16 = [var_14, var_1, var_7, var_15]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/32 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key1_content'
    var_4 = 'key2'
    var_5 = 5
    var_6 = 9
    var_7 = 'key2_content'
    var_8 = 'value1'
    var_9 = 10
    var_10 = 16
    var_11 = 'value1_content'
    var_12 = 'value2'
    var_13 = 17
    var_14 = 23
    var_15 = 'value2_content'
    var_16 = 24
    var_17 = 'test_content'



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello world'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'test'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'test'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'dict content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'dict content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'large'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'a'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = 'key2'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'test content'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 18/20 statements.


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
    var_19 = 'key1'
    var_20 = 'key2'
    var_21 = 'key1'
    var_22 = 'key2'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 12/30 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'key1: value1'
    var_4 = 'value1'
    var_5 = 7
    var_6 = 12
    var_7 = 'key2'
    var_8 = 'key2: value2'
    var_9 = 'value2'
    var_10 = 25
    var_11 = 'key1: value1, key2: value2'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 15
    var_7 = 'key1value1'



# Parsed testcases at query #42
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 20/25 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.Token(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.Token(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = []



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 22/25 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1: value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = module_0.Token(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = 'key2: value2'
    var_17 = module_0.Token(var_13, var_14, var_15, var_16)
    var_18 = (var_3, var_8)
    var_19 = (var_12, var_17)
    var_20 = [var_18, var_19]
    var_21 = 'key1: value1, key2: value2'
    var_22 = 'key1'
    var_23 = 'key2'
    var_24 = 'key1'
    var_25 = 'key2'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_token_init_sets_start_index. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_token_init_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 0
    var_3 = 10
    var_4 = 'test_content'



# Parsed testcases at query #49
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'x'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = [var_9, var_1, var_6, var_7]



# Parsed testcases at query #51
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'some_content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'dict_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'dict_content'



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 13/27 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key2'
    var_4 = 5
    var_5 = 8
    var_6 = 'value1'
    var_7 = 10
    var_8 = 15
    var_9 = 'value2'
    var_10 = 17
    var_11 = 22
    var_12 = 'key1: value1, key2: value2'



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'test2'
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 5
    var_6 = module_0.Token(var_0, var_1, var_5, var_3)
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == 'not a token')
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == None)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4 == {'value': 'test'})
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 10
    var_5 = 'example'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'example'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = module_0.Token(var_3, var_4, var_2)
    var_6 = var_5._value
    var_7 = bool(var_5._value == var_3)
    assert var_7 is True
    var_8 = var_5._start_index
    assert var_8 == 0
    var_9 = var_5._end_index
    assert var_9 == 3
    var_10 = var_5._content
    assert var_10 == ''



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'b'
    var_5 = 1
    var_6 = module_0.Token(var_4, var_5, var_5, var_2)
    var_7 = 'c'
    var_8 = 2
    var_9 = module_0.Token(var_7, var_8, var_8, var_2)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.ListToken(var_10, var_1, var_8, var_2)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [var_3, var_6, var_9])
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 0
    var_15 = var_11._end_index
    assert var_15 == 2
    var_16 = var_11._content
    assert var_16 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = 2
    var_4 = module_0.Token(var_3, var_0, var_0)
    var_5 = [var_2, var_4]
    var_6 = module_0.ListToken(var_5, var_1, var_0)
    var_7 = var_6._value
    var_8 = bool(var_6._value == [var_2, var_4])
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 1
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/23 statements.


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



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 10
    var_6 = 'some_content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == [1, 2, 3])
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 10
    var_12 = var_7._content
    assert var_12 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = 'content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'a'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 20/26 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = module_0.Token(var_9, var_10, var_11, var_9)
    var_13 = 'value2'
    var_14 = 17
    var_15 = 22
    var_16 = 'key2value2'
    var_17 = module_0.Token(var_13, var_14, var_15, var_16)
    var_18 = {var_3: var_8, var_12: var_17}
    var_19 = 'key1value1key2value2'
    var_20 = [var_18, var_1, var_15, var_19]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = 'key2'
    var_3 = 'value2'
    var_4 = 0
    var_5 = 10
    var_6 = 'test_content'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_token_init_start_index. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/20 statements.


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
    var_19 = 'key1'
    var_20 = 'key2'
    var_21 = 'key1'
    var_22 = 'key2'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1value1'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 'key2value2'
    var_11 = module_0.Token(var_9, var_1, var_2, var_10)
    var_12 = 'value2'
    var_13 = module_0.Token(var_12, var_6, var_7, var_10)
    var_14 = (var_4, var_8)
    var_15 = (var_11, var_13)
    var_16 = [var_14, var_15]
    var_17 = 'content'
    var_18 = 'key1'
    var_19 = 'key2'
    var_20 = 'key1'
    var_21 = 'key2'



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 42
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'hello'

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
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'x'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    var_5 = bool(var_3._value == [])
    assert var_5 is True
    var_6 = var_3._start_index
    assert var_6 == 0
    var_7 = var_3._end_index
    assert var_7 == 0
    var_8 = var_3._content
    assert var_8 == 'a'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1000
    var_4 = 2000
    var_5 = 'x'
    var_6 = 2001
    var_7 = var_5 * var_6
    var_8 = module_0.Token(var_2, var_3, var_4, var_7)
    var_9 = var_8._value
    var_10 = bool(var_8._value == {'key': 'value'})
    assert var_10 is True
    var_11 = var_8._start_index
    assert var_11 == 1000
    var_12 = var_8._end_index
    assert var_12 == 2000
    var_13 = var_8._content
    var_14 = len(var_13)
    assert var_14 == 2001

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = {var_0: var_5, var_1: var_8}
    var_10 = 10
    var_11 = 'test content'
    var_12 = module_0.Token(var_9, var_7, var_10, var_11)
    var_13 = var_12._value
    var_14 = bool(var_12._value == var_9)
    assert var_14 is True
    var_15 = var_12._start_index
    assert var_15 == 5
    var_16 = var_12._end_index
    assert var_16 == 10
    var_17 = var_12._content
    assert var_17 == 'test content'



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 15
    var_6 = 'some_content'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = var_7._value
    var_9 = bool(var_7._value == var_3)
    assert var_9 is True
    var_10 = var_7._start_index
    assert var_10 == 5
    var_11 = var_7._end_index
    assert var_11 == 15
    var_12 = var_7._content
    assert var_12 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = 'dict_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == 'dict_content'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 10/11 statements.


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
    var_9 = 'key1value1'
    var_10 = [var_8, var_1, var_6, var_9]



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 5
    var_9 = 15
    var_10 = 'some content here'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = var_11._value
    var_13 = bool(var_11._value == var_7)
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 5
    var_15 = var_11._end_index
    assert var_15 == 15
    var_16 = var_11._content
    assert var_16 == 'some content here'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 0
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 'data'
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'x'



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 0
    var_3 = 10
    var_4 = 'test_content'
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 'key1'
    var_8 = 'key2'



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 18/19 statements.


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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 'key1value1'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 20/26 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 12
    var_11 = 15
    var_12 = 'key1value1key2'
    var_13 = module_0.Token(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 17
    var_16 = 22
    var_17 = 'key1value1key2value2'
    var_18 = module_0.Token(var_14, var_15, var_16, var_17)
    var_19 = {var_3: var_8, var_13: var_18}
    var_20 = [var_19, var_1, var_16, var_17]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 22/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 4
    var_7 = 9
    var_8 = 'value1_content'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = 'key2'
    var_11 = 10
    var_12 = 13
    var_13 = 'key2_content'
    var_14 = module_0.Token(var_10, var_11, var_12, var_13)
    var_15 = 'value2'
    var_16 = 14
    var_17 = 19
    var_18 = 'value2_content'
    var_19 = module_0.Token(var_15, var_16, var_17, var_18)
    var_20 = {var_4: var_9, var_14: var_19}
    var_21 = 'key1_content:value1_content,key2_content:value2_content'
    var_22 = [var_20, var_1, var_17, var_21]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dict_token_init_predicate_false. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10
    var_6 = 'key1value1'



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 20/27 statements.


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
    var_18 = [var_16, var_17]
    var_19 = 'key1value1key2value2'



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = 'xyz'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0
    var_6 = var_3._value
    assert var_6 == 'x'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 9/26 statements.


def test_case_0():
    var_0 = '_value'
    var_1 = '_get_value'
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = [var_1]
    var_5 = [var_1]
    var_6 = 0
    var_7 = 10
    var_8 = 'test_content'



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 5
    var_2 = 10
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 5
    var_7 = var_4._end_index
    assert var_7 == 10
    var_8 = var_4._content
    assert var_8 == 'test_content'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dict_token_init. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {var_3: var_8}
    var_10 = [var_9, var_1, var_6, var_7]
    var_11 = 'key1'
    var_12 = 'key1'



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = var_2._start_index
    assert var_3 == 0
    var_4 = var_2._end_index
    assert var_4 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'large'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'x'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'hello world'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_token_init_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dicttoken_constructor. Retrieved 25/36 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1_value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 5
    var_7 = 10
    var_8 = 'value1_content'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = 'key2_value'
    var_14 = module_0.Token(var_10, var_11, var_12, var_13)
    var_15 = 'value2'
    var_16 = 17
    var_17 = 22
    var_18 = 'value2_content'
    var_19 = module_0.Token(var_15, var_16, var_17, var_18)
    var_20 = (var_4, var_9)
    var_21 = (var_14, var_19)
    var_22 = [var_20, var_21]
    var_23 = 25
    var_24 = 'key1value1key2value2'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dict_token_init_creates_child_keys_and_tokens. Retrieved 31/33 statements.


def test_case_0():
    var_0 = 'Token'
    var_1 = ()
    var_2 = '_value'
    var_3 = '_get_value'
    var_4 = 'key1'
    var_5 = lambda self: var_4
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = type(var_0, var_1, var_6)
    var_8 = var_7()
    var_9 = ()
    var_10 = 'value1'
    var_11 = lambda self: var_10
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = type(var_0, var_9, var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = 'key2'
    var_17 = lambda self: var_16
    var_18 = {var_2: var_16, var_3: var_17}
    var_19 = type(var_0, var_15, var_18)
    var_20 = var_19()
    var_21 = ()
    var_22 = 'value2'
    var_23 = lambda self: var_22
    var_24 = {var_2: var_22, var_3: var_23}
    var_25 = type(var_0, var_21, var_24)
    var_26 = var_25()
    var_27 = {var_8: var_14, var_20: var_26}
    var_28 = 0
    var_29 = 10
    var_30 = 'test_content'
    var_31 = [var_27, var_28, var_29, var_30]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_start_index_assignment. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'content'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''



# Parsed testcases at query #38
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 1
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = 3
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = [var_4, var_8]
    var_10 = '[a,b]'
    var_11 = module_0.ListToken(var_9, var_1, var_7, var_10)
    var_12 = var_11._value
    var_13 = bool(var_11._value == [var_4, var_8])
    assert var_13 is True
    var_14 = var_11._start_index
    assert var_14 == 0
    var_15 = var_11._end_index
    assert var_15 == 3
    var_16 = var_11._content
    assert var_16 == '[a,b]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 1
    var_3 = '[]'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == 0
    var_8 = var_4._end_index
    assert var_8 == 1
    var_9 = var_4._content
    assert var_9 == '[]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = [var_2]
    var_4 = module_0.ListToken(var_3, var_1, var_1)
    var_5 = var_4._value
    var_6 = bool(var_4._value == [var_2])
    assert var_6 is True
    var_7 = var_4._start_index
    assert var_7 == 0
    var_8 = var_4._end_index
    assert var_8 == 0
    var_9 = var_4._content
    assert var_9 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = [Token(value=i, start_index=i, end_index=i, content=str(i)) for i in var_1]
    var_3 = 0
    var_4 = 4
    var_5 = '01234'
    var_6 = module_0.ListToken(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 4
    var_11 = var_6._content
    assert var_11 == '01234'



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._value
    assert var_5 == 'test_value'
    var_6 = var_4._start_index
    assert var_6 == 0
    var_7 = var_4._end_index
    assert var_7 == 5
    var_8 = var_4._content
    assert var_8 == 'test_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = var_3._value
    assert var_4 == 42
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
    var_2 = 'abc'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._value
    assert var_4 is None
    var_5 = var_3._start_index
    assert var_5 == 0
    var_6 = var_3._end_index
    assert var_6 == 0
    var_7 = var_3._content
    assert var_7 == 'abc'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = 15
    var_5 = 'some_content'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_2)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 5
    var_10 = var_6._end_index
    assert var_10 == 15
    var_11 = var_6._content
    assert var_11 == 'some_content'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 10
    var_6 = module_0.Token(var_3, var_4, var_5)
    var_7 = var_6._value
    var_8 = bool(var_6._value == var_3)
    assert var_8 is True
    var_9 = var_6._start_index
    assert var_9 == 0
    var_10 = var_6._end_index
    assert var_10 == 10
    var_11 = var_6._content
    assert var_11 == ''

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 'x'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = var_3._start_index
    assert var_4 == 0
    var_5 = var_3._end_index
    assert var_5 == 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1000
    var_2 = 2000
    var_3 = 'a'
    var_4 = 2001
    var_5 = var_3 * var_4
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = var_6._start_index
    assert var_7 == 1000
    var_8 = var_6._end_index
    assert var_8 == 2000



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dict_token_init. Retrieved 17/18 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 10
    var_7 = 'key1value1'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = module_0.Token(var_9, var_1, var_2, var_9)
    var_11 = 'value2'
    var_12 = 'key2value2'
    var_13 = module_0.Token(var_11, var_5, var_6, var_12)
    var_14 = {var_3: var_8, var_10: var_13}
    var_15 = 20
    var_16 = 'key1value1key2value2'
    var_17 = []



