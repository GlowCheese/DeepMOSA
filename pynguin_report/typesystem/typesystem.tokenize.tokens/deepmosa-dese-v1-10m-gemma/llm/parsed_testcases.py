####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 3
    var_3 = '012345'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 5
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 3
    var_3 = '1234'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = 'val'
    var_4 = 5
    var_5 = 8
    var_6 = '{"key": "val"}'
    var_7 = len(var_6)
    var_8 = 1
    var_9 = var_7 - var_8
    var_10 = len(var_6)
    var_11 = var_10 - var_8



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 16/17 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 15/17 statements.


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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = 5
    var_10 = module_0.Token(var_0, var_9, var_9, var_0)
    var_11 = 7
    var_12 = module_0.Token(var_3, var_11, var_11, var_5)
    var_13 = {var_10: var_12}
    var_14 = 8



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dicttoken_init_args_assignment. Retrieved 16/17 statements.


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
    var_14 = 7
    var_15 = '1:a, 2:b'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_attributes_correctly. Retrieved 13/15 statements.


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
    var_8 = {var_0: var_7}
    var_9 = 'name: Alice'
    var_10 = module_0.DictToken()
    var_11 = {var_3: var_7}
    var_12 = module_0.DictToken()

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = module_0.DictToken()
    var_4 = var_3._child_keys
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_3._child_tokens
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 5
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 3
    var_3 = '0test0'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 1
    var_3 = '10'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'other'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = 20
    var_8 = module_0.Token(var_7, var_1, var_2, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_attributes_correctly. Retrieved 17/18 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 12/14 statements.


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
    var_14 = 0
    var_15 = 7
    var_16 = 'a: 1, b: 2'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = module_0.Token(var_0, var_1, var_1, var_0)
    var_10 = module_0.Token(var_3, var_4, var_4, var_5)
    var_11 = {var_9: var_10}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_list_token_get_child_token. Retrieved 11/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.ListToken(var_4, var_1, var_5, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.ListToken(var_4, var_1, var_5, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = []
    var_5 = 1
    var_6 = module_0.ListToken(var_4, var_5, var_5, var_2)
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = '[][]'
    var_10 = module_0.ListToken(var_7, var_1, var_8, var_9)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 23/44 statements.


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
    var_13 = 'key1'
    var_14 = 'val1'
    var_15 = 10
    var_16 = 'key2'
    var_17 = 12
    var_18 = 16
    var_19 = 'val2'
    var_20 = 18
    var_21 = 22
    var_22 = 'key1: val1, key2: val2'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 14/20 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'val1'
    var_4 = 6
    var_5 = 9
    var_6 = 'key2'
    var_7 = 11
    var_8 = 15
    var_9 = 'val2'
    var_10 = 17
    var_11 = 20
    var_12 = {var_0: var_3, var_6: var_9}
    var_13 = 'key1: val1, key2: val2'

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = 'v'
    var_4 = 2
    var_5 = 3
    var_6 = 'k: v'



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 5
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = '1'
    var_4 = module_0.ListToken(var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = [var_5]
    var_7 = '2'
    var_8 = module_0.ListToken(var_6, var_0, var_0, var_7)
    var_9 = [var_4, var_8]
    var_10 = 0
    var_11 = 5
    var_12 = '1, 2'
    var_13 = module_0.ListToken(var_9, var_10, var_11, var_12)



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 16/17 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 14/16 statements.


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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = 5
    var_10 = module_0.Token(var_0, var_9, var_9, var_0)
    var_11 = 7
    var_12 = module_0.Token(var_3, var_11, var_11, var_5)
    var_13 = {var_10: var_12}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 10/14 statements.
# Partially parsed test_dict_token_constructor_empty_dict. Retrieved 3/5 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'val'
    var_5 = 5
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = {var_0: var_7}
    var_9 = 'key: val'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 18/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'k1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 10
    var_5 = 2
    var_6 = 'v1'
    var_7 = module_0.Token(var_4, var_5, var_5, var_6)
    var_8 = 4
    var_9 = 'k2'
    var_10 = module_0.Token(var_5, var_8, var_8, var_9)
    var_11 = 20
    var_12 = 6
    var_13 = 'v2'
    var_14 = module_0.Token(var_11, var_12, var_12, var_13)
    var_15 = {var_3: var_7, var_10: var_14}
    var_16 = 'k1v1k2v2'
    var_17 = 7



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data: 123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = 5
    var_5 = 'abcde'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 16/17 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 13/16 statements.


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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = {var_2: var_6}
    var_10 = {var_2: var_6}
    var_11 = 5
    var_12 = 'a: 1...'



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 15/16 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 9/11 statements.
# Partially parsed test_dict_token_constructor_inequality. Retrieved 10/11 statements.


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
    var_14 = 'a1 b2'

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
    var_8 = 'k1'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a1'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a1'
    var_9 = module_0.Token(var_0, var_1, var_4, var_8)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 14/20 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'val1'
    var_4 = 6
    var_5 = 9
    var_6 = 'key2'
    var_7 = 11
    var_8 = 15
    var_9 = 'val2'
    var_10 = 17
    var_11 = 20
    var_12 = {var_0: var_3, var_6: var_9}
    var_13 = 'key1: val1, key2: val2'



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 4
    var_3 = '012345'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 0
    var_3 = module_0.ListToken(var_0, var_1, var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 24/34 statements.


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
    var_13 = {var_0: var_3}
    var_14 = 'key'
    var_15 = 3
    var_16 = module_0.Token(var_14, var_1, var_15, var_14)
    var_17 = 123
    var_18 = 5
    var_19 = 7
    var_20 = '123'
    var_21 = module_0.Token(var_17, var_18, var_19, var_20)
    var_22 = {var_14: var_21}
    var_23 = 'key: 123'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_token_init_predicate_line_1_false. Retrieved 4/5 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 17/66 statements.


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
    var_13 = 'k'
    var_14 = 'v'
    var_15 = 5
    var_16 = 'content'



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = 'xabcdy'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dict_token_init_initializes_correctly. Retrieved 16/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = 'b'
    var_6 = module_0.Token(var_0, var_4, var_4, var_5)
    var_7 = 4
    var_8 = 'c'
    var_9 = module_0.Token(var_0, var_7, var_7, var_8)
    var_10 = 6
    var_11 = 'd'
    var_12 = module_0.Token(var_0, var_10, var_10, var_11)
    var_13 = {var_3: var_6, var_9: var_12}
    var_14 = 7
    var_15 = 'a: b, c: d'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dict_token_init_logic. Retrieved 7/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'val'
    var_5 = 2
    var_6 = 4



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_maps. Retrieved 16/25 statements.


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
    var_14 = 10
    var_15 = 'content'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'val1'
    var_4 = 6
    var_5 = 9
    var_6 = 'key2'
    var_7 = 11
    var_8 = 15
    var_9 = 'val2'
    var_10 = 17
    var_11 = 20
    var_12 = 'key1: val1, key2: val2'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_values_correctly. Retrieved 18/19 statements.
# Partially parsed test_dict_token_constructor_value_property_returns_dictionary. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 8/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'val1'
    var_5 = 6
    var_6 = 9
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = 'key2'
    var_9 = 11
    var_10 = 15
    var_11 = module_0.Token(var_8, var_9, var_10, var_8)
    var_12 = 'val123'
    var_13 = 17
    var_14 = 22
    var_15 = module_0.Token(var_12, var_13, var_14, var_12)
    var_16 = {var_3: var_7, var_11: var_15}
    var_17 = 'key1: val1, key2: val123'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'

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



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '0123456789'
    var_1 = 2
    var_2 = 5
    var_3 = None
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = "'data'"
    var_1 = None
    var_2 = 0
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = repr(var_4)
    assert var_5 == 'Token("\'data\'")'



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_list_token_get_child_token. Retrieved 8/9 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.ListToken(var_4, var_1, var_5, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 1
    var_3 = '[]'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = '1'
    var_4 = module_0.ListToken(var_1, var_2, var_2, var_3)
    var_5 = [var_4]
    var_6 = '[1]'
    var_7 = module_0.ListToken(var_5, var_2, var_0, var_6)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 'a'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 1
    var_6 = '[a]'
    var_7 = module_0.ListToken(var_4, var_1, var_5, var_6)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 1
    var_3 = '[]'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = []
    var_6 = module_0.ListToken(var_5, var_1, var_2, var_3)
    var_7 = []
    var_8 = module_0.ListToken(var_7, var_2, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = 1
    var_3 = '[]'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ListToken('[]')"



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 16/17 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 10/12 statements.


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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = 4

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = 4



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 10/16 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'b'
    var_3 = 2
    var_4 = 'c'
    var_5 = 4
    var_6 = 'd'
    var_7 = 6
    var_8 = 7
    var_9 = 'a: b, c: d'

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 'v'
    var_3 = 2
    var_4 = 'k: v'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 16/17 statements.


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
    var_14 = 7
    var_15 = '1: a, 2: b'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'a'
    var_5 = 2
    var_6 = module_0.Token(var_4, var_5, var_5, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 16/17 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 9/10 statements.
# Partially parsed test_dict_token_constructor_equality. Retrieved 10/12 statements.


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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.Token(var_3, var_4, var_4, var_5)
    var_7 = {var_2: var_6}
    var_8 = 'a: 1'
    var_9 = {var_2: var_6}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 19/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 123
    var_5 = 5
    var_6 = 8
    var_7 = '123'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'key2'
    var_10 = 9
    var_11 = 13
    var_12 = module_0.Token(var_9, var_10, var_11, var_9)
    var_13 = 'hello'
    var_14 = 14
    var_15 = 19
    var_16 = module_0.Token(var_13, var_14, var_15, var_13)
    var_17 = {var_3: var_8, var_12: var_16}
    var_18 = 'key1123key2hello'



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 0
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_correctly. Retrieved 10/15 statements.
# Partially parsed test_dict_token_constructor_logic. Retrieved 6/24 statements.
# Partially parsed test_dict_token_constructor_value_mapping. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'val1'
    var_4 = 6
    var_5 = 10
    var_6 = 'key2'
    var_7 = 12
    var_8 = 16
    var_9 = 'a'

def test_case_0():
    var_0 = 'v1'
    var_1 = 0
    var_2 = 1
    var_3 = 'k1'
    var_4 = 5
    var_5 = 'k1:v1'

def test_case_0():
    var_0 = 'name'
    var_1 = 'value'
    var_2 = 0
    var_3 = 10
    var_4 = 'name: value'



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.ListToken(var_5, var_1, var_1, var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 15/31 statements.
# Partially parsed test_dict_token_string_property. Retrieved 6/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'a'
    var_5 = 1
    var_6 = 2
    var_7 = '1'
    var_8 = 'b'
    var_9 = 4
    var_10 = 5
    var_11 = 6
    var_12 = 7
    var_13 = '2'
    var_14 = 'a1b2'

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 1
    var_3 = 'abc'
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dict_token_init_initializes_internal_mappings. Retrieved 16/17 statements.


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
    var_14 = 7
    var_15 = '1:a, 2:b'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dict_token_initialization_calls_super. Retrieved 16/29 statements.


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
    var_14 = 10
    var_15 = 'content'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_listtoken_constructor_with_child_tokens. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = '[]'
    var_2 = 0
    var_3 = 1
    var_4 = module_0.ListToken(var_0, var_2, var_3, var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 10
    var_4 = module_0.ListToken(var_1, var_2, var_3, var_0)
    var_5 = [var_4]
    var_6 = 0
    var_7 = '[inner]'
    var_8 = module_0.ListToken(var_5, var_6, var_3, var_7)
    var_9 = var_8._value[var_6]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 3
    var_3 = 'abcde'
    var_4 = module_0.ListToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = '1'
    var_4 = module_0.ListToken(var_1, var_2, var_0, var_3)
    var_5 = [var_0]
    var_6 = module_0.ListToken(var_5, var_2, var_0, var_3)
    var_7 = 2
    var_8 = [var_7]
    var_9 = '2'
    var_10 = module_0.ListToken(var_8, var_2, var_0, var_9)



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '[]'
    var_3 = module_0.ListToken(var_0, var_1, var_1, var_2)



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dict_token_constructor_initialization. Retrieved 17/23 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'v1'
    var_3 = 2
    var_4 = 3
    var_5 = 'b'
    var_6 = 5
    var_7 = 'v2'
    var_8 = 7
    var_9 = 8
    var_10 = {var_0: var_2, var_5: var_7}
    var_11 = '{"a": "v1", "b": "v2"}'
    var_12 = len(var_11)
    var_13 = 1
    var_14 = var_12 - var_13
    var_15 = len(var_11)
    var_16 = var_15 - var_13



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 2
    var_3 = '012345'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "Token('test')"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dict_token_init_initializes_internal_mappings. Retrieved 10/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = 'a: 1'
    var_6 = 'b'
    var_7 = 4
    var_8 = 'a: 1, b: 2'
    var_9 = 6



# Parsed testcases at query #24
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
    var_16 = module_0.DictToken()



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data_123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = '012345'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = '10'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)
    var_7 = 20
    var_8 = module_0.Token(var_7, var_1, var_2, var_3)



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #34
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = '0abc4'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 10
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #38
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #40
#--------------------------




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
    var_14 = module_0.DictToken()



# Parsed testcases at query #41
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data_123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dict_token_init_initializes_correctly. Retrieved 15/31 statements.


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
    var_9 = 9
    var_10 = 12
    var_11 = 'val2'
    var_12 = 14
    var_13 = 17
    var_14 = 'key1: val1, key2: val2'



# Parsed testcases at query #43
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abcde'
    var_1 = 1
    var_2 = 3
    var_3 = 'val'
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 0
    var_2 = 3
    var_3 = 'val'
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)
    var_5 = repr(var_4)
    assert var_5 == "Token('data')"



# Parsed testcases at query #44
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello'
    var_4 = module_0.Token(var_3, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 10
    var_2 = 0
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = repr(var_4)
    assert var_5 == "Token('data')"

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'different'
    var_6 = module_0.Token(var_0, var_1, var_2, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 20
    var_6 = module_0.Token(var_5, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #45
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = 'abcde fghij'
    var_3 = module_0.Token(var_0, var_1, var_0, var_2)



# Parsed testcases at query #46
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data_val'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #47
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'data_value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #48
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 5
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = '[]'
    var_5 = module_0.Token(var_2, var_3, var_3, var_4)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_dict_token_init_initializes_child_maps. Retrieved 17/26 statements.


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
    var_8 = 'k2'
    var_9 = module_0.Token(var_0, var_1, var_1, var_8)
    var_10 = 'key2'
    var_11 = 'v2'
    var_12 = module_0.Token(var_0, var_1, var_1, var_11)
    var_13 = 'val2'
    var_14 = {var_3: var_6, var_9: var_12}
    var_15 = 5
    var_16 = 'k1:v1, k2:v2'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_attributes_correctly. Retrieved 8/12 statements.
# Partially parsed test_dict_token_constructor_string_property. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = 'val'
    var_4 = 5
    var_5 = 8
    var_6 = '{"key": "val"}'
    var_7 = 13

def test_case_0():
    var_0 = 'k'
    var_1 = 0
    var_2 = 1
    var_3 = 'v'
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = '{"k": "v"}'



# Parsed testcases at query #51
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #52
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #53
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = 'hello'
    var_2 = 0
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = "'1'"
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = repr(var_3)
    assert var_4 == 'Token("\'1\'")'



# Parsed testcases at query #54
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #55
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = '123456'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #56
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 5
    var_3 = 'abcde'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



