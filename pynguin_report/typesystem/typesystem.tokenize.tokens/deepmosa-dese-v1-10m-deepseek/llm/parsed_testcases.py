####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 10
    var_3 = 'example_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


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
    var_9 = '{key: value}'



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
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = module_0.DictToken()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = 'key1:value1'



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 9
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = 'abc'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = []
    var_2 = 1
    var_3 = 5
    var_4 = module_0.ListToken(var_1, var_2, var_3, var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 10
    var_2 = 20
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 9
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_child_tokens. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_sets_content_properties. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_inherits_from_token. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example_value'
    var_1 = 0
    var_2 = 12
    var_3 = 'example_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example_value'
    var_1 = 0
    var_2 = 12
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #14
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
    var_10 = module_0.DictToken()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


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
    var_9 = 'key value'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_child_keys_and_tokens. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_inherits_token_properties. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_handles_empty_dict. Retrieved 3/4 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_initialization_with_non_integer_start_index. Retrieved 6/7 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'not_an_int'
    var_2 = 10
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = var_4._start_index



# Parsed testcases at query #18
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
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_1_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = '_child_keys'
    var_4 = '_child_tokens'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 9
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = 'example'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 123
    var_2 = 0
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 456
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello world'
    var_1 = None
    var_2 = 6
    var_3 = 10
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = module_0.Token(var_1, var_2, var_3, var_0)
    var_6 = 2
    var_7 = module_0.Token(var_6, var_2, var_3, var_0)



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 2
    var_6 = '123'
    var_7 = module_0.ListToken(var_3, var_4, var_5, var_6)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dict_token_constructor_initializes_child_tokens. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_sets_content_and_indices. Retrieved 10/11 statements.
# Partially parsed test_dict_token_constructor_inherits_from_token. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 9
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 10
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 1
    var_6 = module_0.Token(var_0, var_5, var_2, var_3)



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'test_content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_init_creates_child_keys_and_tokens. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = 'value'
    var_5 = 4
    var_6 = 8
    var_7 = module_0.Token(var_4, var_5, var_6)
    var_8 = {var_3: var_7}
    var_9 = 'key: value'



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 456
    var_1 = 1
    var_2 = 3
    var_3 = 'abcdef'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #32
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
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 3
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 10/11 statements.


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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_predicate_false. Retrieved 11/12 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = module_0.Token(var_4, var_0, var_0, var_2)
    var_6 = {var_3: var_5}
    var_7 = module_0.Token(var_0, var_1, var_1, var_2)
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_4, var_0, var_0, var_2)
    var_10 = {var_0: var_9}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 13



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = 2
    var_5 = module_0.Token(var_0, var_4, var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 4
    var_3 = 'value1'
    var_4 = 5
    var_5 = 10



# Parsed testcases at query #39
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = 'content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value2'
    var_6 = 1
    var_7 = 6
    var_8 = 'different content'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)



# Parsed testcases at query #40
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 2
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 13



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 10/11 statements.


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
    var_9 = 'keyvalue'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value2'
    var_5 = module_0.Token(var_4, var_1, var_2, var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 1
    var_5 = module_0.Token(var_0, var_4, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 5
    var_5 = module_0.Token(var_0, var_1, var_4, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value'



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test2'
    var_5 = module_0.Token(var_4, var_1, var_2, var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 1
    var_5 = module_0.Token(var_0, var_4, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 4
    var_5 = module_0.Token(var_0, var_1, var_4, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'test'



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 0
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 42
    var_2 = 0
    var_3 = 4
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)



# Parsed testcases at query #7
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
    var_10 = module_0.DictToken()



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example content'
    var_1 = 42
    var_2 = 0
    var_3 = 6
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = 5
    var_3 = 'example'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dict_token_constructor_simple. Retrieved 13/14 statements.
# Partially parsed test_dict_token_constructor_multiple_entries. Retrieved 21/22 statements.


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
    var_9 = 6
    var_10 = 10
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key1'
    var_6 = 4
    var_7 = module_0.Token(var_5, var_3, var_6, var_0)
    var_8 = 'value1'
    var_9 = 7
    var_10 = 12
    var_11 = module_0.Token(var_8, var_9, var_10, var_0)
    var_12 = 'key2'
    var_13 = 15
    var_14 = 18
    var_15 = module_0.Token(var_12, var_13, var_14, var_0)
    var_16 = 'value2'
    var_17 = 21
    var_18 = 26
    var_19 = module_0.Token(var_16, var_17, var_18, var_0)
    var_20 = {var_7: var_11, var_15: var_19}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 11/12 statements.


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
    var_10 = 9



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 8/9 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = len(var_0)
    var_6 = 1
    var_7 = var_5 - var_6



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 28/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1: value1, key2: value2'
    var_1 = 0
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 0
    var_8 = 3
    var_9 = module_0.Token(var_5, var_7, var_8, var_0)
    var_10 = 14
    var_11 = 17
    var_12 = module_0.Token(var_6, var_10, var_11, var_0)
    var_13 = {var_5: var_9, var_6: var_12}
    var_14 = 'value1'
    var_15 = 6
    var_16 = 11
    var_17 = module_0.Token(var_14, var_15, var_16, var_0)
    var_18 = 'value2'
    var_19 = 20
    var_20 = 25
    var_21 = module_0.Token(var_18, var_19, var_20, var_0)
    var_22 = {var_5: var_17, var_6: var_21}
    var_23 = var_13[var_5]
    var_24 = var_13[var_6]
    var_25 = var_22[var_5]
    var_26 = var_22[var_6]
    var_27 = {var_23: var_25, var_24: var_26}



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 8
    var_3 = 'prefix_test_suffix'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'example'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dict_token_initialization. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "value"}'



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 0
    var_2 = 6
    var_3 = 'example'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 0
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 3
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dict_token_init_with_empty_value. Retrieved 3/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "value"}'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


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



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'example content'
    var_1 = 'example'
    var_2 = 0
    var_3 = 6
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = module_0.DictToken()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'value1'
    var_5 = 5
    var_6 = 11
    var_7 = module_0.Token(var_4, var_5, var_6, var_4)
    var_8 = {var_3: var_7}
    var_9 = 'key1:value1'
    var_10 = module_0.Token(var_0, var_1, var_2, var_0)
    var_11 = {var_0: var_10}
    var_12 = module_0.Token(var_4, var_5, var_6, var_4)
    var_13 = {var_0: var_12}



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = module_0.Token(var_0, var_1, var_2, var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 7
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.Token(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 12



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'sample'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 3
    var_2 = 7
    var_3 = module_0.Token(var_0, var_1, var_2)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'not_a_dict'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.DictToken()



# Parsed testcases at query #34
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
    var_10 = module_0.DictToken()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_token_get_position_returns_correct_position. Retrieved 5/6 statements.
# Partially parsed test_token_get_position_with_empty_content. Retrieved 4/5 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = 'test content'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = module_0.Token(var_0, var_1, var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 5
    var_2 = 9
    var_3 = 'hello world'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = 20
    var_3 = 'test'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_0.Token(var_0, var_1, var_2, var_3)
    var_6 = 99
    var_7 = module_0.Token(var_6, var_1, var_2, var_3)
    var_8 = 11
    var_9 = module_0.Token(var_0, var_8, var_2, var_3)
    var_10 = 21
    var_11 = module_0.Token(var_0, var_1, var_10, var_3)

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
    var_0 = None
    var_1 = 0
    var_2 = 'line1\nline2\nline3'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 8

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 5
    var_3 = 'hello'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dict_token_constructor. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 4
    var_7 = 8
    var_8 = module_0.Token(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}



# Parsed testcases at query #38
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)



