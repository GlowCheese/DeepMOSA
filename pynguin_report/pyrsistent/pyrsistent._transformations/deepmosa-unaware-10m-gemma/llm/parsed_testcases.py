####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-z]+@domain\\.com$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@domain.com'
    var_12 = 'USER@domain.com'
    var_13 = 'user@domain.org'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = '[^@]+@[^@]+\\.[^@]+'
    var_18 = module_0.rex(var_17)
    var_19 = 'user@example.com'
    var_20 = 'invalid-email'
    var_21 = '^exact$'
    var_22 = module_0.rex(var_21)
    var_23 = 'exact'
    var_24 = 'exact_extra'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^[0-9]+$'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = 'abc'
    var_9 = ''
    var_10 = '^pre'
    var_11 = module_0.rex(var_10)
    var_12 = 'prefix'
    var_13 = 'pre'
    var_14 = 'suffix'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = 'a'
    var_20 = [var_19]
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ' '



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'not_exact'
    var_19 = '^[a-z]+_\\d{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'test_01'
    var_22 = 'test_1'
    var_23 = 'TEST_01'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^\\d+$'
    var_6 = module_0.rex(var_5)
    var_7 = ''
    var_8 = '^pre'
    var_9 = module_0.rex(var_8)
    var_10 = 'prefix'
    var_11 = 'pre'
    var_12 = 'aprefix'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '^[a-z]+\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc12'
    var_23 = 'abc1'
    var_24 = 'ABC12'
    var_25 = '12abc'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre'
    var_7 = module_0.rex(var_6)
    var_8 = 'prefix'
    var_9 = 'pre'
    var_10 = 'apple'
    var_11 = 123
    var_12 = None
    var_13 = [var_2]
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '
    var_18 = '^[A-Z]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'HELLO'
    var_21 = 'hello'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 're'
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'a'
    var_19 = [var_18]
    var_20 = '[a-z]+'
    var_21 = module_0.rex(var_20)
    var_22 = 'ABC'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = '^[a-z]+@example\\.com$'
    var_18 = module_0.rex(var_17)
    var_19 = 'user@example.com'
    var_20 = 'user@other.com'
    var_21 = 'USER@example.com'
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ' '



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = 123
    var_6 = '^test_'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_user'
    var_9 = 'prod_user'
    var_10 = 'exact_match'
    var_11 = module_0.rex(var_10)
    var_12 = 'exact_match_extra'
    var_13 = '[A-Z]+'
    var_14 = module_0.rex(var_13)
    var_15 = 'HELLO'
    var_16 = 'hello'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = None



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '45'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^test_'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_case'
    var_12 = 'testing'
    var_13 = 'not_test'
    var_14 = '^exact$'
    var_15 = module_0.rex(var_14)
    var_16 = 'exact'
    var_17 = 'exact_extra'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^user_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user_01'
    var_12 = 'user_99'
    var_13 = 'user_1'
    var_14 = 'admin_01'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '
    var_19 = '^[A-Z]+$'
    var_20 = module_0.rex(var_19)
    var_21 = 'HELLO'
    var_22 = 'hello'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^[a-z]+_[0-9]{2}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'abc_12'
    var_18 = 'abc_1'
    var_19 = 'ABC_12'
    var_20 = 'abc_ab'
    var_21 = 'anything'
    var_22 = module_0.ny(var_21)
    assert var_22 is True
    var_23 = module_0.ny(var_12)
    assert var_23 is True
    var_24 = module_0.ny(var_11)
    assert var_24 is True



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = ''
    var_10 = '[a-z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello'
    var_13 = 'Hello'
    var_14 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.user@example.com'
    var_17 = 'invalid-email'
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = None
    var_22 = True



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^test_'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_case'
    var_9 = 't_case'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 123
    var_15 = None
    var_16 = [var_2]
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = ' '



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = 'pre'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = 123
    var_14 = None
    var_15 = 'list'
    var_16 = [var_15]
    var_17 = '^exact$'
    var_18 = module_0.rex(var_17)
    var_19 = 'exact'
    var_20 = 'not_exact'
    var_21 = '^[A-Z]+$'
    var_22 = module_0.rex(var_21)
    var_23 = 'HELLO'
    var_24 = 'hello'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[^@]+@[^@]+\\.[^@]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'invalid-email'
    var_13 = '^exact$'
    var_14 = module_0.rex(var_13)
    var_15 = 'exact'
    var_16 = 'not exact'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^\\d+$'
    var_6 = module_0.rex(var_5)
    var_7 = ''
    var_8 = '^pre'
    var_9 = module_0.rex(var_8)
    var_10 = 'prefix'
    var_11 = 'pre'
    var_12 = 'post'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = [var_2]
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ' '



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^test_'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_case'
    var_9 = 'testing'
    var_10 = 'pre_test'
    var_11 = 123
    var_12 = None
    var_13 = [var_2]
    var_14 = '^[a-z]+_[0-9]{2}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'abc_12'
    var_17 = 'abc_1'
    var_18 = 'ABC_12'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = 123
    var_10 = None
    var_11 = [var_2]
    var_12 = '^exact$'
    var_13 = module_0.rex(var_12)
    var_14 = 'exact'
    var_15 = 'exac'
    var_16 = '^[a-z]+_\\d{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test_01'
    var_19 = 'TEST_01'
    var_20 = 'abc_1'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = None
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '123a'
    var_10 = ''
    var_11 = '.+@.+\\..+'
    var_12 = module_0.rex(var_11)
    var_13 = 'test@example.com'
    var_14 = 'invalid-email'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ' '
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = callable(var_19)



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = '123a'
    var_5 = ''
    var_6 = '^test_.*'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_function'
    var_9 = 'test_123'
    var_10 = 'function_test'
    var_11 = 'ABC'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc'
    var_14 = 123
    var_15 = None
    var_16 = [var_2]
    var_17 = '^[a-z]+_\\d{2}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abc_12'
    var_20 = 'abc_1'
    var_21 = 'ABC_12'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'reprefix'
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = 'a'
    var_18 = [var_17]
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ''
    var_22 = ' '
    var_23 = '[a-z]+[0-9]'
    var_24 = module_0.rex(var_23)
    var_25 = 'abc1'
    var_26 = 'ABC1'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre_.*'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'post_test'
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'HELLO'
    var_13 = 'hello'
    var_14 = 123
    var_15 = None
    var_16 = [var_2]
    var_17 = '^[a-z]{3}-\\d{2}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abc-12'
    var_20 = 'abcd-12'
    var_21 = 'abc-1'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[a-z]+@[a-z]+\\.com$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'user@domain.net'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abc '
    var_5 = '^[0-9]+'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = '123abc'
    var_9 = 'a123'
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello'
    var_13 = 'Hello'
    var_14 = 'h1'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = 123
    var_19 = None
    var_20 = 'a'
    var_21 = [var_20]
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ''
    var_25 = ' '



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'string'
    var_14 = [var_13]
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'exact_extra'
    var_19 = '^[a-z]+_\\d{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'item_01'
    var_22 = 'ITEM_01'
    var_23 = 'item_1'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = 123
    var_5 = None
    var_6 = '^pre'
    var_7 = module_0.rex(var_6)
    var_8 = 'prefix'
    var_9 = 'pre'
    var_10 = 'append'
    var_11 = '^\\d+$'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = '123a'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'
    var_19 = 'list'
    var_20 = [var_19]
    var_21 = 'key'
    var_22 = 'val'
    var_23 = {var_21: var_22}
    var_24 = True
    var_25 = '^[a-z]+_[0-9]{2}$'
    var_26 = module_0.rex(var_25)
    var_27 = 'abc_12'
    var_28 = 'ABC_12'
    var_29 = 'abc_1'
    var_30 = 'abc_123'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-z]+_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_01'
    var_12 = 'abc_99'
    var_13 = 'TEST_01'
    var_14 = 'test_1'
    var_15 = 'test_ab'
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'anything'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre'
    var_7 = module_0.rex(var_6)
    var_8 = 'prefix'
    var_9 = 'pre'
    var_10 = 'apre'
    var_11 = 'Hello'
    var_12 = module_0.rex(var_11)
    var_13 = 'hello'
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 123
    var_17 = module_0.rex(var_14)
    var_18 = None
    var_19 = module_0.rex(var_14)
    var_20 = []
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ''
    var_24 = ' '
    var_25 = '^[a-z]+_\\d{2}$'
    var_26 = module_0.rex(var_25)
    var_27 = 'test_01'
    var_28 = 'TEST_01'
    var_29 = 'abc_1'
    var_30 = 'abc_123'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'not_exact'
    var_19 = '^[a-z]+_\\d{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'hello_99'
    var_22 = 'hello_9'
    var_23 = 'HELLO_99'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 123
    var_3 = None
    var_4 = []
    var_5 = '[a-z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc'
    var_8 = 'python'
    var_9 = 'ABC'
    var_10 = '123'
    var_11 = '^fixed$'
    var_12 = module_0.rex(var_11)
    var_13 = 'fixed'
    var_14 = 'not fixed'
    var_15 = 'fixedly'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^[0-9]+$'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = 'abc'
    var_9 = ''
    var_10 = 'test'
    var_11 = module_0.rex(var_10)
    var_12 = 'testing'
    var_13 = 'great test day'
    var_14 = 't'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = [var_2]
    var_20 = True
    var_21 = module_0.rex(var_15)
    var_22 = 'anything'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '^[a-z]+_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user_01'
    var_23 = 'user_1'
    var_24 = 'User_01'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'aprefix'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '12a'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = None
    var_18 = 123
    var_19 = 'a'
    var_20 = 'b'
    var_21 = [var_19, var_20]
    var_22 = '^[a-z]+_\\d{2}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test_01'
    var_25 = 'TEST_01'
    var_26 = 'test_1'
    var_27 = 'test_abc'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = 123
    var_6 = None
    var_7 = [var_2]
    var_8 = '^pre'
    var_9 = module_0.rex(var_8)
    var_10 = 'prefix'
    var_11 = 're'
    var_12 = '^exact$'
    var_13 = module_0.rex(var_12)
    var_14 = 'exact'
    var_15 = 'exactness'
    var_16 = '\\b[A-Z][a-z]\\b'
    var_17 = module_0.rex(var_16)
    var_18 = 'Ab'
    var_19 = 'Abc'
    var_20 = 'aB'
    var_21 = '.*'
    var_22 = module_0.rex(var_21)
    var_23 = 'anything'



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = 123
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'post_test'
    var_10 = None
    var_11 = '^[A-Z]+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'HELLO'
    var_14 = 'Hello'
    var_15 = '^\\w+@\\w+\\.com$'
    var_16 = module_0.rex(var_15)
    var_17 = 'user@domain.com'
    var_18 = 'user@domain.org'
    var_19 = '!@#$'
    var_20 = '^$'
    var_21 = module_0.rex(var_20)
    var_22 = ' '



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^[a-z]+_\\d{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'hello_99'
    var_19 = 'hello_9'
    var_20 = 'HELLO_99'



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'string'
    var_14 = [var_13]
    var_15 = module_0.rex(var_3)
    var_16 = 'ABC'
    var_17 = '^[a-z]+_\\d{2}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'hello_99'
    var_20 = 'hello_9'
    var_21 = 'HELLO_99'
    var_22 = 'abc_def'



# Parsed testcases at query #39
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'post_data'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '^[a-z]{3}-\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc-12'
    var_23 = 'abcd-12'
    var_24 = 'abc-1'



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'post_test'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '^[a-z]+_\\d{2}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello_99'
    var_16 = 'hello_9'
    var_17 = 'HELLO_99'



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = 'hi'
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = 'pre'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = 'string'
    var_20 = [var_19]
    var_21 = True
    var_22 = 'ABC'
    var_23 = module_0.rex(var_22)
    var_24 = 'abc'
    var_25 = '^[a-z]+[0-9]$'
    var_26 = module_0.rex(var_25)
    var_27 = 'abc1'



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '[a-z]+@[a-z]+\\.com'
    var_16 = module_0.rex(var_15)
    var_17 = 'test@example.com'
    var_18 = 'TEST@example.com'
    var_19 = 'test@example.net'
    var_20 = '^exact$'
    var_21 = module_0.rex(var_20)
    var_22 = 'exact'
    var_23 = 'ex'



# Parsed testcases at query #43
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = True
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'not_exact'
    var_20 = '^[a-z]+_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'item_01'
    var_23 = 'ITEM_01'
    var_24 = 'item_1'
    var_25 = 'item_abc'



# Parsed testcases at query #44
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^test_'
    var_6 = module_0.rex(var_5)
    var_7 = 'test_case'
    var_8 = 'example_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = []
    var_15 = '^[a-z]+_\\d{2}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'data_01'
    var_18 = 'data_1'
    var_19 = 'DATA_01'
    var_20 = 'abc_def'
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ' '



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^test_'
    var_6 = module_0.rex(var_5)
    var_7 = 'test_item'
    var_8 = 'item_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = '^[a-z]+_\\d{2}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'data_01'
    var_20 = 'DATA_01'
    var_21 = 'data_1'
    var_22 = 'data_abc'
    var_23 = '^$'
    var_24 = module_0.rex(var_23)
    var_25 = ' '



# Parsed testcases at query #46
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = 'pre_.*'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'exact_extra'
    var_19 = module_0.rex(var_3)
    var_20 = 'ABC'



# Parsed testcases at query #47
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = '^[a-z]+$'
    var_5 = module_0.rex(var_4)
    var_6 = 'abc'
    var_7 = 'abc1'
    var_8 = 123
    var_9 = None
    var_10 = [var_2]
    var_11 = '[^@]+@[^@]+\\.[^@]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'test@example.com'
    var_14 = 'invalid-email'
    var_15 = '.+'
    var_16 = module_0.rex(var_15)
    var_17 = 'a'
    var_18 = ''



# Parsed testcases at query #48
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^[a-z]+_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_01'
    var_12 = 'abc_99'
    var_13 = 'TEST_01'
    var_14 = 'test_1'
    var_15 = 'test_aa'
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'anything'



# Parsed testcases at query #49
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = 'pre'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = 123
    var_14 = None
    var_15 = 'list'
    var_16 = [var_15]
    var_17 = '^exact_string$'
    var_18 = module_0.rex(var_17)
    var_19 = 'exact_string'
    var_20 = 'exact_string_extra'
    var_21 = '^[A-Z][a-z]+$'
    var_22 = module_0.rex(var_21)
    var_23 = 'Hello'
    var_24 = 'hello'
    var_25 = 'HELL'



# Parsed testcases at query #50
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'test_pre'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = '^[a-z]+_\\d{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'abc_12'
    var_19 = 'abc_1'
    var_20 = 'ABC_12'



# Parsed testcases at query #51
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = 'hi'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'apre'
    var_10 = 'ing$'
    var_11 = module_0.rex(var_10)
    var_12 = 'running'
    var_13 = 'sing'
    var_14 = 'singers'
    var_15 = '^\\d+$'
    var_16 = module_0.rex(var_15)
    var_17 = '123'
    var_18 = '12a'
    var_19 = ''
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = 123
    var_23 = None
    var_24 = 'a'
    var_25 = [var_24]
    var_26 = '^$'
    var_27 = module_0.rex(var_26)
    var_28 = ' '



# Parsed testcases at query #52
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^\\d+$'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = 'abc'
    var_9 = ''
    var_10 = 'cat'
    var_11 = module_0.rex(var_10)
    var_12 = 'category'
    var_13 = 'scat'
    var_14 = 'dog'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = [var_2]



# Parsed testcases at query #53
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'suffix'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '12a'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'a'
    var_19 = [var_18]
    var_20 = '^[a-z]+\\d$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc1'
    var_23 = 'ABC1'
    var_24 = '^$'
    var_25 = module_0.rex(var_24)
    var_26 = ''
    var_27 = ' '



# Parsed testcases at query #54
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'not_exact'
    var_19 = '^[A-Z]+$'
    var_20 = module_0.rex(var_19)
    var_21 = 'HELLO'
    var_22 = 'hello'
    var_23 = '^[a-z]{2}-\\d{3}$'
    var_24 = module_0.rex(var_23)
    var_25 = 'ab-123'
    var_26 = 'abc-123'
    var_27 = 'ab-12'



# Parsed testcases at query #55
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^test_'
    var_6 = module_0.rex(var_5)
    var_7 = 'test_case'
    var_8 = 'sample_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'exact_suffix'
    var_20 = '^[a-z]+$'
    var_21 = module_0.rex(var_20)
    var_22 = 'ABC'



# Parsed testcases at query #56
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^test_'
    var_6 = module_0.rex(var_5)
    var_7 = 'test_function'
    var_8 = 'function_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^[a-z]+@[a-z]+\\.com$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@gmail.com'
    var_19 = 'user@gmail.net'
    var_20 = 'User@gmail.com'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^[A-Z]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'HELLO'
    var_18 = 'hello'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ' '



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = 123
    var_6 = '^test_'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_case'
    var_9 = 'testing'
    var_10 = 'not_test'
    var_11 = '\\buser\\d\\b'
    var_12 = module_0.rex(var_11)
    var_13 = 'user1'
    var_14 = 'user2'
    var_15 = 'users1'
    var_16 = 'myuser1'
    var_17 = '^[A-Z]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'HELLO'
    var_20 = 'Hello'
    var_21 = '.*'
    var_22 = module_0.rex(var_21)
    var_23 = 'anything'
    var_24 = None



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'not_exact'
    var_20 = '^[A-Z]+$'
    var_21 = module_0.rex(var_20)
    var_22 = 'HELLO'
    var_23 = 'hello'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'ex'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'string'
    var_14 = [var_13]
    var_15 = '[a-z]+'
    var_16 = module_0.rex(var_15)
    var_17 = 'hello'
    var_18 = 'Hello'
    var_19 = '^exact$'
    var_20 = module_0.rex(var_19)
    var_21 = 'exact'
    var_22 = 'not_exact'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^a.*'
    var_7 = module_0.rex(var_6)
    var_8 = 'apple'
    var_9 = 'banana'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '[^@]+@[^@]+\\.[^@]+'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid-email'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '\\buser_\\d{2}\\b'
    var_21 = module_0.rex(var_20)
    var_22 = 'user_01'
    var_23 = 'user_1'
    var_24 = 'my_user_01_data'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'ab'
    var_5 = '^[0-9]+$'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = ''
    var_9 = '^pre'
    var_10 = module_0.rex(var_9)
    var_11 = 'prefix'
    var_12 = 'suffix'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = 'test'
    var_18 = [var_17]
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ' '
    var_22 = '^[a-z]+\\d{2}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test12'
    var_25 = 'TEST12'
    var_26 = 'abc1'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_data'
    var_8 = 'data_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = '^hello$'
    var_16 = module_0.rex(var_15)
    var_17 = 'hello'
    var_18 = 'hello world'
    var_19 = '^[A-Z]+$'
    var_20 = module_0.rex(var_19)
    var_21 = 'ABC'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'preheat'
    var_9 = 'post'
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = 123
    var_19 = None
    var_20 = 'a'
    var_21 = [var_20]
    var_22 = '^[A-Z][a-z]+$'
    var_23 = module_0.rex(var_22)
    var_24 = 'Hello'
    var_25 = 'hello'
    var_26 = 'HELLO'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^pre_.*'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'post_test'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '^[a-z]+_\\d{2}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello_99'
    var_16 = 'hello_9'
    var_17 = 'HELLO_99'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'not_exact'
    var_20 = '^[a-z]+_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc_12'
    var_23 = 'abc_1'
    var_24 = 'ABC_12'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = 123
    var_10 = None
    var_11 = [var_2]
    var_12 = 'exact'
    var_13 = module_0.rex(var_12)
    var_14 = 'exact_match'
    var_15 = '[a-z]+'
    var_16 = module_0.rex(var_15)
    var_17 = 'ABC'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'a'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'not_exact'
    var_20 = '\\b[A-Z]{3}\\b'
    var_21 = module_0.rex(var_20)
    var_22 = 'ABC'
    var_23 = 'abcd'
    var_24 = 'AB'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^user_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'user_123'
    var_3 = 'user_0'
    var_4 = 'user_abc'
    var_5 = 'admin_123'
    var_6 = 'user_123_extra'
    var_7 = 123
    var_8 = None
    var_9 = [var_2]
    var_10 = '^fixed$'
    var_11 = module_0.rex(var_10)
    var_12 = 'fixed'
    var_13 = 'fixed_suffix'
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'anything'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = None
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '12345'
    var_9 = '123a45'
    var_10 = ''
    var_11 = '\\bcat\\b'
    var_12 = module_0.rex(var_11)
    var_13 = 'the cat sat'
    var_14 = 'category'
    var_15 = 'concatenate'
    var_16 = '^[aeiou]$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a'
    var_19 = 'e'
    var_20 = 'b'
    var_21 = '.*'
    var_22 = module_0.rex(var_21)
    var_23 = module_0.rex(var_21)
    var_24 = 'anything'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 10
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = (var_3, var_5)
    var_7 = [var_6]
    var_8 = module_0.transform(var_0, var_7)
    assert var_8 == 20
    var_9 = 'a'
    var_10 = 'b'
    var_11 = 1
    var_12 = {var_9: var_11, var_10: var_4}
    var_13 = [var_9]
    var_14 = {var_9: var_4, var_10: var_4}
    var_15 = 'users'
    var_16 = 'alice'
    var_17 = 'bob'
    var_18 = 'age'
    var_19 = 25
    var_20 = {var_18: var_19}
    var_21 = 30
    var_22 = {var_18: var_21}
    var_23 = {var_16: var_20, var_17: var_22}
    var_24 = [var_15, var_16, var_18]
    var_25 = 26
    var_26 = {var_18: var_25}
    var_27 = {var_18: var_21}
    var_28 = 'user_1'
    var_29 = 'user_2'
    var_30 = 'admin'
    var_31 = 20
    var_32 = 50
    var_33 = {var_28: var_0, var_29: var_31, var_30: var_32}
    var_34 = 'user_.*'
    var_35 = module_0.rex(var_34)
    var_36 = [var_35]
    var_37 = 11
    var_38 = 21
    var_39 = {var_28: var_37, var_29: var_38, var_30: var_32}
    var_40 = {var_9: var_11, var_10: var_4}
    var_41 = 3
    var_42 = {var_9: var_4, var_10: var_41}
    var_43 = 'c'
    var_44 = {var_9: var_11, var_10: var_4, var_43: var_41}
    var_45 = lambda k, v: k == var_10
    var_46 = {var_9: var_11, var_43: var_41}
    var_47 = {var_10: var_4, var_43: var_41}
    var_48 = {var_9: var_0, var_10: var_31, var_43: var_21}
    var_49 = 15
    var_50 = lambda k, v: v > var_49
    var_51 = [var_50]
    var_52 = 31
    var_53 = {var_9: var_0, var_10: var_38, var_43: var_52}
    var_54 = 'item_1'
    var_55 = 'item_2'
    var_56 = 'other'
    var_57 = {var_54: var_11, var_55: var_4, var_56: var_41}
    var_58 = 'item'
    var_59 = lambda k: k.startswith(var_58)
    var_60 = [var_59]
    var_61 = {var_54: var_4, var_55: var_41, var_56: var_41}
    var_62 = {var_9: var_11}
    var_63 = 'new_key'
    var_64 = 'sub_key'
    var_65 = [var_63, var_64]
    var_66 = 100
    var_67 = lambda x: var_66
    var_68 = [var_65, var_67]
    var_69 = {var_64: var_66}
    var_70 = [var_11]
    var_71 = 'vals'
    var_72 = 'x'
    var_73 = 'y'
    var_74 = {var_72: var_11, var_73: var_4}
    var_75 = 'z'
    var_76 = 5
    var_77 = lambda _: var_76
    var_78 = [var_71, var_75, var_77]
    var_79 = {var_72: var_4, var_73: var_11, var_75: var_76}
    var_80 = {var_9: var_11}
    var_81 = 'non_existent'
    var_82 = {var_9: var_11}



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 10.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = {var_0: var_3, var_1: var_3}
    var_7 = 'users'
    var_8 = 'alice'
    var_9 = 'bob'
    var_10 = 'age'
    var_11 = 25
    var_12 = {var_10: var_11}
    var_13 = 30
    var_14 = {var_10: var_13}
    var_15 = {var_8: var_12, var_9: var_14}
    var_16 = [var_7, var_8, var_10]
    var_17 = 26
    var_18 = {var_10: var_17}
    var_19 = {var_10: var_13}
    var_20 = {var_8: var_18, var_9: var_19}
    var_21 = 'user_1'
    var_22 = 'user_2'
    var_23 = 'admin'
    var_24 = 10
    var_25 = 20
    var_26 = 50
    var_27 = {var_21: var_24, var_22: var_25, var_23: var_26}
    var_28 = 'user_.*'
    var_29 = module_0.rex(var_28)
    var_30 = [var_29]
    var_31 = 11
    var_32 = 21
    var_33 = {var_21: var_31, var_22: var_32, var_23: var_26}
    var_34 = 'x'
    var_35 = 'y'
    var_36 = {var_34: var_2, var_35: var_3}
    var_37 = 3
    var_38 = {var_34: var_3, var_35: var_37}
    var_39 = 'keep'
    var_40 = 'remove_me'
    var_41 = 'also_remove'
    var_42 = {var_39: var_2, var_40: var_3, var_41: var_37}
    var_43 = [var_40]
    var_44 = [var_41]
    var_45 = {var_39: var_2}
    var_46 = 'c'
    var_47 = {var_0: var_24, var_1: var_25, var_46: var_13}
    var_48 = 15
    var_49 = lambda k, v: v > var_48
    var_50 = [var_49]
    var_51 = 31
    var_52 = {var_0: var_24, var_1: var_32, var_46: var_51}
    var_53 = {var_0: var_2}
    var_54 = 'non_existent'
    var_55 = [var_54]
    var_56 = {var_0: var_2}
    var_57 = {}
    var_58 = 'new_key'
    var_59 = [var_58]
    var_60 = 100
    var_61 = (var_59, var_60)
    var_62 = [var_61]
    var_63 = {var_58: var_60}
    var_64 = 4
    var_65 = 0
    var_66 = [var_65, var_65]
    var_67 = 'vals'
    var_68 = [var_67, var_65]
    var_69 = [var_67, var_2]
    var_70 = [var_67, var_3]
    var_71 = lambda x: x * var_24
    var_72 = (var_70, var_71)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_3, var_1: var_3}
    var_6 = 'users'
    var_7 = 'alice'
    var_8 = 'bob'
    var_9 = 'score'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_9: var_12}
    var_14 = {var_7: var_11, var_8: var_13}
    var_15 = 11
    var_16 = {var_9: var_15}
    var_17 = {var_9: var_12}
    var_18 = {var_7: var_16, var_8: var_17}
    var_19 = 'user_1'
    var_20 = 'user_2'
    var_21 = 'admin'
    var_22 = 50
    var_23 = {var_19: var_10, var_20: var_12, var_21: var_22}
    var_24 = 'user_.*'
    var_25 = module_0.rex(var_24)
    var_26 = {var_19: var_10, var_20: var_12, var_21: var_22}
    var_27 = 21
    var_28 = {var_19: var_15, var_20: var_27, var_21: var_22}
    var_29 = 30
    var_30 = {var_2: var_10, var_3: var_12, var_0: var_29}
    var_31 = {var_2: var_15, var_3: var_27, var_0: var_29}
    var_32 = 'small'
    var_33 = 'large'
    var_34 = 5
    var_35 = {var_32: var_34, var_33: var_12}
    var_36 = 15
    var_37 = lambda k, v: v > var_36
    var_38 = {var_32: var_34, var_33: var_27}
    var_39 = 'keep'
    var_40 = 'remove'
    var_41 = {var_39: var_2, var_40: var_3}
    var_42 = {var_39: var_2}
    var_43 = {var_0: var_2}
    var_44 = 'non_existent'
    var_45 = {var_0: var_2}
    var_46 = 0
    var_47 = 'c'
    var_48 = {var_1: var_2, var_47: var_3}
    var_49 = {var_47: var_3}
    var_50 = {}
    var_51 = 'new_key'
    var_52 = {var_51: var_2}
    var_53 = 'x'
    var_54 = 'y'
    var_55 = {var_53: var_2, var_54: var_3}
    var_56 = 3
    var_57 = {var_53: var_3, var_54: var_56}
    var_58 = True
    var_59 = lambda x, y, z: var_58
    var_60 = [var_2]



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 10.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.dec(var_0)
    assert var_1 == 0
    var_2 = 0
    var_3 = module_0.dec(var_2)
    assert var_3 == -1
    var_4 = 100
    var_5 = module_0.dec(var_4)
    assert var_5 == 99
    var_6 = -5
    var_7 = module_0.dec(var_6)
    assert var_7 == -6
    var_8 = 1.5
    var_9 = module_0.dec(var_8)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {var_0: var_3, var_2: var_5}
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_8, var_10]
    var_13 = 'non_existent'
    var_14 = 99
    var_15 = 'any'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 5.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 2
    var_5 = {var_0: var_4}
    var_6 = 'user'
    var_7 = 'age'
    var_8 = 'name'
    var_9 = 25
    var_10 = 'Alice'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = [var_6, var_7]
    var_13 = 26
    var_14 = {var_7: var_13, var_8: var_10}
    var_15 = 'id_1'
    var_16 = 'id_2'
    var_17 = 'other'
    var_18 = 10
    var_19 = 20
    var_20 = 30
    var_21 = {var_15: var_18, var_16: var_19, var_17: var_20}
    var_22 = 'id_.*'
    var_23 = module_0.rex(var_22)
    var_24 = 11
    var_25 = 21
    var_26 = {var_15: var_24, var_16: var_25, var_17: var_20}
    var_27 = 'b'
    var_28 = {var_0: var_1, var_27: var_4}
    var_29 = [var_0]
    var_30 = {var_0: var_1, var_27: var_4}
    var_31 = []
    var_32 = {var_27: var_4}
    var_33 = 'c'
    var_34 = 4
    var_35 = {var_0: var_1, var_27: var_4, var_33: var_34}
    var_36 = 3
    var_37 = 5
    var_38 = {var_0: var_1, var_27: var_36, var_33: var_37}
    var_39 = 'x1'
    var_40 = 'y1'
    var_41 = {var_39: var_18, var_40: var_19}
    var_42 = 'x'
    var_43 = lambda k: k.startswith(var_42)
    var_44 = {var_39: var_24, var_40: var_19}
    var_45 = {var_0: var_1}
    var_46 = {var_27: var_4}
    var_47 = 0
    var_48 = [var_47, var_0]
    var_49 = [var_47, var_18]
    var_50 = [var_49]
    var_51 = {var_0: var_1}
    var_52 = 'new_path'
    var_53 = 'sub_key'
    var_54 = [var_52, var_53]
    var_55 = 99
    var_56 = lambda x: var_55
    var_57 = [var_54, var_56]
    var_58 = {var_53: var_55}
    var_59 = {var_0: var_1, var_27: var_4}
    var_60 = {var_0: var_4, var_27: var_36}



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 10.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = 1
    var_3 = module_0.ny(var_2)
    assert var_3 is True
    var_4 = 'any string'
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    var_6 = []
    var_7 = module_0.ny(var_6)
    assert var_7 is True
    var_8 = {}
    var_9 = module_0.ny(var_8)
    assert var_9 is True
    var_10 = True
    var_11 = module_0.ny(var_10)
    assert var_11 is True
    var_12 = False
    var_13 = module_0.ny(var_12)
    assert var_13 is True



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abc '
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'apple'
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = None
    var_17 = 123
    var_18 = 'a'
    var_19 = [var_18]



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.dec(var_0)
    assert var_1 == 0
    var_2 = 0
    var_3 = module_0.dec(var_2)
    assert var_3 == -1
    var_4 = 100
    var_5 = module_0.dec(var_4)
    assert var_5 == 99
    var_6 = -5
    var_7 = module_0.dec(var_6)
    assert var_7 == -6
    var_8 = 0.5
    var_9 = module_0.dec(var_8)



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = [var_0]
    var_9 = {var_0: var_3, var_1: var_3}
    var_10 = 'users'
    var_11 = 'alice'
    var_12 = 'bob'
    var_13 = 'score'
    var_14 = 10
    var_15 = {var_13: var_14}
    var_16 = 5
    var_17 = {var_13: var_16}
    var_18 = 's'
    var_19 = [var_10, var_11, var_18]
    var_20 = {var_1: var_14}
    var_21 = [var_0, var_1]
    var_22 = 11
    var_23 = {var_1: var_22}
    var_24 = 'apple'
    var_25 = 'banana'
    var_26 = 'cherry'
    var_27 = 3
    var_28 = {var_24: var_2, var_25: var_3, var_26: var_27}
    var_29 = 'a.*'
    var_30 = module_0.rex(var_29)
    var_31 = {var_24: var_3, var_25: var_3, var_26: var_27}
    var_32 = 'c'
    var_33 = {var_0: var_2, var_1: var_3, var_32: var_27}
    var_34 = [var_0, var_1]
    var_35 = {var_32: var_27}
    var_36 = 20
    var_37 = {var_0: var_14, var_1: var_36, var_32: var_16}
    var_38 = 9
    var_39 = lambda k, v: v > var_38
    var_40 = 21
    var_41 = {var_0: var_22, var_1: var_40, var_32: var_16}
    var_42 = 30
    var_43 = 0
    var_44 = lambda i: i % var_3 == var_43
    var_45 = 31
    var_46 = 'outer'
    var_47 = 'inner'
    var_48 = 'target'
    var_49 = {var_48: var_2}
    var_50 = [var_46, var_47, var_48]
    var_51 = {var_48: var_3}
    var_52 = {var_0: var_2}
    var_53 = [var_1, var_32]
    var_54 = {var_32: var_3}
    var_55 = {var_0: var_2}
    var_56 = 'non_existent'
    var_57 = [var_56]
    var_58 = {var_0: var_2}
    var_59 = {var_0: var_2, var_1: var_3}
    var_60 = {var_0: var_3, var_1: var_27}



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'non_existent'
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = 99



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = "\n    Tests the 'ny' matcher which is designed to return True \n    regardless of the input provided.\n    "
    var_1 = None
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = 1
    var_4 = module_0.ny(var_3)
    assert var_4 is True
    var_5 = 'test'
    var_6 = module_0.ny(var_5)
    assert var_6 is True
    var_7 = []
    var_8 = module_0.ny(var_7)
    assert var_8 is True
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_0.ny(var_11)
    assert var_12 is True
    var_13 = True
    var_14 = module_0.ny(var_13)
    assert var_14 is True
    var_15 = False
    var_16 = module_0.ny(var_15)
    assert var_16 is True



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = '^test_'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_function'
    var_9 = 'testing'
    var_10 = 'my_test_function'
    var_11 = 'ABC'
    var_12 = module_0.rex(var_11)
    var_13 = 123
    var_14 = None
    var_15 = [var_2]
    var_16 = '^[a-z]+_\\d{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user_01'
    var_19 = 'admin_99'
    var_20 = 'user_1'
    var_21 = 'USER_01'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^user_'
    var_6 = module_0.rex(var_5)
    var_7 = 'user_admin'
    var_8 = 'guest_admin'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '^$'
    var_21 = module_0.rex(var_20)
    var_22 = ' '
    var_23 = '[a-z]+@[a-z]+\\.com'
    var_24 = module_0.rex(var_23)
    var_25 = 'test@example.com'
    var_26 = 'test@example.net'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\n    Test that the ny matcher returns True for any input, \n    regardless of its type or value.\n    '
    var_1 = None
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = 0
    var_4 = module_0.ny(var_3)
    assert var_4 is True
    var_5 = 1
    var_6 = module_0.ny(var_5)
    assert var_6 is True
    var_7 = ''
    var_8 = module_0.ny(var_7)
    assert var_8 is True
    var_9 = 'anything'
    var_10 = module_0.ny(var_9)
    assert var_10 is True
    var_11 = []
    var_12 = module_0.ny(var_11)
    assert var_12 is True
    var_13 = {}
    var_14 = module_0.ny(var_13)
    assert var_14 is True
    var_15 = False
    var_16 = module_0.ny(var_15)
    assert var_16 is True
    var_17 = 2
    var_18 = 3
    var_19 = [var_5, var_17, var_18]
    var_20 = module_0.ny(var_19)
    assert var_20 is True



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.dec(var_0)
    assert var_1 == 0
    var_2 = 0
    var_3 = module_0.dec(var_2)
    assert var_3 == -1
    var_4 = 100
    var_5 = module_0.dec(var_4)
    assert var_5 == 99
    var_6 = -5
    var_7 = module_0.dec(var_6)
    assert var_7 == -6
    var_8 = 'string'
    var_9 = module_0.dec(var_8)



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'non_existent'
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_8, var_10]
    var_13 = 99
    var_14 = 'x'
    var_15 = 100
    var_16 = {var_14: var_15}
    var_17 = module_0.discard(var_16, var_14)



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 99
    var_7 = module_0.inc(var_6)
    assert var_7 == 100
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'non_existent'
    var_8 = 10
    var_9 = 20
    var_10 = 30
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_8, var_10]
    var_13 = 5
    var_14 = {var_0: var_3, var_1: var_4}
    var_15 = [var_0]
    var_16 = 'outer'
    var_17 = 'inner'
    var_18 = 'stay'
    var_19 = {var_17: var_3, var_18: var_4}
    var_20 = [var_16, var_17]



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.dec(var_0)
    assert var_1 == 0
    var_2 = 0
    var_3 = module_0.dec(var_2)
    assert var_3 == -1
    var_4 = 100
    var_5 = module_0.dec(var_4)
    assert var_5 == 99
    var_6 = -5
    var_7 = module_0.dec(var_6)
    assert var_7 == -6
    var_8 = 'string'
    var_9 = module_0.dec(var_8)



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = module_0.rex(var_0)
    var_2 = 'apples'
    var_3 = 123
    var_4 = None
    var_5 = '^a.*e$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ace'
    var_8 = 'abcde'
    var_9 = 'applepie'
    var_10 = 'banana'
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = module_0.rex(var_16)
    var_19 = []
    var_20 = module_0.rex(var_16)
    var_21 = {}



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '[a-z]+@[a-z]+\\.com'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@domain.com'
    var_23 = 'user@domain.net'



# Parsed testcases at query #43
#--------------------------


import pyrsistent._transformations as module_0
import builtins as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = 1
    var_3 = module_0.ny(var_2)
    assert var_3 is True
    var_4 = 'string'
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    var_6 = []
    var_7 = module_0.ny(var_6)
    assert var_7 is True
    var_8 = {}
    var_9 = module_0.ny(var_8)
    assert var_9 is True
    var_10 = False
    var_11 = module_0.ny(var_10)
    assert var_11 is True
    var_12 = module_1.object()
    var_13 = module_0.ny(var_12)
    assert var_13 is True



# Parsed testcases at query #44
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 10.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0
import builtins as module_1

def test_case_0():
    var_0 = '\n    Test the ny matcher, which should return True for any input value.\n    '
    var_1 = None
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = True
    var_4 = module_0.ny(var_3)
    assert var_4 is True
    var_5 = False
    var_6 = module_0.ny(var_5)
    assert var_6 is False
    var_7 = 123
    var_8 = module_0.ny(var_7)
    assert var_8 is True
    var_9 = 'any string'
    var_10 = module_0.ny(var_9)
    assert var_10 is True
    var_11 = []
    var_12 = module_0.ny(var_11)
    assert var_12 is True
    var_13 = {}
    var_14 = module_0.ny(var_13)
    assert var_14 is True
    var_15 = module_1.object()
    var_16 = module_0.ny(var_15)
    assert var_16 is True



# Parsed testcases at query #46
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = "\n    Tests that the 'ny' matcher returns True regardless of the input provided.\n    "
    var_1 = None
    var_2 = module_0.ny(var_1)
    assert var_2 is True
    var_3 = 0
    var_4 = module_0.ny(var_3)
    assert var_4 is True
    var_5 = ''
    var_6 = module_0.ny(var_5)
    assert var_6 is True
    var_7 = []
    var_8 = module_0.ny(var_7)
    assert var_8 is True
    var_9 = {}
    var_10 = module_0.ny(var_9)
    assert var_10 is True
    var_11 = False
    var_12 = module_0.ny(var_11)
    assert var_12 is True
    var_13 = 'any string'
    var_14 = module_0.ny(var_13)
    assert var_14 is True
    var_15 = 123
    var_16 = module_0.ny(var_15)
    assert var_16 is True
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.ny(var_20)
    assert var_21 is True



# Parsed testcases at query #47
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'e'
    var_3 = 1
    var_4 = 'c'
    var_5 = 'd'
    var_6 = 2
    var_7 = 3
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 10
    var_10 = 20
    var_11 = [var_0]
    var_12 = 'name_of_key_not_needed_but_path_is'
    var_13 = [var_1, var_12, var_4]
    var_14 = [var_1, var_4]
    var_15 = '^[ab]'
    var_16 = module_0.rex(var_15)
    var_17 = module_0.rex(var_15)
    var_18 = 4
    var_19 = {var_4: var_7, var_5: var_18}
    var_20 = [var_0]
    var_21 = 9
    var_22 = 19
    var_23 = [var_1, var_4]
    var_24 = 0
    var_25 = [var_2, var_24]
    var_26 = [var_0]
    var_27 = 'non_existent'
    var_28 = [var_27]
    var_29 = 'new'
    var_30 = True
    var_31 = {var_29: var_30}
    var_32 = True
    var_33 = {var_29: var_32}
    var_34 = 'ghost'
    var_35 = 'path'
    var_36 = [var_34, var_35]



# Parsed testcases at query #48
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 123
    var_5 = None
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = ''
    var_10 = '^pre'
    var_11 = module_0.rex(var_10)
    var_12 = 'prefix'
    var_13 = 'pre'
    var_14 = 'aprefix'
    var_15 = '^[a-z]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'hello'
    var_18 = 'Hello'
    var_19 = 'h1'
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = 'a'
    var_23 = [var_22]
    var_24 = 'key'
    var_25 = 'val'
    var_26 = {var_24: var_25}



# Parsed testcases at query #49
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.inc(var_0)
    assert var_1 == 2
    var_2 = 0
    var_3 = module_0.inc(var_2)
    assert var_3 == 1
    var_4 = -1
    var_5 = module_0.inc(var_4)
    assert var_5 == 0
    var_6 = 10.5
    var_7 = module_0.inc(var_6)
    var_8 = 'string'
    var_9 = module_0.inc(var_8)



# Parsed testcases at query #50
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre_'
    var_6 = module_0.rex(var_5)
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = 'pre'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = 123
    var_13 = None
    var_14 = 'a'
    var_15 = [var_14]
    var_16 = '^exact$'
    var_17 = module_0.rex(var_16)
    var_18 = 'exact'
    var_19 = 'exact_extra'
    var_20 = '^[A-Z]+$'
    var_21 = module_0.rex(var_20)
    var_22 = 'HELLO'
    var_23 = 'hello'



# Parsed testcases at query #51
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '12a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-z]+@example\\.com$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'test@gmail.com'
    var_13 = '123@example.com'
    var_14 = '^hello$'
    var_15 = module_0.rex(var_14)
    var_16 = 'hello'
    var_17 = 'hello world'



