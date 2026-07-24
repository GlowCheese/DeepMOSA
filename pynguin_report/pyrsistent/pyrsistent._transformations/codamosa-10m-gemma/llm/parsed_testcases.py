####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'non_existent'
    var_6 = 10
    var_7 = 20
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = [var_6, var_8]
    var_11 = 5
    var_12 = [var_6, var_7, var_8]



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^apple'
    var_6 = module_0.rex(var_5)
    var_7 = 'applepie'
    var_8 = 'pineapple'
    var_9 = '[a-z]+_[0-9]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_123'
    var_12 = 'TEST_123'
    var_13 = 'test_abc'
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'anything'
    var_18 = module_0.rex(var_14)
    var_19 = None
    var_20 = []



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'non_existent'
    var_6 = 10
    var_7 = 20
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = 1
    var_11 = 99



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = [var_0]
    var_7 = {var_0: var_3, var_1: var_3}
    var_8 = 'users'
    var_9 = 'alice'
    var_10 = 'bob'
    var_11 = 'age'
    var_12 = 25
    var_13 = {var_11: var_12}
    var_14 = 30
    var_15 = {var_11: var_14}
    var_16 = {var_9: var_13, var_10: var_15}
    var_17 = [var_8, var_9, var_11]



# Parsed testcases at query #5
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
    var_17 = {var_14: var_15}
    var_18 = module_0.discard(var_17, var_14)
    var_19 = 'y'
    var_20 = module_0.discard(var_17, var_19)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^[a-z]+@domain\\.com$'
    var_6 = module_0.rex(var_5)
    var_7 = 'user@domain.com'
    var_8 = 'user123@domain.com'
    var_9 = 'USER@domain.com'
    var_10 = 'start'
    var_11 = module_0.rex(var_10)
    var_12 = 'end'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = None
    var_20 = module_0.rex(var_17)
    var_21 = []
    var_22 = module_0.rex(var_17)
    var_23 = {}



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
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = 123
    var_13 = None
    var_14 = 'list'
    var_15 = [var_14]
    var_16 = '^[a-z]+_[0-9]{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'hello_99'
    var_19 = 'hello_9'
    var_20 = 'Hello_99'
    var_21 = 'abc_def'



# Parsed testcases at query #8
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
    var_7 = 'pre_test'
    var_8 = 'test_pre'
    var_9 = None
    var_10 = '^[a-z]+_\\d{2}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello_99'
    var_13 = 'hello_9'
    var_14 = 'HELLO_99'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '
    var_19 = '.*'
    var_20 = module_0.rex(var_19)
    var_21 = module_0.rex(var_19)
    var_22 = []
    var_23 = module_0.rex(var_19)
    var_24 = {}



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'test_pre'
    var_10 = 'pre_'
    var_11 = [var_10]
    var_12 = '^[a-z]+_\\d{2}$'
    var_13 = module_0.rex(var_12)
    var_14 = 'hello_99'
    var_15 = 'hello_9'
    var_16 = 'HELLO_99'
    var_17 = 'item_abc'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = ' '
    var_22 = '.*'
    var_23 = module_0.rex(var_22)
    var_24 = 'anything'
    var_25 = True



# Parsed testcases at query #10
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
    var_12 = 'user123@domain.com'
    var_13 = 'USER@domain.com'
    var_14 = '^exact_string$'
    var_15 = module_0.rex(var_14)
    var_16 = 'exact_string'
    var_17 = 'exact_string_extra'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^test_'
    var_6 = module_0.rex(var_5)
    var_7 = 'test_case'
    var_8 = 'test'
    var_9 = 'production_case'
    var_10 = '^[a-z]+_\\d{2}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc_12'
    var_13 = 'abc_1'
    var_14 = 'ABC_12'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = None
    var_18 = True



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^pre'
    var_7 = module_0.rex(var_6)
    var_8 = 'prefix'
    var_9 = 'pre'
    var_10 = 'suffix'
    var_11 = '^[a-z]+_\\d{2}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test_01'
    var_14 = 'test_1'
    var_15 = 'TEST_01'
    var_16 = 'abc_def'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = True
    var_20 = module_0.rex(var_17)
    var_21 = []
    var_22 = module_0.rex(var_17)
    var_23 = {}



# Parsed testcases at query #13
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
    var_8 = '^[a-z]+@example\\.com$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@gmail.com'
    var_12 = 'USER@example.com'
    var_13 = '^Start'
    var_14 = module_0.rex(var_13)
    var_15 = 'Start here'
    var_16 = 'The Start'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ' '



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple_pie'
    var_3 = 'banana'
    var_4 = '^[0-9]+$'
    var_5 = module_0.rex(var_4)
    var_6 = '123'
    var_7 = '123a'
    var_8 = ''
    var_9 = 'pre'
    var_10 = module_0.rex(var_9)
    var_11 = 'prefix'
    var_12 = 'unprefix'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = 123
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = '^[a-z]+_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'test_01'
    var_23 = 'test_1'
    var_24 = 'TEST_01'
    var_25 = 'abc_def'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'suffix'
    var_9 = '^[A-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = 'hello'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = 'a'
    var_18 = [var_17]
    var_19 = True
    var_20 = '^[a-z]+_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'test_01'
    var_23 = 'test_1'
    var_24 = 'TEST_01'
    var_25 = 'abc_def'



# Parsed testcases at query #16
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
    var_9 = '^[a-z]+_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_01'
    var_12 = 'TEST_01'
    var_13 = 'test_1'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = 'hello'
    var_20 = None
    var_21 = 'a'
    var_22 = [var_21]



# Parsed testcases at query #17
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
    var_19 = '^[a-z]+_[0-9]{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'abc_12'
    var_22 = 'abc_1'
    var_23 = 'ABC_12'
    var_24 = 'abc_123'



# Parsed testcases at query #18
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
    var_19 = 'exact_extra'
    var_20 = '^[A-Z]+$'
    var_21 = module_0.rex(var_20)
    var_22 = 'HELLO'
    var_23 = 'hello'



# Parsed testcases at query #19
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
    var_9 = 'aprefix'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = 'a'
    var_20 = [var_19]
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ' '



# Parsed testcases at query #20
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
    var_15 = '^[a-z]+@[a-z]+\\.com$'
    var_16 = module_0.rex(var_15)
    var_17 = 'user@example.com'
    var_18 = 'user@example.net'
    var_19 = 'User@example.com'



# Parsed testcases at query #21
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
    var_9 = 'aprefix'
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '[a-z]'
    var_14 = module_0.rex(var_13)
    var_15 = 'a'
    var_16 = 'A'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = 123
    var_20 = None
    var_21 = [var_15]
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ''
    var_25 = ' '



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abc '
    var_5 = '^abc'
    var_6 = module_0.rex(var_5)
    var_7 = 'abcde'
    var_8 = 'zabc'
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = '12a'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'any_string'
    var_17 = 123
    var_18 = None
    var_19 = 'a'
    var_20 = [var_19]
    var_21 = '^[a-z]+_\\d{2}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'test_01'
    var_24 = 'TEST_01'
    var_25 = 'test_1'
    var_26 = 'test_abc'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abc '
    var_5 = '^user_'
    var_6 = module_0.rex(var_5)
    var_7 = 'user_123'
    var_8 = 'admin_123'
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '12345'
    var_12 = '123a5'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = [var_2]
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = ' '
    var_22 = '^[A-Z]{2}-\\d{3}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'AB-123'
    var_25 = 'abc-123'
    var_26 = 'A-123'
    var_27 = 'AB-12'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = 'test_\\w+'
    var_7 = module_0.rex(var_6)
    var_8 = 'test_abc'
    var_9 = 'test_123'
    var_10 = 'other_abc'
    var_11 = ''
    var_12 = '^exact$'
    var_13 = module_0.rex(var_12)
    var_14 = 'exact'
    var_15 = 'exact_extra'
    var_16 = '^[a-z]{3}-[0-9]$'
    var_17 = module_0.rex(var_16)
    var_18 = 'abc-1'
    var_19 = 'abcd-1'
    var_20 = 'abc-a'
    var_21 = 'ABC-1'
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ' '



# Parsed testcases at query #25
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
    var_18 = '^ABC'
    var_19 = module_0.rex(var_18)
    var_20 = 'ABC'
    var_21 = 'abc'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple_pie'
    var_3 = 'banana'
    var_4 = 'pre'
    var_5 = module_0.rex(var_4)
    var_6 = 'prefix'
    var_7 = 'superprefix'
    var_8 = 'a.*z'
    var_9 = module_0.rex(var_8)
    var_10 = 'abcz'
    var_11 = 'az'
    var_12 = 'abc'
    var_13 = '[0-9]+'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 123
    var_19 = None
    var_20 = 'test'
    var_21 = [var_20]
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ''
    var_25 = ' '



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc123'
    var_3 = 'abc'
    var_4 = 'abc123def'
    var_5 = 'xyz123'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^fixed$'
    var_10 = module_0.rex(var_9)
    var_11 = 'fixed'
    var_12 = 'fixed_extra'
    var_13 = '^abc$'
    var_14 = module_0.rex(var_13)
    var_15 = 'ABC'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



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
    var_9 = 'aprefix'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = [var_2]
    var_20 = '^$'
    var_21 = module_0.rex(var_20)
    var_22 = ' '



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = ''
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'append'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello'
    var_13 = 123
    var_14 = None
    var_15 = 'list'
    var_16 = [var_15]
    var_17 = 'ABC'
    var_18 = module_0.rex(var_17)
    var_19 = '^[a-z]+_[0-9]{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'test_01'
    var_22 = 'test_1'
    var_23 = 'TEST_01'
    var_24 = 'abc_def'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^[a-z]+@domain\\.com$'
    var_6 = module_0.rex(var_5)
    var_7 = 'user@domain.com'
    var_8 = 'user@other.com'
    var_9 = 'USER@domain.com'
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = ' '
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = None
    var_18 = 'list'
    var_19 = [var_18]
    var_20 = 1
    var_21 = '[a-z]'
    var_22 = module_0.rex(var_21)
    var_23 = 'a'
    var_24 = 'A'
    var_25 = '1'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'non_existent'
    var_6 = 10
    var_7 = 20
    var_8 = 30
    var_9 = 'any'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {var_2: var_5}
    var_8 = 'non_existent'
    var_9 = 10
    var_10 = 20
    var_11 = 30
    var_12 = [var_9, var_10, var_11]
    var_13 = [var_9, var_11]
    var_14 = 5
    var_15 = 'anything'



# Parsed testcases at query #3
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
    var_7 = 'pre_test'
    var_8 = 'post_test'
    var_9 = None
    var_10 = '[a-z]+@[a-z]+\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@test.com'
    var_13 = 'user@test.net'
    var_14 = '123@test.com'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'anything'
    var_19 = True
    var_20 = []



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'suffix'
    var_10 = '^[a-z]+@[a-z]+\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test@example.com'
    var_13 = 'TEST@example.com'
    var_14 = 'test@example.org'
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = None
    var_18 = module_0.rex(var_15)
    var_19 = []
    var_20 = module_0.rex(var_15)
    var_21 = {}
    var_22 = '^exact$'
    var_23 = module_0.rex(var_22)
    var_24 = 'exact'
    var_25 = 'not_exact'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {var_2: var_5}
    var_8 = {var_0: var_3}
    var_9 = 'non_existent'
    var_10 = {var_0: var_3}
    var_11 = 10
    var_12 = 20
    var_13 = 30
    var_14 = [var_11, var_12, var_13]
    var_15 = [var_11, var_13]
    var_16 = 'anything'



# Parsed testcases at query #6
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
    var_16 = '^[a-z]+@[a-z]+\\.com$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example.net'
    var_20 = 'User@example.com'



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
    var_8 = 'test_pre'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = 'anything'
    var_16 = module_0.ny(var_15)
    assert var_16 is True
    var_17 = module_0.ny(var_11)
    assert var_17 is True
    var_18 = module_0.ny(var_12)
    assert var_18 is True
    var_19 = '^[a-z]+_\\d{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'hello_99'
    var_22 = 'hello_9'
    var_23 = 'HELLO_99'
    var_24 = 'abc_def'



# Parsed testcases at query #8
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
    var_7 = {var_2: var_5}
    var_8 = 'non_existent'
    var_9 = 10
    var_10 = 20
    var_11 = 30
    var_12 = [var_9, var_10, var_11]
    var_13 = [var_9, var_11]
    var_14 = 5
    var_15 = 'x'
    var_16 = 'y'
    var_17 = 100
    var_18 = 200
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = module_0.discard(var_19, var_15)
    var_21 = {var_15: var_17}
    var_22 = 'z'
    var_23 = module_0.discard(var_21, var_22)



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
    var_19 = '^[a-z]+_\\d{2}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'hello_99'
    var_22 = 'hello_9'
    var_23 = 'HELLO_99'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = e.persistent()[var_1]
    assert var_5 == 2
    var_6 = 'non_existent'
    var_7 = 10
    var_8 = 20
    var_9 = 30
    var_10 = [var_7, var_8, var_9]
    var_11 = [var_7, var_9]
    var_12 = 5



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^pre'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix'
    var_8 = 'pre'
    var_9 = 'post'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = None
    var_13 = module_0.rex(var_10)
    var_14 = []
    var_15 = module_0.rex(var_10)
    var_16 = {}
    var_17 = '^[a-z]+_\\d{2}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test_01'
    var_20 = 'test_1'
    var_21 = 'TEST_01'
    var_22 = 'abc_def'
    var_23 = '^$'
    var_24 = module_0.rex(var_23)
    var_25 = ''
    var_26 = ' '



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
    var_20 = '^[a-z]{3}_\\d{2}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc_12'
    var_23 = 'abcd_12'
    var_24 = 'abc_1'
    var_25 = 'ABC_12'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abcde'
    var_5 = 'def'
    var_6 = '123abc'
    var_7 = '^\\d+$'
    var_8 = module_0.rex(var_7)
    var_9 = '123'
    var_10 = '12a'
    var_11 = ''
    var_12 = '^[A-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'HELLO'
    var_15 = 'hello'
    var_16 = 123
    var_17 = None
    var_18 = [var_2]
    var_19 = '\\buser_\\d+\\b'
    var_20 = module_0.rex(var_19)
    var_21 = 'user_1'
    var_22 = 'user_999'
    var_23 = 'myuser_1'
    var_24 = 'user_abc'
    var_25 = '^$'
    var_26 = module_0.rex(var_25)
    var_27 = ' '



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^[a-z]+@[a-z]+\\.com$'
    var_7 = module_0.rex(var_6)
    var_8 = 'test@example.com'
    var_9 = 'test@example.net'
    var_10 = 'TEST@example.com'
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^prefix'
    var_16 = module_0.rex(var_15)
    var_17 = 'prefix_suffix'
    var_18 = 'not_prefix'
    var_19 = '[a-v]'
    var_20 = module_0.rex(var_19)
    var_21 = 'a'
    var_22 = 'z'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '12a'
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
    var_16 = '[a-z]+@[a-z]+\\.com'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@domain.com'
    var_19 = 'user@domain.org'
    var_20 = 'User@domain.com'



# Parsed testcases at query #16
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
    var_16 = '^[a-z]+_[0-9]{2}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'abc_12'
    var_19 = 'abc_1'
    var_20 = 'ABC_12'
    var_21 = 'abc_123'
    var_22 = '^$'
    var_23 = module_0.rex(var_22)
    var_24 = ' '



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'test_pre'
    var_10 = '^[a-z]+_\\d{2}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello_99'
    var_13 = 'hello_9'
    var_14 = 'HELLO_99'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = True
    var_21 = module_0.rex(var_18)
    var_22 = []
    var_23 = module_0.rex(var_18)
    var_24 = {}



# Parsed testcases at query #18
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
    var_17 = '^[A-Z]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'HELLO'
    var_20 = 'hello'
    var_21 = '^[a-z]+_\\d{2}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user_01'
    var_24 = 'user_1'
    var_25 = 'USER_01'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = 'ab'
    var_4 = 123
    var_5 = None
    var_6 = '\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = ''
    var_10 = '^start_.*_end$'
    var_11 = module_0.rex(var_10)
    var_12 = 'start_anything_end'
    var_13 = 'start_end'
    var_14 = 'start_middle'
    var_15 = 'middle_end'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ' '
    var_19 = '[a-z]+'
    var_20 = module_0.rex(var_19)
    var_21 = 'hello'
    var_22 = 'HELLO'



# Parsed testcases at query #20
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
    var_9 = 'aprefix'
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 123
    var_16 = None
    var_17 = [var_2]
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = ' '
    var_22 = '^[a-z]+_\\d{2}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test_01'
    var_25 = 'TEST_01'
    var_26 = 'test_1'
    var_27 = 'test_abc'



# Parsed testcases at query #21
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
    var_11 = '^hello$'
    var_12 = module_0.rex(var_11)
    var_13 = 'hello'
    var_14 = 'hello world'
    var_15 = '^[aeiou]'
    var_16 = module_0.rex(var_15)
    var_17 = 'orange'
    var_18 = 'pear'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ' '



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^apple$'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'apples'
    var_4 = 'pineapple'
    var_5 = '^a.*e$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ace'
    var_8 = 'abc'
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = None
    var_13 = []
    var_14 = module_0.rex(var_9)
    var_15 = '123'
    var_16 = '123a'
    var_17 = '^ABC$'
    var_18 = module_0.rex(var_17)
    var_19 = 'ABC'



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
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^test_'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_function'
    var_12 = 'testing'
    var_13 = 'my_test_function'
    var_14 = '^[a-z]+_[0-9]{2}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'data_99'
    var_17 = 'data_9'
    var_18 = 'DATA_99'



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
    var_8 = [var_2]
    var_9 = '^pre'
    var_10 = module_0.rex(var_9)
    var_11 = 'prefix_value'
    var_12 = 'pre'
    var_13 = 'post'
    var_14 = '^[a-z]+_\\d{2}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test_01'
    var_17 = 'test_1'
    var_18 = 'TEST_01'
    var_19 = 'abc_123'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'test_pre'
    var_10 = ''
    var_11 = '^exact$'
    var_12 = module_0.rex(var_11)
    var_13 = 'exact'
    var_14 = 'exact_suffix'
    var_15 = '^[a-z]+_[0-9]{2}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'abc_12'
    var_18 = 'abc_1'
    var_19 = 'ABC_12'
    var_20 = 'abc_ab'
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ' '



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^prefix'
    var_6 = module_0.rex(var_5)
    var_7 = 'prefix_suffix'
    var_8 = 'not_prefix'
    var_9 = '^[a-z]+_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello_99'
    var_12 = 'hello_9'
    var_13 = 'HELLO_99'
    var_14 = 'test_abc'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '
    var_19 = '.*'
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = None
    var_23 = []



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'abc'
    var_1 = module_0.rex(var_0)
    var_2 = 'abcd'
    var_3 = 'ab'
    var_4 = 123
    var_5 = None
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = '12a'
    var_10 = ''
    var_11 = 'Hello'
    var_12 = module_0.rex(var_11)
    var_13 = 'hello'
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = []
    var_18 = True



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
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 123
    var_18 = None
    var_19 = 'a'
    var_20 = [var_19]
    var_21 = '^[a-z]+_[0-9]{2}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'test_01'
    var_24 = 'test_1'
    var_25 = 'TEST_01'
    var_26 = 'abc_99'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = None
    var_6 = '^pre_'
    var_7 = module_0.rex(var_6)
    var_8 = 'pre_test'
    var_9 = 'test_pre'
    var_10 = ''
    var_11 = '^[a-z]+_[0-9]{2}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'item_01'
    var_14 = 'item_1'
    var_15 = 'ITEM_01'
    var_16 = 'abc_def'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = 'xyz'
    var_21 = module_0.rex(var_20)



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
    var_20 = '^[A-Z]+$'
    var_21 = module_0.rex(var_20)
    var_22 = 'HELLO'
    var_23 = 'hello'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = 123
    var_5 = '^[a-z]+@domain\\.com$'
    var_6 = module_0.rex(var_5)
    var_7 = 'user@domain.com'
    var_8 = 'USER@domain.com'
    var_9 = 'user@other.com'
    var_10 = '.+'
    var_11 = module_0.rex(var_10)
    var_12 = 'a'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = None
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = True



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = '0'
    var_4 = 'abc'
    var_5 = '123a'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '[^@]+@[^@]+\\.[^@]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'invalid-email'
    var_13 = '^hello$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello'
    var_16 = 'hello world'

import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'anything'
    var_3 = ''
    var_4 = 123



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
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-z]+_\\d{2}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test_01'
    var_12 = 'test_1'
    var_13 = 'TEST_01'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '



