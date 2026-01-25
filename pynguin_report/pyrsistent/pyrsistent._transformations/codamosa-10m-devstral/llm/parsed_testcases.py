####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_7 = module_0.discard(var_6, var_1)
    var_8 = {var_0: var_3, var_1: var_4}
    var_9 = module_0.discard(var_8, var_2)
    var_10 = [var_3, var_4, var_5]
    var_11 = 1
    var_12 = module_0.discard(var_10, var_11)
    var_13 = 'x'
    var_14 = module_0.discard(var_10, var_13)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'user@example.com'
    var_8 = 'user@example'
    var_9 = 'user@.com'
    var_10 = '^test$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test'
    var_13 = 'Test'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '



# Parsed testcases at query #3
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
    var_7 = module_0.discard(var_6, var_1)
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_9 = 'd'
    var_10 = module_0.discard(var_8, var_9)
    var_11 = {}
    var_12 = module_0.discard(var_11, var_0)
    var_13 = [var_3, var_4, var_5]
    var_14 = 1
    var_15 = module_0.discard(var_13, var_14)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_1: var_2}
    var_6 = {var_0: var_5}
    var_7 = 'a1'
    var_8 = 'a2'
    var_9 = 3
    var_10 = {var_7: var_2, var_8: var_3, var_1: var_9}
    var_11 = 'a\\d'
    var_12 = module_0.rex(var_11)
    var_13 = {var_0: var_2, var_1: var_3}
    var_14 = {var_0: var_2}
    var_15 = {var_0: var_2}
    var_16 = [var_2, var_3, var_9]
    var_17 = {var_0: var_2, var_1: var_3}
    var_18 = {var_0: var_2, var_1: var_3}
    var_19 = lambda k: k == var_0
    var_20 = {var_0: var_2, var_1: var_3}
    var_21 = lambda k, v: v == var_3



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
    var_7 = module_0.discard(var_6, var_1)
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_9 = 'd'
    var_10 = module_0.discard(var_8, var_9)
    var_11 = {}
    var_12 = module_0.discard(var_11, var_0)
    var_13 = [var_3, var_4, var_5]
    var_14 = 1
    var_15 = module_0.discard(var_13, var_14)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'abc_test'
    var_4 = 123
    var_5 = '^[a-z]+_[0-9]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc_123'
    var_8 = 'ABC_123'
    var_9 = 'abc_123_'
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'abc'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc123def'
    var_11 = 'abcdef'
    var_12 = '^[a-zA-Z_][a-zA-Z0-9_]*$'
    var_13 = module_0.rex(var_12)
    var_14 = 'valid_var'
    var_15 = '1invalid_var'
    var_16 = 'invalid-var'
    var_17 = '(?i)^hello$'
    var_18 = module_0.rex(var_17)
    var_19 = 'HELLO'
    var_20 = 'hello'
    var_21 = 'Hello'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[A-Z][a-z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Hello'
    var_8 = 'hello'
    var_9 = 'HELLO'
    var_10 = '.*'
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = ''



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = 'test'
    var_9 = module_0.rex(var_8)
    var_10 = 'prefix_test'
    var_11 = '^[a-zA-Z0-9_]+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'valid_name'
    var_14 = 'invalid@name'
    var_15 = '^TEST$'
    var_16 = module_0.rex(var_15)
    var_17 = 'TEST'
    var_18 = '^(\\w+)_(\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'name_123'
    var_21 = 'name_abc'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '^([a-zA-Z]+)_(\\d{4})$'
    var_9 = module_0.rex(var_8)
    var_10 = 'prefix_2023'
    var_11 = 'prefix_23'
    var_12 = 'prefix_2023_extra'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid-email'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = module_0.rex(var_7)
    var_9 = 'anything'
    var_10 = '^\\w+@\\w+\\.\\w+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid@email'
    var_14 = '^Test$'
    var_15 = module_0.rex(var_14)
    var_16 = 'Test'
    var_17 = 'test'
    var_18 = '(?i)^test$'
    var_19 = module_0.rex(var_18)
    var_20 = 'TEST'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_value'
    var_3 = 'other_value'
    var_4 = '^TEST'
    var_5 = module_0.rex(var_4)
    var_6 = 'TEST'
    var_7 = 'test'
    var_8 = '^\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = '123'
    var_11 = 'abc'
    var_12 = 123
    var_13 = None
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 'TEST'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-z]+@[a-z]+\\.[a-z]{2,3}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'user@sub.example.com'
    var_13 = 'USER@example.com'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '123456789'
    var_21 = '123-45-678'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'Hello123'
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc123def'
    var_14 = 'abcdef'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ' '



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Case$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Case'
    var_9 = 'case'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email@'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = 'not empty'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'file_123'
    var_22 = 'file_abc'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'helloworld'
    var_4 = '^hello.*'
    var_5 = module_0.rex(var_4)
    var_6 = 'goodbye'
    var_7 = module_0.rex(var_0)
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = ' '
    var_14 = '^\\d+$'
    var_15 = module_0.rex(var_14)
    var_16 = '123'
    var_17 = 'abc'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = 123
    var_15 = None
    var_16 = [var_12]
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z]+)@([a-zA-Z]+\\.[a-zA-Z]+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'user@example'
    var_24 = 'user@.com'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = '^TEST'
    var_6 = module_0.rex(var_5)
    var_7 = 'TEST'
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = '^\\d+$'
    var_15 = module_0.rex(var_14)
    var_16 = '123'
    var_17 = 'abc'
    var_18 = '12a3'



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = '^Test'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 123
    var_9 = None
    var_10 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_11 = module_0.rex(var_10)
    var_12 = '123-45-6789'
    var_13 = '12-34-5678'
    var_14 = '1234-56-7890'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^a\\.b$'
    var_19 = module_0.rex(var_18)
    var_20 = 'a.b'
    var_21 = 'ab'
    var_22 = 'aXb'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = 123
    var_11 = None
    var_12 = [var_9]
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^test\\.$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.'
    var_20 = 'test..'
    var_21 = '^([a-zA-Z]+)(\\d+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'abc123'
    var_24 = '123abc'
    var_25 = 'abc'
    var_26 = '123'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '^\\d{3}$'
    var_9 = module_0.rex(var_8)
    var_10 = '1234'
    var_11 = '12'
    var_12 = '^[A-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = '^test\\.$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.'
    var_19 = 'test'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abcabc'
    var_5 = module_0.rex(var_2)
    var_6 = 'xabc'
    var_7 = 'xabcd'
    var_8 = 123
    var_9 = None
    var_10 = [var_2]
    var_11 = '^a.c$'
    var_12 = module_0.rex(var_11)
    var_13 = 'axc'
    var_14 = 'a1c'
    var_15 = 'ac'
    var_16 = '^ABC$'
    var_17 = module_0.rex(var_16)
    var_18 = 'ABC'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^[A-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = '^user@\\w+\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@.com'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^([a-zA-Z0-9._-]+)@([a-zA-Z0-9._-]+)\\.([a-zA-Z]{2,})$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.user@example.com'
    var_20 = 'invalid@.com'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^user@\\w+\\.com$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example.org'
    var_15 = 'user@example'
    var_16 = '123'
    var_17 = None
    var_18 = 123
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^(\\d{3})-(\\d{3})-(\\d{4})$'
    var_23 = module_0.rex(var_22)
    var_24 = '123-456-7890'
    var_25 = '1234567890'
    var_26 = '123-456-789'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^hello\\.world$'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello.world'
    var_13 = 'helloworld'
    var_14 = '123'
    var_15 = None
    var_16 = 123
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z0-9_]+)@([a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'invalid.email'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = '^.*\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.txt.bak'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = module_0.rex(var_15)
    var_18 = 'anything'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'a.c'
    var_5 = module_0.rex(var_4)
    var_6 = 'axc'
    var_7 = 'ac'
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = 'a\\.b'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = 'hello'
    var_9 = module_0.rex(var_8)
    var_10 = 'hello world'
    var_11 = 'world hello'
    var_12 = 'hello\\.world'
    var_13 = module_0.rex(var_12)
    var_14 = 'hello.world'
    var_15 = 'helloworld'
    var_16 = '[A-Z]+'
    var_17 = module_0.rex(var_16)
    var_18 = 'ABC'
    var_19 = 'abc'
    var_20 = 'cat|dog'
    var_21 = module_0.rex(var_20)
    var_22 = 'cat'
    var_23 = 'dog'
    var_24 = 'bird'



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 'test_123_extra'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'ab'
    var_14 = 'aXb'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'a'
    var_19 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'user@example.com'
    var_22 = 'invalid-email'
    var_23 = 'user@.com'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = 'hello'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello world'
    var_10 = 'world hello'
    var_11 = '[A-Z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'ABC'
    var_14 = 'abc'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = module_0.rex(var_7)
    var_9 = 'anything'
    var_10 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid-email'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = 123
    var_7 = None
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc123def'
    var_11 = 'abcdef'
    var_12 = '[A-Z]+'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = '\\w+@\\w+\\.\\w+'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 10



# Parsed testcases at query #2
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
    var_7 = module_0.discard(var_6, var_1)
    var_8 = {var_0: var_3, var_1: var_4}
    var_9 = module_0.discard(var_8, var_2)
    var_10 = [var_3, var_4, var_5]
    var_11 = 1
    var_12 = module_0.discard(var_10, var_11)
    var_13 = 'x'
    var_14 = module_0.discard(var_10, var_13)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abcabc'
    var_5 = '^a.*c$'
    var_6 = module_0.rex(var_5)
    var_7 = 'a123c'
    var_8 = 'a123'
    var_9 = '123c'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'a'
    var_17 = '^a\\.b$'
    var_18 = module_0.rex(var_17)
    var_19 = 'a.b'
    var_20 = 'ab'
    var_21 = 'aXb'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 123
    var_6 = None
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]{2,3}$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example.co.uk'
    var_11 = 'USER@EXAMPLE.COM'
    var_12 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_13 = module_0.rex(var_12)
    var_14 = '123-45-6789'
    var_15 = '12-345-6789'
    var_16 = '123-456-789'



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
    var_7 = module_0.discard(var_6, var_1)
    var_8 = {var_0: var_3, var_1: var_4}
    var_9 = module_0.discard(var_8, var_2)
    var_10 = [var_3, var_4, var_5]
    var_11 = 1
    var_12 = module_0.discard(var_10, var_11)
    var_13 = 'x'
    var_14 = module_0.discard(var_10, var_13)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^[A-Z][a-z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Hello'
    var_8 = 'hello'
    var_9 = 'HELLO'
    var_10 = '^[a-z]+\\.$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.'
    var_13 = 'test'
    var_14 = '.test'
    var_15 = 123
    var_16 = None
    var_17 = [var_13]
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@example.com'
    var_24 = 'invalid.email@'
    var_25 = 'noatsign.com'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'
    var_18 = 'aXb'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = '^a\\.b$'
    var_12 = module_0.rex(var_11)
    var_13 = 'a.b'
    var_14 = 'aXb'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^([a-z]+)_(\\d{4})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'file_2023'
    var_21 = 'file_23'
    var_22 = 'FILE_2023'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = '^hello.*'
    var_5 = module_0.rex(var_4)
    var_6 = 'goodbye'
    var_7 = module_0.rex(var_0)
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'aXb'
    var_18 = '^HELLO$'
    var_19 = module_0.rex(var_18)
    var_20 = 'HELLO'
    var_21 = '(?i)^hello$'
    var_22 = module_0.rex(var_21)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'abc_'
    var_12 = '_123'
    var_13 = 'ABC_123'
    var_14 = '^\\w+@\\w+\\.\\w+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'user@example.com'
    var_17 = 'user@example'
    var_18 = 'user@.com'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabcy'
    var_6 = 'xyz'
    var_7 = 123
    var_8 = None
    var_9 = '^a.c$'
    var_10 = module_0.rex(var_9)
    var_11 = 'axc'
    var_12 = 'a-c'
    var_13 = 'ac'
    var_14 = '^ABC$'
    var_15 = module_0.rex(var_14)
    var_16 = 'ABC'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = 'a'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '\\d{3}-\\d{2}-\\d{4}'
    var_10 = module_0.rex(var_9)
    var_11 = '123-45-6789'
    var_12 = '12-34-5678'
    var_13 = '1234-56-7890'
    var_14 = '[A-Z][a-z]+'
    var_15 = module_0.rex(var_14)
    var_16 = 'Hello'
    var_17 = 'hello'
    var_18 = 'HELLO'
    var_19 = 'foo\\.bar'
    var_20 = module_0.rex(var_19)
    var_21 = 'foo.bar'
    var_22 = 'foobar'
    var_23 = 'fooXbar'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'test\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test123'
    var_4 = 'test456'
    var_5 = 'test'
    var_6 = '123test'
    var_7 = 123
    var_8 = None
    var_9 = [var_3]
    var_10 = '^hello$'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello'
    var_13 = 'helloworld'
    var_14 = 'sayhello'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '^test\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.txt'
    var_17 = 'testxt'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_10 = module_0.rex(var_9)
    var_11 = '123-45-6789'
    var_12 = '12-34-5678'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'invalid-email'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'testing'
    var_4 = 'not_test'
    var_5 = 123
    var_6 = None
    var_7 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_8 = module_0.rex(var_7)
    var_9 = '123-45-6789'
    var_10 = '12-34-5678'
    var_11 = 'abc-def-ghij'
    var_12 = '^[A-Z]'
    var_13 = module_0.rex(var_12)
    var_14 = 'Hello'
    var_15 = 'hello'
    var_16 = '^.*\\.$'
    var_17 = module_0.rex(var_16)
    var_18 = 'file.txt'
    var_19 = 'file'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '^abc.*'
    var_5 = module_0.rex(var_4)
    var_6 = 'abc123'
    var_7 = 'xyz'
    var_8 = module_0.rex(var_0)
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = 'a'
    var_15 = '^a\\.b$'
    var_16 = module_0.rex(var_15)
    var_17 = 'a.b'
    var_18 = 'ab'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = 123
    var_12 = None
    var_13 = [var_2]
    var_14 = '^user@\\w+\\.com$'
    var_15 = module_0.rex(var_14)
    var_16 = 'user@example.com'
    var_17 = 'user@example.com.'
    var_18 = 'user@example'
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^([a-zA-Z]+)(\\d{2,4})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'abc123'
    var_25 = '123abc'
    var_26 = 'abc12'
    var_27 = 'abc1'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 'no_match'
    var_5 = 123
    var_6 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'user@example.com'
    var_9 = 'invalid.email'
    var_10 = 'another@test.org'
    var_11 = '.*'
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = ''



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '[A-Z][a-z]+'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^\\w+@\\w+\\.\\w+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example'
    var_15 = 'user@.com'
    var_16 = 123
    var_17 = None
    var_18 = 'test'
    var_19 = [var_18]
    var_20 = ''
    var_21 = module_0.rex(var_20)
    var_22 = 'anything'
    var_23 = '^([a-zA-Z]+)(\\d+)$'
    var_24 = module_0.rex(var_23)
    var_25 = 'abc123'
    var_26 = '123abc'
    var_27 = 'abc'
    var_28 = '123'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^test\\.txt$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.txt'
    var_13 = 'testxt'
    var_14 = '123'
    var_15 = None
    var_16 = 123
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = ' '
    var_21 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@domain.com'
    var_24 = 'user@domain.org'
    var_25 = 'user@.com'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'abc_test'
    var_4 = 'test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9]+\\.[a-zA-Z0-9]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'invalid.email'
    var_12 = 'another@valid.com'
    var_13 = '^CaseSensitive$'
    var_14 = module_0.rex(var_13)
    var_15 = 'CaseSensitive'
    var_16 = 'casesensitive'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^[a-z]+\\.$'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello.'
    var_12 = 'hello'
    var_13 = 'hello!'
    var_14 = 123
    var_15 = None
    var_16 = 'test'
    var_17 = [var_16]
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@example.com'
    var_24 = 'invalid.email'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 123
    var_6 = None
    var_7 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_8 = module_0.rex(var_7)
    var_9 = '123-45-6789'
    var_10 = '12-34-5678'
    var_11 = '1234-56-7890'
    var_12 = '^[A-Z]'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '123'
    var_10 = None
    var_11 = 123
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '.*\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.txt.bak'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^(?P<name>\\w+)-(?P<value>\\d+)$'
    var_17 = module_0.rex(var_16)
    var_18 = 'name-123'
    var_19 = 'name-value'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'abc_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9]+\\.[a-zA-Z0-9]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid-email'
    var_13 = 'another.valid@email.co.uk'
    var_14 = '^[A-Z]'
    var_15 = module_0.rex(var_14)
    var_16 = 'Uppercase'
    var_17 = 'lowercase'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Test$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'aXb'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'other_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'invalid-email'
    var_12 = 'another.user@domain.co.uk'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '12-34-5678'
    var_20 = '1234-56-7890'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'foo_123'
    var_11 = 'bar_456'
    var_12 = 'baz_'
    var_13 = '_123'
    var_14 = 'FOO_123'
    var_15 = '^[A-Z][a-z]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'Hello'
    var_18 = 'hello'
    var_19 = 'HELLO'
    var_20 = 'H3llo'



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^[a-z]+\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'file.txt'
    var_14 = 'file.txt.bak'
    var_15 = 'file'
    var_16 = 123
    var_17 = None
    var_18 = 'test'
    var_19 = [var_18]
    var_20 = ''
    var_21 = module_0.rex(var_20)
    var_22 = 'anything'
    var_23 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_24 = module_0.rex(var_23)
    var_25 = 'user@example.com'
    var_26 = 'invalid.email@'
    var_27 = 'noatsign.com'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 123
    var_6 = None
    var_7 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_8 = module_0.rex(var_7)
    var_9 = '123-45-6789'
    var_10 = '12-34-5678'
    var_11 = '1234-56-7890'
    var_12 = '^[A-Z]'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabc'
    var_6 = 'xyz'
    var_7 = 123
    var_8 = None
    var_9 = [var_2]
    var_10 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test@example.com'
    var_13 = 'test@example'
    var_14 = 'test@example.com.'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'a'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc123def'
    var_11 = 'abcdef'
    var_12 = '^[a-zA-Z_][a-zA-Z0-9_]*$'
    var_13 = module_0.rex(var_12)
    var_14 = 'valid_var'
    var_15 = '1invalid_var'
    var_16 = 'invalid-var'



