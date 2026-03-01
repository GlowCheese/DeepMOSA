####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''
    var_14 = '\\d+'
    var_15 = module_0.rex(var_14)
    var_16 = '123'
    var_17 = '123abc'



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'testing'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = 'test'
    var_8 = [var_7]
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = None
    var_13 = '^\\d+$'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = 'abc'
    var_17 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test@example.com'
    var_20 = 'invalid_email'
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ''
    var_24 = 'a'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = 'test1'
    var_18 = 'test2'
    var_19 = 'other'
    var_20 = 'test3'
    var_21 = [var_17, var_18, var_19, var_20]
    var_22 = module_0.rex(var_0)
    var_23 = [k for k in var_21 if var_28(k)]



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = 'test_abc'
    var_12 = '^[A-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'HELLO'
    var_15 = 'Hello'
    var_16 = '123'
    var_17 = ''
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = 'any_string'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any_string'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'a.b'
    var_9 = module_0.rex(var_8)
    var_10 = 'axb'
    var_11 = 'ab'
    var_12 = '\\d{3}-\\d{2}'
    var_13 = module_0.rex(var_12)
    var_14 = '123-45'
    var_15 = '12-345'
    var_16 = 'abc-de'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'any'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello'
    var_12 = 'HELLO'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = callable(var_18)



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '[a-z]+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any_string'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[A-Z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'UPPER'
    var_16 = 'lower'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = 'a'
    var_21 = '^a+$'
    var_22 = module_0.rex(var_21)
    var_23 = 'aaa'
    var_24 = 'aaab'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = '\\d+'
    var_5 = module_0.rex(var_4)
    var_6 = '123'
    var_7 = 'abc'
    var_8 = '123abc'
    var_9 = '.*'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = None
    var_19 = '^test$'
    var_20 = module_0.rex(var_19)
    var_21 = 'test'
    var_22 = 'testing'
    var_23 = '^Hello'
    var_24 = module_0.rex(var_23)
    var_25 = 'Hello World'
    var_26 = 'hello world'
    var_27 = '^[A-Z][a-z]+\\d{2,3}$'
    var_28 = module_0.rex(var_27)
    var_29 = 'Test123'
    var_30 = 'Test12'
    var_31 = 'test123'
    var_32 = 'Test1234'
    var_33 = 'T123'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = {}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '.*ing$'
    var_15 = module_0.rex(var_14)
    var_16 = 'testing'
    var_17 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test@example.com'
    var_20 = 'invalid_email'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = 'abc123'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 'any_string'
    var_19 = '\\d{3}-\\d{2}-\\d{4}'
    var_20 = module_0.rex(var_19)
    var_21 = '123-45-6789'
    var_22 = '12-345-6789'
    var_23 = '123-45-678'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^a.b$'
    var_9 = module_0.rex(var_8)
    var_10 = 'a.b'
    var_11 = 'axb'
    var_12 = 'a.b.c'
    var_13 = '^hello$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello'
    var_16 = 'Hello'
    var_17 = '^\\w+@\\w+\\.\\w+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test@example.com'
    var_20 = 'invalid_email'
    var_21 = '@example.com'
    var_22 = '.*'
    var_23 = module_0.rex(var_22)
    var_24 = callable(var_23)
    var_25 = ''
    var_26 = module_0.rex(var_25)
    var_27 = 'any'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 'abc123'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = 'any string'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''
    var_14 = '\\d{3}-\\d{2}-\\d{4}'
    var_15 = module_0.rex(var_14)
    var_16 = '123-45-6789'
    var_17 = '12-45-6789'
    var_18 = '123-456-789'
    var_19 = 'abc-def-ghij'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 'abc123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'
    var_16 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '12-345-6789'
    var_20 = '123-45-678'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'abc123'
    var_11 = 'ABC'
    var_12 = ''



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^a.*b$'
    var_8 = module_0.rex(var_7)
    var_9 = 'ab'
    var_10 = 'axxxb'
    var_11 = 'ba'
    var_12 = 'a'
    var_13 = 'b'
    var_14 = '^\\d+$'
    var_15 = module_0.rex(var_14)
    var_16 = '123'
    var_17 = ''
    var_18 = '123a'
    var_19 = 'a123'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = 'abc123'
    var_15 = ''
    var_16 = '\\d+'
    var_17 = module_0.rex(var_16)
    var_18 = '123'
    var_19 = 'abc123def'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'any string'



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '[a-z]+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-345-6789'
    var_17 = '123-45-67890'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid-email'
    var_17 = 'a1'
    var_18 = 'b2'
    var_19 = 'c'
    var_20 = 'd3'
    var_21 = 'e'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = '^\\w\\d$'
    var_24 = module_0.rex(var_23)
    var_25 = [k for k in var_22 if pattern(k)]



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = {}
    var_10 = '[a-z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = '123'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = callable(var_19)
    var_21 = '^match$'
    var_22 = module_0.rex(var_21)
    var_23 = 'match'
    var_24 = module_0.rex(var_21)
    var_25 = 'nomatch'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^[A-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'HELLO'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '^[a-z]+_[a-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello_world'
    var_16 = 'test_case'
    var_17 = 'Hello_World'
    var_18 = 'hello-world'
    var_19 = 'hello'
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = callable(var_21)
    var_23 = module_0.rex(var_12)
    var_24 = 'any'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = 'a.b'
    var_11 = module_0.rex(var_10)
    var_12 = 'acb'
    var_13 = 'a b'
    var_14 = 'ab'
    var_15 = '\\d{3}-\\d{2}-\\d{4}'
    var_16 = module_0.rex(var_15)
    var_17 = '123-45-6789'
    var_18 = '12-345-6789'
    var_19 = 'abc-def-ghij'
    var_20 = ''
    var_21 = module_0.rex(var_20)
    var_22 = 'any string'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '[a-z]+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any_string'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^a(b)c$'
    var_18 = module_0.rex(var_17)
    var_19 = 0
    var_20 = 1



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = '[a-z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc'
    var_14 = 'ABC'
    var_15 = '123'
    var_16 = ''
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '123!@#'
    var_21 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_22 = module_0.rex(var_21)
    var_23 = '123-45-6789'
    var_24 = '12-345-6789'
    var_25 = '123-45-67890'
    var_26 = 'abc-de-fghi'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = '^\\d+$'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test@example.com'
    var_18 = 'invalid_email'
    var_19 = '^a.*b$'
    var_20 = module_0.rex(var_19)
    var_21 = 'axxxb'
    var_22 = 'ab'
    var_23 = 'ba'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^\\w+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'hello'
    var_17 = 'hello world'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'test@example.com'
    var_21 = 'invalid-email'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\w+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_function'
    var_3 = 'test_method'
    var_4 = 'test'
    var_5 = '_test'
    var_6 = 'test_'
    var_7 = 'not_test'
    var_8 = 'test-with-dash'
    var_9 = 123
    var_10 = None
    var_11 = [var_4]
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = '123abc'
    var_17 = ''
    var_18 = '^[A-Z][a-z]*$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Hello'
    var_21 = 'hello'
    var_22 = 'H'
    var_23 = '.*'
    var_24 = module_0.rex(var_23)
    var_25 = 'anything'
    var_26 = '123!@#'



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = '[a-z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc'
    var_14 = 'ABC'
    var_15 = '123'
    var_16 = ''
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = 'any string'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = 'hello123'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = 'any string'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid-email'
    var_17 = '^foo'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)
    var_20 = 'foo_bar'
    var_21 = 'bar_foo'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = 'hello123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = 'abc123'
    var_15 = ''
    var_16 = 'a'
    var_17 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-345-6789'
    var_21 = '123-456-789'
    var_22 = '123-45-67890'
    var_23 = 'abc-de-fghi'



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello'
    var_12 = 'HELLO'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = callable(var_18)



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_string'
    var_3 = 'string_test'
    var_4 = 123
    var_5 = None
    var_6 = []
    var_7 = module_0.rex(var_0)
    var_8 = 'test'
    var_9 = 'Test'
    var_10 = '^test_\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test_123'
    var_13 = 'test_abc'
    var_14 = 'prefix_test_123'
    var_15 = '^a\\.b$'
    var_16 = module_0.rex(var_15)
    var_17 = 'a.b'
    var_18 = 'a_b'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ''
    var_22 = 'a'
    var_23 = 'pattern'
    var_24 = module_0.rex(var_23)
    var_25 = 'group'



# Parsed testcases at query #39
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'testing'
    var_6 = 'no_match'
    var_7 = 'TEST'
    var_8 = 123
    var_9 = [var_4]
    var_10 = 'key'
    var_11 = {var_10: var_4}
    var_12 = '^\\d+$'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test@example.com'
    var_19 = 'invalid_email'
    var_20 = 'test1'
    var_21 = 'test2'
    var_22 = 'other'
    var_23 = 'test3'
    var_24 = [var_20, var_21, var_22, var_23, var_8]
    var_25 = module_0.rex(var_0)
    var_26 = [k for k in var_24 if var_33(k)]



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = None
    var_9 = '^[a-z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc'
    var_12 = 'ABC'
    var_13 = 'abc123'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 'any string'



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = {}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'invalid_email'
    var_18 = '^a'
    var_19 = module_0.rex(var_18)
    var_20 = 'apple'
    var_21 = 'banana'
    var_22 = 'apricot'



# Parsed testcases at query #43
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a'
    var_4 = '123'
    var_5 = 'a123b'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^test.*'
    var_9 = module_0.rex(var_8)
    var_10 = 'testing'
    var_11 = 'test'
    var_12 = 'atest'
    var_13 = True
    var_14 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_15 = module_0.rex(var_14)
    var_16 = '123-45-6789'
    var_17 = '12-345-6789'
    var_18 = '123-45-67890'
    var_19 = 123456789



# Parsed testcases at query #44
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = '[a-z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc'
    var_14 = 'ABC'
    var_15 = 'abc123'
    var_16 = '123'
    var_17 = '\\d{3}-\\d{2}-\\d{4}'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-345-6789'
    var_21 = '123-45-67890'
    var_22 = '123-45-678'
    var_23 = ''
    var_24 = module_0.rex(var_23)
    var_25 = 'any'



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'hello'
    var_11 = 'Hello'
    var_12 = 'hello123'
    var_13 = ''
    var_14 = '\\d{3}-\\d{2}-\\d{4}'
    var_15 = module_0.rex(var_14)
    var_16 = '123-45-6789'
    var_17 = '12-345-6789'
    var_18 = 'abc-def-ghij'
    var_19 = '.*'
    var_20 = module_0.rex(var_19)
    var_21 = 'any_string'
    var_22 = 'with spaces'



# Parsed testcases at query #46
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^foo'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)
    var_20 = 'foo_bar'
    var_21 = 'bar_foo'



# Parsed testcases at query #47
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = [var_2]
    var_7 = '^a.*z$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abcz'
    var_10 = module_0.rex(var_7)
    var_11 = 'abczy'
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'



# Parsed testcases at query #48
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = None
    var_9 = 'a.b'
    var_10 = module_0.rex(var_9)
    var_11 = 'acb'
    var_12 = 'a b'
    var_13 = 'ab'
    var_14 = '\\d{3}-\\d{2}'
    var_15 = module_0.rex(var_14)
    var_16 = '123-45'
    var_17 = '12-345'
    var_18 = 'abc-de'
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'any'
    var_22 = '^[A-Z][a-z]*$'
    var_23 = module_0.rex(var_22)
    var_24 = 'Hello'
    var_25 = 'hello'
    var_26 = 'HELLO'



# Parsed testcases at query #49
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^exact$'
    var_15 = module_0.rex(var_14)
    var_16 = 'exact'
    var_17 = 'exact_extra'
    var_18 = '^a.*b$'
    var_19 = module_0.rex(var_18)
    var_20 = 'axxxb'
    var_21 = 'axxxc'



# Parsed testcases at query #50
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'lowercase'
    var_16 = 'MixedCase'



# Parsed testcases at query #51
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = 'a.b'
    var_12 = module_0.rex(var_11)
    var_13 = 'acb'
    var_14 = 'a b'
    var_15 = 'ab'
    var_16 = '\\d{3}-\\d{2}-\\d{4}'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '12-345-6789'
    var_20 = '123-456-789'
    var_21 = ''
    var_22 = module_0.rex(var_21)
    var_23 = 'any string'
    var_24 = '^[A-Z][a-z]*$'
    var_25 = module_0.rex(var_24)
    var_26 = 'Hello'
    var_27 = 'hello'
    var_28 = 'HELLO'
    var_29 = 'Hello123'



# Parsed testcases at query #52
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other'
    var_7 = 'a.b'
    var_8 = module_0.rex(var_7)
    var_9 = 'axb'
    var_10 = 'a b'
    var_11 = 'ab'
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = ''



# Parsed testcases at query #53
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'lowercase'
    var_13 = 'MixedCase'
    var_14 = 'UPPERCASE'
    var_15 = 'with123'
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'any string'
    var_20 = 'another'
    var_21 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_22 = module_0.rex(var_21)
    var_23 = '123-45-6789'
    var_24 = '12-345-6789'
    var_25 = '123-456-789'



# Parsed testcases at query #54
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^\\w+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello'
    var_16 = 'hello world'
    var_17 = '^a.*z$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abcz'
    var_20 = 'axyz'



# Parsed testcases at query #55
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = 'test_value'
    var_14 = 'group'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'



# Parsed testcases at query #56
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other'
    var_7 = 'a.*b'
    var_8 = module_0.rex(var_7)
    var_9 = 'ab'
    var_10 = 'axxxb'
    var_11 = 'ba'
    var_12 = ''
    var_13 = '\\d+'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = 'abc'
    var_17 = '123abc'



# Parsed testcases at query #57
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = 'test_abc'
    var_12 = '^[A-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'HELLO'
    var_15 = 'Hello'
    var_16 = '123'



# Parsed testcases at query #58
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = ''
    var_12 = '^[a-z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'abc'
    var_15 = 'ABC'
    var_16 = 'abc123'



# Parsed testcases at query #59
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'testing'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = 'test'
    var_8 = [var_7]
    var_9 = 'key'
    var_10 = {var_9: var_7}
    var_11 = None
    var_12 = '^\\d+$'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test@example.com'
    var_19 = 'invalid_email'
    var_20 = '^foo'
    var_21 = module_0.rex(var_20)
    var_22 = module_0.rex(var_20)
    var_23 = 'foo_bar'



# Parsed testcases at query #60
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\w+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_function'
    var_3 = 'test_123'
    var_4 = 'invalid'
    var_5 = 'test-with-dash'
    var_6 = 123
    var_7 = 'test_list'
    var_8 = [var_7]
    var_9 = None
    var_10 = ''
    var_11 = 'test_'
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = '123abc'



# Parsed testcases at query #61
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = 'hello123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #62
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'any string'



# Parsed testcases at query #63
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = [var_2]
    var_7 = 'a.b'
    var_8 = module_0.rex(var_7)
    var_9 = 'axb'
    var_10 = 'ab'
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = ''



# Parsed testcases at query #64
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 'test_abc'
    var_7 = 123
    var_8 = [var_2]
    var_9 = None
    var_10 = '[a-z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = '123'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'
    var_19 = '123!@#'
    var_20 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_21 = module_0.rex(var_20)
    var_22 = '123-45-6789'
    var_23 = '12-345-6789'
    var_24 = '123-456-789'
    var_25 = 'abc-de-fghi'



# Parsed testcases at query #65
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '[a-z]+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'invalid_email'
    var_18 = '^a'
    var_19 = module_0.rex(var_18)
    var_20 = 'apple'
    var_21 = module_0.rex(var_18)
    var_22 = 'banana'



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a1'
    var_4 = 'a'
    var_5 = '123'
    var_6 = 'a123b'
    var_7 = 123
    var_8 = None
    var_9 = []
    var_10 = '^[A-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'ABC'
    var_13 = 'AbC'
    var_14 = 'abc'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = callable(var_19)



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test@example.com'
    var_18 = 'invalid-email'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = [var_2]
    var_7 = '^a.*z$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abcz'
    var_10 = module_0.rex(var_7)
    var_11 = 'abcy'
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^foo'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)
    var_20 = 'foo_bar'
    var_21 = 'bar_foo'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 'test_abc'
    var_7 = 123
    var_8 = None
    var_9 = []
    var_10 = {}
    var_11 = '[a-z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'hello'
    var_14 = 'HELLO'
    var_15 = '123'
    var_16 = '\\d{3}-\\d{2}-\\d{4}'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '12-345-6789'
    var_20 = 'abc-de-fghi'
    var_21 = '.*'
    var_22 = module_0.rex(var_21)
    var_23 = callable(var_22)
    var_24 = ''
    var_25 = module_0.rex(var_24)
    var_26 = 'anything'
    var_27 = '\\.\\*\\?\\+'
    var_28 = module_0.rex(var_27)
    var_29 = '.*?+'
    var_30 = 'test'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a'
    var_4 = '123'
    var_5 = 'a12b'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^test.*'
    var_9 = module_0.rex(var_8)
    var_10 = 'test_string'
    var_11 = 'not_test'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = 'any_string'
    var_15 = ''



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^[A-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'HELLO'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'user@domain.org'
    var_17 = 'Test@example.com'
    var_18 = 'test@example'
    var_19 = '@example.com'
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = callable(var_21)
    var_23 = 'any string'
    var_24 = '^\\d+\\.\\d+$'
    var_25 = module_0.rex(var_24)
    var_26 = '3.14'
    var_27 = '0.5'
    var_28 = '3'
    var_29 = '.5'
    var_30 = '3.'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^a\\.b$'
    var_8 = module_0.rex(var_7)
    var_9 = 'a.b'
    var_10 = 'ab'
    var_11 = '^[A-Z]+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'ABC'
    var_14 = 'abc'
    var_15 = '^\\w+@\\w+\\.\\w+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test@example.com'
    var_18 = 'invalid-email'
    var_19 = '.*'
    var_20 = module_0.rex(var_19)
    var_21 = callable(var_20)
    var_22 = 'any string'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = 'test_abc'
    var_6 = '123_test'
    var_7 = 123
    var_8 = [var_2]
    var_9 = 'key'
    var_10 = {var_9: var_2}
    var_11 = '[a-z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc'
    var_14 = 'ABC'
    var_15 = '123'
    var_16 = 'abc123'
    var_17 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-45-6789'
    var_21 = '123-456-789'
    var_22 = 'abc-de-fghi'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = 'a.b'
    var_11 = module_0.rex(var_10)
    var_12 = 'acb'
    var_13 = 'a b'
    var_14 = 'ab'
    var_15 = '\\d{3}-\\d{2}'
    var_16 = module_0.rex(var_15)
    var_17 = '123-45'
    var_18 = '12-345'
    var_19 = 'abc-de'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = 'a.b'
    var_11 = module_0.rex(var_10)
    var_12 = 'acb'
    var_13 = 'ab'
    var_14 = 'a\nb'
    var_15 = '\\d{3}-\\d{2}-\\d{4}'
    var_16 = module_0.rex(var_15)
    var_17 = '123-45-6789'
    var_18 = '12-345-6789'
    var_19 = '123-456-789'
    var_20 = ''
    var_21 = module_0.rex(var_20)
    var_22 = 'any string'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^[A-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'HELLO'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '^[a-z]{3}-\\d{2}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'abc-12'
    var_16 = 'xyz-99'
    var_17 = 'ABC-12'
    var_18 = 'abcd-12'
    var_19 = 'abc-123'
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = callable(var_21)
    var_23 = '^a\\.b$'
    var_24 = module_0.rex(var_23)
    var_25 = 'a.b'
    var_26 = 'aXb'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'hello'
    var_11 = 'Hello'
    var_12 = 'hello123'
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'any string'



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello'
    var_12 = 'HELLO'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'any'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = callable(var_18)
    var_20 = '\\d+\\.\\d+'
    var_21 = module_0.rex(var_20)
    var_22 = '3.14'
    var_23 = 'abc'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = 'hello123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'abc123'
    var_12 = 'ABC'
    var_13 = ''



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^[A-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'HELLO'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = 'HELLO123'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'any'
    var_16 = '^[a-z]+@[a-z]+\\.[a-z]{2,3}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test@example.com'
    var_19 = 'user@domain.org'
    var_20 = 'Test@example.com'
    var_21 = 'test@example'
    var_22 = 'test@example.comm'
    var_23 = '.*'
    var_24 = module_0.rex(var_23)
    var_25 = callable(var_24)
    var_26 = '^\\d+\\.\\d+$'
    var_27 = module_0.rex(var_26)
    var_28 = '3.14'
    var_29 = '0.5'
    var_30 = '3'
    var_31 = '3.'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[A-Z][a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'Hello'
    var_11 = 'hello'
    var_12 = 'HELLO'
    var_13 = 'Hello123'
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'any string'
    var_18 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_19 = module_0.rex(var_18)
    var_20 = '123-45-6789'
    var_21 = '12-345-6789'
    var_22 = '123-45-678'
    var_23 = 'abc-de-fghi'



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a'
    var_4 = '123'
    var_5 = 'a123b'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^test.*'
    var_9 = module_0.rex(var_8)
    var_10 = 'test_string'
    var_11 = 'not_test'
    var_12 = ''
    var_13 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-345-6789'
    var_17 = 'abc-def-ghij'



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = ''
    var_12 = 'a.b'
    var_13 = module_0.rex(var_12)
    var_14 = 'acb'
    var_15 = 'a b'
    var_16 = 'ab'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a123'
    var_3 = 'a'
    var_4 = '123'
    var_5 = 'a12b'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^test.*'
    var_9 = module_0.rex(var_8)
    var_10 = 'test_string'
    var_11 = 'not_test'
    var_12 = ''
    var_13 = '^\\d{3}-\\d{2}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45'
    var_16 = '12-345'
    var_17 = 'abc-de'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'invalid_email'
    var_18 = '^(\\d+)-(\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = '123-456'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc'
    var_12 = 'ABC'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = callable(var_1)
    var_18 = '^[A-Z][a-z]*$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Test'
    var_21 = 'TEST'
    var_22 = 'Test123'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[A-Za-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello'
    var_16 = 'hello123'
    var_17 = 'test_key'
    var_18 = 'other_key'
    var_19 = 1
    var_20 = 2
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.rex(var_0)



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\w+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_function'
    var_3 = 'test_123'
    var_4 = 'test_'
    var_5 = 'not_test'
    var_6 = 'test-with-dash'
    var_7 = ''
    var_8 = 123
    var_9 = None
    var_10 = 'test_list'
    var_11 = [var_10]
    var_12 = '\\d+'
    var_13 = module_0.rex(var_12)
    var_14 = '123'
    var_15 = 'abc'
    var_16 = '123abc'
    var_17 = 'abc123'
    var_18 = '^[A-Z][a-z]*$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Hello'
    var_21 = 'hello'
    var_22 = 'HELLO'
    var_23 = 'H'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'testing'
    var_5 = 'no_match'
    var_6 = '^\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = '123'
    var_9 = 'abc'
    var_10 = '123abc'
    var_11 = 123
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = '^Test'
    var_20 = module_0.rex(var_19)
    var_21 = 'Test'
    var_22 = 'test'
    var_23 = '^a\\.b$'
    var_24 = module_0.rex(var_23)
    var_25 = 'a.b'
    var_26 = 'aXb'
    var_27 = '^$'
    var_28 = module_0.rex(var_27)
    var_29 = ''
    var_30 = 'a'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = '^\\d+$'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test@example.com'
    var_18 = 'invalid-email'
    var_19 = '^a+b*$'
    var_20 = module_0.rex(var_19)
    var_21 = 'aaaab'
    var_22 = 'b'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = 'hello'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello world'
    var_16 = 'world hello'
    var_17 = '^TEST$'
    var_18 = module_0.rex(var_17)
    var_19 = 'TEST'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 123
    var_7 = '^[A-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'HELLO'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '^[a-z]+_[a-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello_world'
    var_16 = 'test_case'
    var_17 = 'Hello_World'
    var_18 = 'hello-world'
    var_19 = 'hello'
    var_20 = '.*'
    var_21 = module_0.rex(var_20)
    var_22 = callable(var_21)
    var_23 = module_0.rex(var_20)
    var_24 = '__call__'
    var_25 = hasattr(var_23, var_24)



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^exact$'
    var_14 = module_0.rex(var_13)
    var_15 = 'exact'
    var_16 = 'exact_extra'
    var_17 = 'pattern'
    var_18 = module_0.rex(var_17)
    var_19 = '__call__'
    var_20 = hasattr(var_18, var_19)



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'abc123'
    var_11 = 'ABC'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test0'
    var_4 = 'test'
    var_5 = '123test'
    var_6 = 'test_abc'
    var_7 = 123
    var_8 = None
    var_9 = [var_4]
    var_10 = '[a-z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'hello'
    var_13 = 'HELLO'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'any'
    var_17 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-345-6789'



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 'abc123'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = 'any string'



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any_string'



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^a.*b$'
    var_8 = module_0.rex(var_7)
    var_9 = 'ab'
    var_10 = 'axxxb'
    var_11 = 'ba'
    var_12 = 'a'
    var_13 = '^\\d+$'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = '12a3'
    var_17 = ''



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = 'abc123'
    var_13 = ''
    var_14 = '.*'
    var_15 = module_0.rex(var_14)
    var_16 = 'any_string'



# Parsed testcases at query #39
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^test'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other'
    var_7 = 'a.*b'
    var_8 = module_0.rex(var_7)
    var_9 = 'ab'
    var_10 = 'axxxb'
    var_11 = 'ba'
    var_12 = ''
    var_13 = '\\d{3}-\\d{2}'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45'
    var_16 = '12-345'
    var_17 = 'abc-de'



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '[a-z]+'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'ABC'
    var_12 = ''
    var_13 = '\\d{3}-\\d{2}-\\d{4}'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-345-6789'
    var_17 = 'abc-def-ghij'



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^a'
    var_18 = module_0.rex(var_17)
    var_19 = 'apple'
    var_20 = module_0.rex(var_17)
    var_21 = 'banana'



# Parsed testcases at query #43
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^\\w+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'hello'
    var_17 = 'hello world'
    var_18 = '^a.*b$'
    var_19 = module_1.compile(var_18)
    var_20 = module_0.rex(var_18)
    var_21 = 'axyzb'
    var_22 = module_1.match(var_21)
    var_23 = bool(var_22)



# Parsed testcases at query #44
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = None
    var_11 = ''
    var_12 = 'a.b'
    var_13 = module_0.rex(var_12)
    var_14 = 'acb'
    var_15 = 'ab'
    var_16 = '\\d{3}-\\d{2}'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45'
    var_19 = '12-345'



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^exact$'
    var_14 = module_0.rex(var_13)
    var_15 = 'exact'
    var_16 = 'exact_extra'
    var_17 = 'test1'
    var_18 = 'test2'
    var_19 = 'other'
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = {var_17: var_20, var_18: var_21, var_19: var_22}
    var_24 = module_0.rex(var_0)



# Parsed testcases at query #46
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^a'
    var_18 = module_0.rex(var_17)
    var_19 = 'apple'
    var_20 = module_0.rex(var_17)
    var_21 = 'banana'



# Parsed testcases at query #47
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'invalid_email'
    var_17 = '^foo'
    var_18 = module_0.rex(var_17)
    var_19 = module_0.rex(var_17)
    var_20 = 'foo_bar'
    var_21 = 'bar_foo'



# Parsed testcases at query #48
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^\\w+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'hello'
    var_17 = 'hello world'
    var_18 = '^a(b)c$'
    var_19 = module_0.rex(var_18)
    var_20 = 1



# Parsed testcases at query #49
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = '^[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc'
    var_11 = 'abc123'
    var_12 = 'ABC'
    var_13 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-345-6789'
    var_17 = '123-45-67890'
    var_18 = '.*'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = 'any string'
    var_22 = 'another'



# Parsed testcases at query #50
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = 'other_123'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'hello'
    var_10 = 'Hello'
    var_11 = '123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'



# Parsed testcases at query #51
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'testing'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = 'test'
    var_8 = [var_7]
    var_9 = 'key'
    var_10 = {var_9: var_7}
    var_11 = '^\\d+$'
    var_12 = module_0.rex(var_11)
    var_13 = '123'
    var_14 = 'abc'
    var_15 = '^exact$'
    var_16 = module_0.rex(var_15)
    var_17 = 'exact'
    var_18 = 'exact_extra'
    var_19 = 'pattern'
    var_20 = module_0.rex(var_19)
    var_21 = 'group'



# Parsed testcases at query #52
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'invalid_email'
    var_18 = 'test1'
    var_19 = 'test2'
    var_20 = 'other'
    var_21 = 'test3'
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = module_0.rex(var_0)
    var_24 = [k for k in var_22 if var_29(k)]



# Parsed testcases at query #53
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'other_string'
    var_6 = 123
    var_7 = None
    var_8 = [var_4]
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = '123'
    var_12 = 'abc'
    var_13 = '^exact$'
    var_14 = module_0.rex(var_13)
    var_15 = 'exact'
    var_16 = 'exact_extra'
    var_17 = 'prefix_exact'
    var_18 = 'a1'
    var_19 = 'b2'
    var_20 = 'c'
    var_21 = 'd3'
    var_22 = 'e4f'
    var_23 = [var_18, var_19, var_20, var_21, var_22]
    var_24 = '^[a-z]\\d$'
    var_25 = module_0.rex(var_24)
    var_26 = [k for k in var_23 if pattern(k)]



# Parsed testcases at query #54
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = None
    var_9 = '^[a-z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc'
    var_12 = 'ABC'
    var_13 = 'abc123'
    var_14 = ''
    var_15 = '.*'
    var_16 = module_0.rex(var_15)
    var_17 = 'any string'



# Parsed testcases at query #55
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = None
    var_8 = 'list'
    var_9 = [var_8]
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'invalid_email'
    var_18 = '^a'
    var_19 = module_0.rex(var_18)
    var_20 = 'apple'
    var_21 = 'banana'
    var_22 = 'apricot'
    var_23 = 'cherry'
    var_24 = [var_20, var_21, var_22, var_23]
    var_25 = [item for item in var_24 if matcher_a(item)]



# Parsed testcases at query #56
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 'abc123'
    var_12 = ''
    var_13 = '.*'
    var_14 = module_0.rex(var_13)
    var_15 = 'any string'
    var_16 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '12-345-6789'
    var_20 = '123-45-678'



# Parsed testcases at query #57
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = 'abc123'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 'any string'



# Parsed testcases at query #58
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test_string'
    var_4 = 'test'
    var_5 = 'no_match'
    var_6 = 123
    var_7 = [var_4]
    var_8 = 'key'
    var_9 = {var_8: var_4}
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = 'abc'
    var_14 = '^exact$'
    var_15 = module_0.rex(var_14)
    var_16 = 'exact'
    var_17 = 'exact_extra'
    var_18 = 'extra_exact'
    var_19 = 'test1'
    var_20 = 'test2'
    var_21 = 'other'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = {var_19: var_22, var_20: var_23, var_21: var_24}
    var_26 = module_0.rex(var_0)



# Parsed testcases at query #59
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 123
    var_6 = 'test123extra'
    var_7 = '^[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 'abc123'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = 'any string'



# Parsed testcases at query #60
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello'
    var_12 = 'HELLO'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = callable(var_18)
    var_20 = '\\.\\*\\+\\?'
    var_21 = module_0.rex(var_20)
    var_22 = '.*+?'
    var_23 = 'test'



# Parsed testcases at query #61
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = 'test'
    var_4 = '123test'
    var_5 = 'test123extra'
    var_6 = 123
    var_7 = [var_2]
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = '^[a-z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc'
    var_13 = 'ABC'
    var_14 = 'abc123'
    var_15 = ''
    var_16 = '.*'
    var_17 = module_0.rex(var_16)
    var_18 = 'any string'
    var_19 = 'another'



# Parsed testcases at query #62
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_0'
    var_4 = 'test_'
    var_5 = '123_test'
    var_6 = 123
    var_7 = None
    var_8 = []
    var_9 = '[a-z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc'
    var_12 = 'ABC'
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'any'
    var_17 = '.*'
    var_18 = module_0.rex(var_17)
    var_19 = callable(var_18)



# Parsed testcases at query #63
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_'
    var_4 = 'test_abc'
    var_5 = 123
    var_6 = [var_2]
    var_7 = 'a.*b'
    var_8 = module_0.rex(var_7)
    var_9 = 'ab'
    var_10 = 'axxxb'
    var_11 = 'ac'
    var_12 = ''
    var_13 = '\\d+'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = 'abc'
    var_17 = '123abc'



