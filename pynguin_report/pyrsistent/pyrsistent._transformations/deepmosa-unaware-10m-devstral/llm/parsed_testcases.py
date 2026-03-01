####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^.*\\.py$'
    var_14 = module_0.rex(var_13)
    var_15 = 'script.py'
    var_16 = 'script.pyc'
    var_17 = 'script'



# Parsed testcases at query #2
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
    var_18 = '^([a-zA-Z0-9]+)@([a-zA-Z0-9]+)\\.([a-zA-Z]{2,})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid@email'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = '123test'
    var_4 = 'test'
    var_5 = 'TEST'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@sub.example.com'
    var_13 = '^[0-9]{3}-[0-9]{2}-[0-9]{4}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-345-6789'
    var_17 = '12345-6789'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '.*'
    var_22 = module_0.rex(var_21)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^([a-z]+)_(\\d{3})$'
    var_7 = module_0.rex(var_6)
    var_8 = 'abc_123'
    var_9 = 'ABC_123'
    var_10 = 'abc_12'
    var_11 = '^test\\.\\d+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.456'
    var_14 = 'test456'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Hello$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Hello'
    var_8 = 'hello'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = 123
    var_14 = None
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@example.com'
    var_24 = 'invalid.email@'



# Parsed testcases at query #6
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
    var_11 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_12 = module_0.rex(var_11)
    var_13 = '123-45-6789'
    var_14 = '12-34-5678'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'



# Parsed testcases at query #7
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'aXb'
    var_20 = 'ab'
    var_21 = '^(\\w+)-(\\d+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'item-123'
    var_24 = 'item-abc'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_11 = '.*\\.txt$'
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



# Parsed testcases at query #2
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@.com'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^[0-9]{3}-[0-9]{2}-[0-9]{4}$'
    var_17 = module_0.rex(var_16)
    var_18 = '123-45-6789'
    var_19 = '123456789'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^user@\\w+\\.com$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'user@example.org'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'any'
    var_16 = '^(?P<name>\\w+)-(?P<id>\\d{3})$'
    var_17 = module_0.rex(var_16)
    var_18 = 'john-123'
    var_19 = 'john-12'



# Parsed testcases at query #4
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
    var_10 = '^test\\.txt$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.txt'
    var_13 = 'testxt'
    var_14 = 'test-txt'
    var_15 = '^TEST$'
    var_16 = module_0.rex(var_15)
    var_17 = 'TEST'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid-email'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'not_test'
    var_5 = 123
    var_6 = None
    var_7 = '^[a-z]{3}_\\d{2}$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc_12'
    var_10 = 'ab_123'
    var_11 = 'ABC_12'
    var_12 = '^test\\.py$'
    var_13 = module_0.rex(var_12)
    var_14 = 'test.py'
    var_15 = 'testpy'
    var_16 = 'testxpy'



# Parsed testcases at query #6
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
    var_10 = '^user@\\w+\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@example.com.'
    var_14 = 'user@example'
    var_15 = '123'
    var_16 = None
    var_17 = 123
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = ' '



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d{3}$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'xyz_987'
    var_12 = 'ABC_123'
    var_13 = 'abc_12'
    var_14 = '^[a-z]+\\.$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.'
    var_17 = 'abc.'
    var_18 = '.test'



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_8 = 'Abc'
    var_9 = 'abc'
    var_10 = 'ABC'
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
    var_22 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test.user@example.com'
    var_25 = 'invalid@email'
    var_26 = 'another.test@sub.domain.co.uk'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^[A-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = 'ABC123'
    var_11 = '^.*\\.txt$'
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



# Parsed testcases at query #11
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
    var_8 = [var_2]
    var_9 = ''
    var_10 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid.email@'
    var_14 = 'another.valid-one@example.co.uk'



# Parsed testcases at query #12
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
    var_9 = '^hello\\.world$'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello.world'
    var_12 = 'helloworld'
    var_13 = '123'
    var_14 = None
    var_15 = 123
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'a'
    var_20 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'invalid.email'



# Parsed testcases at query #13
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
    var_18 = '^([a-z]+)_(\\d{3})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'abc_123'
    var_21 = 'ABC_123'
    var_22 = 'abc_12'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 123
    var_5 = '^abc.*'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc123'
    var_8 = 'xyz'
    var_9 = module_0.rex(var_0)
    var_10 = None
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^a\\.b$'
    var_19 = module_0.rex(var_18)
    var_20 = 'a.b'
    var_21 = 'ab'
    var_22 = 'aXb'



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
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([A-Z][a-z]+)\\s(\\d{4})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Name 2023'
    var_21 = 'name 2023'
    var_22 = 'Name twenty'



# Parsed testcases at query #16
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
    var_17 = 'any'
    var_18 = '^([a-z]+)_(\\d{3})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'abc_123'
    var_21 = 'abc_12'
    var_22 = 'ABC_123'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'ab'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([a-z]+)_(\\d{3})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'abc_123'
    var_21 = 'ABC_123'
    var_22 = 'abc_12'



# Parsed testcases at query #18
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
    var_8 = 'Abc'
    var_9 = 'abc'
    var_10 = 'ABC'
    var_11 = 123
    var_12 = None
    var_13 = [var_2]
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'
    var_21 = 'test-txt'
    var_22 = '^(\\w+)-(\\d+)$'
    var_23 = module_0.rex(var_22)
    var_24 = 'file-123'
    var_25 = 'file-abc'
    var_26 = 'file'



# Parsed testcases at query #19
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'a'
    var_17 = '^([a-z]+)_(\\d{4})$'
    var_18 = module_0.rex(var_17)
    var_19 = 'file_2023'
    var_20 = 'file_23'



# Parsed testcases at query #20
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@.com'
    var_12 = '^[0-9]{3}-[0-9]{2}-[0-9]{4}$'
    var_13 = module_0.rex(var_12)
    var_14 = '123-45-6789'
    var_15 = '123456789'
    var_16 = '12-34-5678'



# Parsed testcases at query #21
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
    var_9 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'test@example'
    var_13 = 'test@.com'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 'TEST'
    var_6 = '(?i)^test'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 123
    var_10 = None
    var_11 = [var_2]
    var_12 = '^user_\\d+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user_123'
    var_15 = 'user_abc'
    var_16 = 'user_123_abc'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'user@example.com'
    var_9 = 'user@example'
    var_10 = 'user@.com'
    var_11 = 'user@example.com.'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = 'any string'
    var_15 = ''



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '[A-Z]+'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = 123
    var_11 = None
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^\\W+$'
    var_16 = module_0.rex(var_15)
    var_17 = '!@#'
    var_18 = '^(a|b|c)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'a'
    var_21 = 'b'
    var_22 = 'c'
    var_23 = 'd'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = None
    var_6 = 'hello'
    var_7 = module_0.rex(var_6)
    var_8 = 'hello world'
    var_9 = 'goodbye'
    var_10 = '[A-Z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'ABC'
    var_13 = 'abc'
    var_14 = '^(?P<name>[a-z]+)_(?P<num>\\d+)$'
    var_15 = module_0.rex(var_14)
    var_16 = 'name_123'
    var_17 = '123_name'



# Parsed testcases at query #26
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
    var_18 = '^(\\w+)-(\\w+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'hello-world'
    var_21 = 'hello'
    var_22 = '^a{2,3}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'aa'
    var_25 = 'aaa'
    var_26 = 'a'
    var_27 = 'aaaa'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'file.txt'
    var_17 = 'file.txt.bak'
    var_18 = '(\\w+)-(\\d+)'
    var_19 = module_0.rex(var_18)
    var_20 = 'file-123'
    var_21 = 'file123'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_[0-9]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'ABC_123'
    var_12 = 'abc_123_'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'a'



# Parsed testcases at query #29
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'user@example.org'
    var_24 = 'user@example'



# Parsed testcases at query #30
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
    var_7 = 'abx123'
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
    var_6 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'user@example.com'
    var_9 = 'invalid.email'
    var_10 = 'another@test.co.uk'
    var_11 = '.*'
    var_12 = module_0.rex(var_11)
    var_13 = 'any string'
    var_14 = ''
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = 'not empty'



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
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^test$'
    var_11 = module_0.rex(var_10)
    var_12 = 123
    var_13 = None
    var_14 = [var_9]
    var_15 = '^test\\.$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.'
    var_18 = 'test..'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ''
    var_22 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'user@example.com'
    var_25 = 'invalid.email@'
    var_26 = 'another.test@domain.co.uk'



# Parsed testcases at query #33
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
    var_9 = '^\\w+@\\w+\\.\\w+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid@email'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^(?P<name>\\w+)-(?P<value>\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'name-123'
    var_21 = 'invalid'



# Parsed testcases at query #34
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
    var_11 = 'ABC_123'
    var_12 = 'abc_def'
    var_13 = '123_abc'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^\\d+\\.\\d+$'
    var_18 = module_0.rex(var_17)
    var_19 = '3.14'
    var_20 = '314'
    var_21 = '3.14.15'



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
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^[a-z]+\\.$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.'
    var_14 = 'test'
    var_15 = '.test'
    var_16 = '^\\d+$'
    var_17 = module_0.rex(var_16)
    var_18 = 123
    var_19 = None
    var_20 = [var_14]
    var_21 = '^$'
    var_22 = module_0.rex(var_21)
    var_23 = ''
    var_24 = ' '
    var_25 = '^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$'
    var_26 = module_0.rex(var_25)
    var_27 = 'user@example.com'
    var_28 = 'invalid.email@'
    var_29 = 'another.test@domain.co.uk'



# Parsed testcases at query #36
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
    var_8 = ''
    var_9 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'user@example'
    var_13 = 'user@.com'
    var_14 = 'user@example.com.'
    var_15 = '^[a-zA-Z0-9_]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'valid_name_123'
    var_18 = 'invalid@name'
    var_19 = 'invalid name'



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = '\\d+'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = 'abc'
    var_9 = '^(a|b)c$'
    var_10 = module_0.rex(var_9)
    var_11 = 'ac'
    var_12 = 'bc'
    var_13 = 'cc'
    var_14 = module_0.rex(var_5)
    var_15 = 123
    var_16 = None
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = '[A-Z]'
    var_20 = module_0.rex(var_19)
    var_21 = 'A'
    var_22 = 'a'



# Parsed testcases at query #38
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
    var_8 = [var_2]
    var_9 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9]+\\.[a-zA-Z0-9]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email'
    var_13 = 'another@valid.org'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'not empty'
    var_18 = '^[0-9]{3}-[0-9]{2}-[0-9]{4}$'
    var_19 = module_0.rex(var_18)
    var_20 = '123-45-6789'
    var_21 = '12-34-5678'
    var_22 = '123456789'



# Parsed testcases at query #39
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
    var_8 = '^[A-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'ABC'
    var_11 = 'abc'
    var_12 = '^test\\$value$'
    var_13 = module_0.rex(var_12)
    var_14 = 'test$value'
    var_15 = 'testvalue'
    var_16 = '^(\\w+)-(\\d+)$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test-123'
    var_19 = 'test123'



# Parsed testcases at query #40
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
    var_11 = 'test'
    var_12 = [var_11]
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^\\w+@\\w+\\.\\w+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'invalid@email'
    var_20 = '^(\\d{3})-(\\d{3})-(\\d{4})$'
    var_21 = module_0.rex(var_20)
    var_22 = '123-456-7890'
    var_23 = '1234567890'



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^abc.*'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc123'
    var_8 = 'xyz'
    var_9 = module_0.rex(var_0)
    var_10 = 123
    var_11 = None
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_12, var_13, var_14]
    var_16 = '^a.c$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a1c'
    var_19 = 'ac'
    var_20 = ''
    var_21 = module_0.rex(var_20)
    var_22 = '^ABC$'
    var_23 = module_0.rex(var_22)
    var_24 = 'ABC'



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^[A-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = 123
    var_11 = None
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^test\\.$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.'
    var_18 = 'test'
    var_19 = 'test..'



# Parsed testcases at query #43
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
    var_13 = ' '
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'



# Parsed testcases at query #44
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = 'ab'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@example.com'
    var_24 = 'invalid.email@'
    var_25 = 'another.valid-one@domain.co.uk'



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'abcabc'
    var_5 = 'a.*c'
    var_6 = module_0.rex(var_5)
    var_7 = 'a123c'
    var_8 = 'ac'
    var_9 = 'ab'
    var_10 = 123
    var_11 = None
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_12, var_13, var_14]
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''



# Parsed testcases at query #46
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'user@example.org'
    var_24 = 'user@example'



# Parsed testcases at query #47
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = '\\d+'
    var_6 = module_0.rex(var_5)
    var_7 = 123
    var_8 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'invalid-email'
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'file.txt'
    var_18 = 'file.txt.bak'



# Parsed testcases at query #48
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
    var_11 = '^test\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.txt'
    var_14 = 'testxt'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'any'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #49
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
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = '^\\w+@\\w+\\.\\w+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'user@example'
    var_16 = 'user@.com'
    var_17 = '^Test$'
    var_18 = module_0.rex(var_17)
    var_19 = 'Test'
    var_20 = 'test'



# Parsed testcases at query #50
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
    var_8 = '^[a-z]+@[a-z]+\\.[a-z]{2,3}$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example.co.uk'
    var_12 = 'USER@EXAMPLE.COM'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^[0-9]{3}-[0-9]{2}-[0-9]{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-34-5678'



# Parsed testcases at query #51
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



# Parsed testcases at query #52
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
    var_15 = '^.*\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'file.txt'
    var_18 = 'file.txt.bak'
    var_19 = 'file'



# Parsed testcases at query #53
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'helloworld'
    var_4 = 'worldhello'
    var_5 = '^hello.*world$'
    var_6 = module_0.rex(var_5)
    var_7 = 'hello123world'
    var_8 = 'helloworld123'
    var_9 = '123helloworld'
    var_10 = 123
    var_11 = None
    var_12 = [var_2]
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^hello\\.world$'
    var_18 = module_0.rex(var_17)
    var_19 = 'hello.world'



# Parsed testcases at query #54
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
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'a'



# Parsed testcases at query #55
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
    var_11 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'invalid.email@'
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z0-9]+)@([a-zA-Z0-9]+\\.[a-zA-Z0-9]+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'invalid-email'



# Parsed testcases at query #57
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_123'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'ABC_123'
    var_12 = 'abc_'
    var_13 = '_123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testTxt'



# Parsed testcases at query #58
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^test\\.txt$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.txt'
    var_13 = 'testxt'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^([a-zA-Z]+)_\\d{3}_(\\w+)$'
    var_18 = module_0.rex(var_17)
    var_19 = 'prefix_123_suffix'
    var_20 = 'prefix_12_suffix'
    var_21 = '123_prefix_suffix'



# Parsed testcases at query #59
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
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = 123
    var_14 = '123'
    var_15 = None
    var_16 = '\\w+@\\w+\\.\\w+'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example'
    var_20 = 'user@.com'
    var_21 = ''
    var_22 = module_0.rex(var_21)
    var_23 = 'anything'
    var_24 = '^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)[a-zA-Z\\d]{8,}$'
    var_25 = module_0.rex(var_24)
    var_26 = 'Password123'
    var_27 = 'password'
    var_28 = 'PASSWORD'
    var_29 = 'Pass123'



# Parsed testcases at query #60
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
    var_10 = '^test\\.\\*$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.*'
    var_13 = 'test*'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '
    var_18 = '^([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #61
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
    var_10 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid@email'
    var_14 = 'user@.com'



# Parsed testcases at query #62
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = '123'
    var_4 = 'abc'
    var_5 = '123abc'
    var_6 = 123
    var_7 = None
    var_8 = [var_3]
    var_9 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'invalid-email'
    var_13 = 'another.test@sub.domain.co.uk'



# Parsed testcases at query #63
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
    var_8 = [var_2]
    var_9 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email'
    var_13 = 'another.user@sub.domain.co.uk'
    var_14 = '^CaseSensitive$'
    var_15 = module_0.rex(var_14)
    var_16 = 'CaseSensitive'
    var_17 = 'casesensitive'
    var_18 = '^test\\.txt$'
    var_19 = module_0.rex(var_18)
    var_20 = 'test.txt'
    var_21 = 'testTxt'



# Parsed testcases at query #64
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@example.com.'
    var_13 = '\\d+'
    var_14 = module_0.rex(var_13)
    var_15 = 'abc123def'
    var_16 = 'abcdef'
    var_17 = '^[A-Z]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'ABC'
    var_20 = 'abc'



# Parsed testcases at query #65
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 'a.*c'
    var_5 = module_0.rex(var_4)
    var_6 = 'a123c'
    var_7 = 'ac'
    var_8 = 'ab'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'a'
    var_14 = 'a\\.b'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'



# Parsed testcases at query #66
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@example.com.'
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_16 = module_0.rex(var_15)
    var_17 = '123-45-6789'
    var_18 = '12-34-5678'



# Parsed testcases at query #67
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]'
    var_6 = module_0.rex(var_5)
    var_7 = 'A'
    var_8 = 'a'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'any'
    var_14 = '^(?P<name>[a-zA-Z]+)_(?P<num>\\d+)$'
    var_15 = module_0.rex(var_14)
    var_16 = 'name_123'
    var_17 = 'name_abc'



# Parsed testcases at query #68
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabcy'
    var_6 = 'ab'
    var_7 = 123
    var_8 = None
    var_9 = '^a.c$'
    var_10 = module_0.rex(var_9)
    var_11 = 'axc'
    var_12 = 'ac'
    var_13 = '^(abc)(def)$'
    var_14 = module_0.rex(var_13)
    var_15 = 'abcdef'



# Parsed testcases at query #69
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
    var_11 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'invalid.email@'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^a\\.b$'
    var_19 = module_0.rex(var_18)
    var_20 = 'a.b'
    var_21 = 'ab'



# Parsed testcases at query #70
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 123
    var_6 = None
    var_7 = ''
    var_8 = '\\d{3}-\\d{2}-\\d{4}'
    var_9 = module_0.rex(var_8)
    var_10 = '123-45-6789'
    var_11 = '12-34-5678'
    var_12 = '1234-56-7890'
    var_13 = '[A-Z][a-z]+'
    var_14 = module_0.rex(var_13)
    var_15 = 'Hello'
    var_16 = 'hello'
    var_17 = 'HELLO'
    var_18 = '^test\\.txt$'
    var_19 = module_0.rex(var_18)
    var_20 = 'test.txt'
    var_21 = 'testxt'
    var_22 = 'test-txt'
    var_23 = '^(\\w+)-(\\d+)$'
    var_24 = module_0.rex(var_23)
    var_25 = 'file-123'
    var_26 = 'file-abc'
    var_27 = 'file'



# Parsed testcases at query #71
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@example.com.org'
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'



# Parsed testcases at query #72
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
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'file_123'
    var_22 = 'file_abc'



# Parsed testcases at query #73
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
    var_9 = '^.*\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.txt.bak'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'any'
    var_18 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email@'



# Parsed testcases at query #74
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
    var_9 = 'any'
    var_10 = '[A-Z]+'
    var_11 = module_0.rex(var_10)
    var_12 = 'ABC'
    var_13 = 'abc'
    var_14 = '^\\W+$'
    var_15 = module_0.rex(var_14)
    var_16 = '!@#'



# Parsed testcases at query #75
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '^a.*c$'
    var_5 = module_0.rex(var_4)
    var_6 = 'a123c'
    var_7 = 'ac'
    var_8 = 123
    var_9 = None
    var_10 = '^[a-z]+_[0-9]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'abc_123'
    var_13 = 'ABC_123'
    var_14 = 'abc123'
    var_15 = 'abc_'



# Parsed testcases at query #76
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
    var_9 = '^hello\\.world$'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello.world'
    var_12 = 'helloworld'
    var_13 = 123
    var_14 = None
    var_15 = [var_8]
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'a'
    var_20 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'invalid.email'



# Parsed testcases at query #77
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
    var_9 = 'any'
    var_10 = '^\\w+@\\w+\\.\\w+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid@email'
    var_14 = '^Test$'
    var_15 = module_0.rex(var_14)
    var_16 = 'Test'
    var_17 = 'test'



# Parsed testcases at query #78
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^\\w+@\\w+\\.\\w+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example'
    var_20 = 'user@.com'



# Parsed testcases at query #79
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #80
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'testing'
    var_4 = 'test123'
    var_5 = 'not_test'
    var_6 = 'test_not'
    var_7 = 123
    var_8 = None
    var_9 = [var_2]
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a3'
    var_14 = ''
    var_15 = '^Test'
    var_16 = module_0.rex(var_15)
    var_17 = 'Test'
    var_18 = '^te.st$'
    var_19 = module_0.rex(var_18)
    var_20 = 'te.st'
    var_21 = 'teast'



# Parsed testcases at query #81
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
    var_10 = 'HELLO'
    var_11 = '123Hello'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^[a-z]+\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'file.txt'
    var_19 = 'file.txt.bak'
    var_20 = 'fileTXT'



# Parsed testcases at query #82
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
    var_7 = '^[a-z]+_[0-9]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc_123'
    var_10 = 'abc_123_'
    var_11 = '_123'
    var_12 = 'abc_'
    var_13 = '^TEST$'
    var_14 = module_0.rex(var_13)
    var_15 = 'TEST'
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'



# Parsed testcases at query #83
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
    var_10 = 'say hello'
    var_11 = '\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'file.txt'
    var_14 = 'file.txt.bak'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'



# Parsed testcases at query #84
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
    var_7 = 'abx123'
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'



# Parsed testcases at query #85
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
    var_10 = 'HELLO'
    var_11 = 'Hello123'
    var_12 = '.*'
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = ''
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ' '
    var_19 = 'a'



# Parsed testcases at query #86
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
    var_10 = 'a.c'
    var_11 = module_0.rex(var_10)
    var_12 = 'axc'
    var_13 = 'ac'
    var_14 = 'a(b|c)d'
    var_15 = module_0.rex(var_14)
    var_16 = 'abd'
    var_17 = 'acd'
    var_18 = 'ad'
    var_19 = '(?i)abc'
    var_20 = module_0.rex(var_19)
    var_21 = 'ABC'
    var_22 = 'AbC'



# Parsed testcases at query #87
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
    var_23 = '^(?P<name>[a-zA-Z]+)-(?P<value>\\d+)$'
    var_24 = module_0.rex(var_23)
    var_25 = 'count-42'
    var_26 = 'count-42-extra'
    var_27 = '42-count'



# Parsed testcases at query #88
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
    var_8 = '^[a-z]+_\\d{3}$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'abc_12'
    var_12 = 'ABC_123'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 'testTxt'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'



# Parsed testcases at query #89
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
    var_10 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@example.org'
    var_14 = 'user@sub.example.com'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = '^TEST$'
    var_20 = module_0.rex(var_19)
    var_21 = 'TEST'
    var_22 = 'test'



# Parsed testcases at query #90
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = '^hello.*world$'
    var_5 = module_0.rex(var_4)
    var_6 = module_0.rex(var_0)
    var_7 = 123
    var_8 = None
    var_9 = '^$'
    var_10 = module_0.rex(var_9)
    var_11 = ''
    var_12 = ' '
    var_13 = '^\\d+$'
    var_14 = module_0.rex(var_13)
    var_15 = '123'
    var_16 = 'abc'



# Parsed testcases at query #91
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
    var_7 = ''
    var_8 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_9 = module_0.rex(var_8)
    var_10 = '123-45-6789'
    var_11 = '12-34-5678'
    var_12 = '1234-56-7890'
    var_13 = '^(?i)hello$'
    var_14 = module_0.rex(var_13)
    var_15 = 'hello'
    var_16 = 'HELLO'
    var_17 = 'Hello'
    var_18 = 'hElLo'



# Parsed testcases at query #92
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
    var_16 = '123'
    var_17 = 123
    var_18 = None
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^(\\d{3})-(\\d{3})-(\\d{4})$'
    var_23 = module_0.rex(var_22)
    var_24 = '123-456-7890'
    var_25 = '1234567890'
    var_26 = '12-34-5678'



# Parsed testcases at query #93
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
    var_7 = 'ab123'
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



# Parsed testcases at query #94
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '[A-Z]+'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = 123
    var_11 = None
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^\\w+@\\w+\\.\\w+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example'



# Parsed testcases at query #95
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
    var_12 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'invalid.email'
    var_16 = 'another.valid@sub.domain.co.uk'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^a\\.b$'
    var_21 = module_0.rex(var_20)
    var_22 = 'a.b'
    var_23 = 'ab'
    var_24 = 'aXb'



# Parsed testcases at query #96
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
    var_9 = 'hello_world'
    var_10 = 'goodbye'
    var_11 = '\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'file.txt'
    var_14 = 'file.txt.bak'
    var_15 = '[A-Z]+'
    var_16 = module_0.rex(var_15)
    var_17 = 'ABC'
    var_18 = 'abc'



# Parsed testcases at query #97
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
    var_8 = [var_2]
    var_9 = '^[a-z]+_\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc_123'
    var_12 = 'ABC_123'
    var_13 = 'abc_123_'
    var_14 = '_abc_123'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'abc'
    var_18 = '^\\w+@\\w+\\.\\w+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'test@example.com'
    var_21 = 'test@example'
    var_22 = 'test@example..com'



# Parsed testcases at query #98
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = callable(var_1)
    var_3 = 'test'
    var_4 = 'testing'
    var_5 = 'test123'
    var_6 = 'not_test'
    var_7 = 'test_not'
    var_8 = '123test'
    var_9 = 123
    var_10 = None
    var_11 = [var_3]
    var_12 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_13 = module_0.rex(var_12)
    var_14 = '123-45-6789'
    var_15 = '12-34-5678'
    var_16 = '1234-56-7890'



# Parsed testcases at query #99
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'abc_test'
    var_5 = 'test'
    var_6 = '^TEST'
    var_7 = module_0.rex(var_6)
    var_8 = 'TEST'
    var_9 = 123
    var_10 = None
    var_11 = '^[a-z]+_\\d{3}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'abc_123'
    var_14 = 'abc_12'
    var_15 = 'ABC_123'
    var_16 = 'abc_1234'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = 'a'



# Parsed testcases at query #100
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
    var_11 = '^a.c$'
    var_12 = module_0.rex(var_11)
    var_13 = 'a1c'
    var_14 = 'ac'
    var_15 = '^ABC$'
    var_16 = module_0.rex(var_15)
    var_17 = 'ABC'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = 'a'



# Parsed testcases at query #101
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
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'ab'



# Parsed testcases at query #102
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
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'



# Parsed testcases at query #103
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^CaseSensitive$'
    var_7 = module_0.rex(var_6)
    var_8 = 'CaseSensitive'
    var_9 = 'casesensitive'
    var_10 = '^special\\$chars\\.test$'
    var_11 = module_0.rex(var_10)
    var_12 = 'special$chars.test'
    var_13 = 'specialchars.test'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'not empty'



# Parsed testcases at query #104
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc123def'
    var_12 = 'abcdef'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 'test-txt'
    var_18 = '^Test$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Test'
    var_21 = 'test'
    var_22 = '^(\\w+)_(\\d+)$'
    var_23 = module_0.rex(var_22)
    var_24 = 'abc_123'
    var_25 = 'abc_def'
    var_26 = '^a{2,4}$'
    var_27 = module_0.rex(var_26)
    var_28 = 'aa'
    var_29 = 'aaa'
    var_30 = 'aaaa'
    var_31 = 'a'
    var_32 = 'aaaaa'
    var_33 = '^[A-Z][a-z]+$'
    var_34 = module_0.rex(var_33)
    var_35 = 'Abc'
    var_36 = 'abc'
    var_37 = 'ABC'
    var_38 = 'A1bc'



# Parsed testcases at query #105
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^Test$'
    var_10 = module_0.rex(var_9)
    var_11 = 'Test'
    var_12 = 'test'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 'test-txt'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = '^([a-zA-Z]+)_(\\d{4})(\\.pdf|\\.txt)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'document_2023.pdf'
    var_24 = 'Document_2023.txt'
    var_25 = 'doc_23.pdf'
    var_26 = 'document_2023.jpg'



# Parsed testcases at query #106
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'a'
    var_17 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'user@example.com'
    var_20 = 'invalid-email'



# Parsed testcases at query #107
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
    var_13 = 'any'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'aXb'
    var_18 = '^(\\w+)-(\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'test-123'
    var_21 = 'test-abc'



# Parsed testcases at query #108
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



# Parsed testcases at query #109
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
    var_9 = '^special\\$char'
    var_10 = module_0.rex(var_9)
    var_11 = 'special$char'
    var_12 = 'specialchar'
    var_13 = '123'
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^([a-zA-Z]+)@([a-zA-Z]+\\.[a-zA-Z]+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #110
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
    var_18 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'user@example.org'



# Parsed testcases at query #111
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
    var_9 = '^hello\\.world$'
    var_10 = module_0.rex(var_9)
    var_11 = 'hello.world'
    var_12 = 'helloworld'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #112
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
    var_7 = '^[A-Z][a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'Hello'
    var_10 = 'hello'
    var_11 = 'HELLO'
    var_12 = '123Hello'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'invalid.email@com'
    var_20 = 'user@example'



# Parsed testcases at query #113
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
    var_10 = '^user@\\w+\\.com$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@example.com.'
    var_14 = 'user@example'
    var_15 = '123'
    var_16 = None
    var_17 = 123
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^(\\d{3})-(\\d{2})-(\\d{4})$'
    var_22 = module_0.rex(var_21)
    var_23 = '123-45-6789'
    var_24 = '1234-56-7890'
    var_25 = '12-34-5678'



# Parsed testcases at query #114
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
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = '^[a-zA-Z]+_\\d{3}_[a-z]{2}$'
    var_13 = module_0.rex(var_12)
    var_14 = 'abc_123_xy'
    var_15 = 'ABC_456_zz'
    var_16 = 'abc_12_xy'
    var_17 = 'abc_1234_xy'
    var_18 = 'abc_123_12'



# Parsed testcases at query #115
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
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'
    var_21 = 'test-txt'
    var_22 = '^test.*$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test123'
    var_25 = 'testing'
    var_26 = 'test!@#'



# Parsed testcases at query #116
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
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'file_123'
    var_22 = 'file_abc'
    var_23 = '^a{2,4}$'
    var_24 = module_0.rex(var_23)
    var_25 = 'aa'
    var_26 = 'aaa'
    var_27 = 'aaaa'
    var_28 = 'a'
    var_29 = 'aaaaa'



# Parsed testcases at query #117
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
    var_10 = '^test\\.$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.'
    var_13 = '123'
    var_14 = None
    var_15 = 123
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'
    var_19 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'user@example.com'
    var_22 = 'invalid.email'



# Parsed testcases at query #118
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = '^TEST'
    var_7 = module_0.rex(var_6)
    var_8 = 'TEST'
    var_9 = 123
    var_10 = None
    var_11 = [var_4]
    var_12 = '^test\\.$'
    var_13 = module_0.rex(var_12)
    var_14 = 'test.'
    var_15 = 'test..'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'abc'
    var_19 = '^test_(.*)'
    var_20 = module_0.rex(var_19)
    var_21 = 'test_'



# Parsed testcases at query #119
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'foo_test'
    var_4 = 'test'
    var_5 = 123
    var_6 = None
    var_7 = '^[a-z]+_\\d{3}$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc_123'
    var_10 = 'ABC_123'
    var_11 = 'abc_12'
    var_12 = 'abc_1234'
    var_13 = '^file\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'file.txt'
    var_16 = 'filextxt'
    var_17 = 'file.txt.bak'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^(\\w+)-(\\d+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'item-123'
    var_24 = 'item-abc'



# Parsed testcases at query #120
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test123'
    var_3 = '123test'
    var_4 = 'test'
    var_5 = '^Test'
    var_6 = module_0.rex(var_5)
    var_7 = 'Test'
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = 123
    var_11 = '123'
    var_12 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'invalid-email'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'



# Parsed testcases at query #121
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '123'
    var_5 = '^abc.*'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc123'
    var_8 = 'ab'
    var_9 = module_0.rex(var_0)
    var_10 = 123
    var_11 = None
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_12, var_13, var_14]
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = '^a\\.b$'
    var_20 = module_0.rex(var_19)
    var_21 = 'a.b'
    var_22 = 'aXb'



# Parsed testcases at query #122
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^user@\\w+\\.com$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example.org'
    var_15 = 'user@.com'
    var_16 = '123'
    var_17 = None
    var_18 = 123
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test.user@example.com'
    var_25 = 'invalid@email'



# Parsed testcases at query #123
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'
    var_20 = 'testXtxt'
    var_21 = '^(\\d+)-(\\w+)$'
    var_22 = module_0.rex(var_21)
    var_23 = '123-abc'
    var_24 = 'abc-123'



# Parsed testcases at query #124
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
    var_9 = 'any'
    var_10 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid@email'
    var_14 = 'noatsign.com'



# Parsed testcases at query #125
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.txt.bak'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'any'
    var_18 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email@'



# Parsed testcases at query #126
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
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testTxt'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'user@example.org'
    var_24 = 'user@example'



# Parsed testcases at query #127
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
    var_14 = '^test\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.txt'
    var_17 = 'testxt'



# Parsed testcases at query #128
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
    var_11 = [var_8]
    var_12 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'invalid.email@'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'



# Parsed testcases at query #129
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'testing'
    var_4 = 'notest'
    var_5 = 123
    var_6 = None
    var_7 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_8 = module_0.rex(var_7)
    var_9 = '123-45-6789'
    var_10 = '12-34-5678'
    var_11 = '1234-56-7890'
    var_12 = '^Test'
    var_13 = module_0.rex(var_12)
    var_14 = 'Test'
    var_15 = '^test\\.com$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.com'
    var_18 = 'testcom'
    var_19 = 'testxcom'



# Parsed testcases at query #130
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'ab'
    var_17 = 'aXb'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = 'a'
    var_22 = '^.*test.*$'
    var_23 = module_0.rex(var_22)
    var_24 = 'prefix_test_suffix'
    var_25 = 'no_test_here'



# Parsed testcases at query #131
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
    var_14 = 'ab'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '



# Parsed testcases at query #132
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



# Parsed testcases at query #133
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = 'a'
    var_17 = '^([a-z]+)_(\\d{3})$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abc_123'
    var_20 = 'ABC_123'
    var_21 = 'abc_12'



# Parsed testcases at query #134
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
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'user@example'
    var_16 = 'user@.com'



# Parsed testcases at query #135
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_abc'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email@'
    var_13 = 'another.valid.email@sub.domain.co.uk'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'
    var_18 = 'aXb'



# Parsed testcases at query #136
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z][a-z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'Hello'
    var_8 = 'hello'
    var_9 = '.*\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.csv'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^(?P<name>[a-zA-Z]+)_(?P<num>\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'name_123'
    var_21 = '123_name'



# Parsed testcases at query #137
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
    var_18 = '^(\\w+)_(\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'word_123'
    var_21 = 'word_123_extra'



# Parsed testcases at query #138
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 123
    var_6 = '^test\\d+'
    var_7 = module_0.rex(var_6)
    var_8 = 'testabc'
    var_9 = None
    var_10 = ''
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a'



# Parsed testcases at query #139
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = '^\\w+@\\w+\\.\\w+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'invalid@email'
    var_16 = 'user@example'
    var_17 = '^Test$'
    var_18 = module_0.rex(var_17)
    var_19 = 'Test'
    var_20 = 'test'



# Parsed testcases at query #140
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'abc_test'
    var_5 = 'test'
    var_6 = '^Test'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 123
    var_10 = None
    var_11 = '^exact$'
    var_12 = module_0.rex(var_11)
    var_13 = 'exact'
    var_14 = 'exact_match'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #141
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
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
    var_19 = '^.*$'
    var_20 = module_0.rex(var_19)
    var_21 = 'anything goes here'



# Parsed testcases at query #142
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
    var_10 = '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid-email'



# Parsed testcases at query #143
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
    var_7 = 'abd123'
    var_8 = module_0.rex(var_0)
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = 'a'



# Parsed testcases at query #144
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
    var_14 = 'any'
    var_15 = '^test\\$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test$'
    var_18 = 'test'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]{2,}$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'invalid@email'
    var_12 = 'noatsign.com'
    var_13 = '^[\\w.-]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'valid-chars.123'
    var_16 = 'invalid chars'



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'invalid-email'
    var_12 = 'another.user@domain.co.uk'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^a\\.b$'
    var_18 = module_0.rex(var_17)
    var_19 = 'a.b'
    var_20 = 'ab'
    var_21 = 'aXb'



# Parsed testcases at query #3
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^[a-zA-Z]+\\.[a-zA-Z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'file.txt'
    var_16 = 'file'
    var_17 = '.txt'



# Parsed testcases at query #4
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
    var_22 = '^(?P<name>\\w+)-(?P<id>\\d{3})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'product-123'
    var_25 = 'product-12'
    var_26 = 'product-1234'



# Parsed testcases at query #5
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
    var_12 = []
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'
    var_21 = 'test-txt'
    var_22 = '^a{2,4}$'
    var_23 = module_0.rex(var_22)
    var_24 = 'aa'
    var_25 = 'aaa'
    var_26 = 'aaaa'
    var_27 = 'a'
    var_28 = 'aaaaa'



# Parsed testcases at query #6
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@example.com.'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'user.name+tag@example.com'
    var_16 = 'user@sub.example.com'
    var_17 = 'user@.com'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = '^test\\d+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'test123abc'
    var_8 = 123
    var_9 = None
    var_10 = ''
    var_11 = '^\\w+@\\w+\\.\\w+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example'
    var_15 = 'user@.com'



# Parsed testcases at query #8
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
    var_8 = 123
    var_9 = None
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'ab'
    var_14 = '^ABC$'
    var_15 = module_0.rex(var_14)
    var_16 = 'ABC'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = 'a'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'foo_test'
    var_4 = 'test'
    var_5 = 123
    var_6 = '^[a-z]+_\\d+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'abc_123'
    var_9 = 'ABC_123'
    var_10 = 'abc_'
    var_11 = '_123'
    var_12 = '^\\d+\\.\\d+$'
    var_13 = module_0.rex(var_12)
    var_14 = '3.14'
    var_15 = '3.14.15'
    var_16 = '.14'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = ' '
    var_21 = '^[a-z]+$'
    var_22 = module_0.rex(var_21)
    var_23 = 'ABC'
    var_24 = 'abc'
    var_25 = '123'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '\\.\\*\\+\\?'
    var_15 = module_0.rex(var_14)
    var_16 = '.*+?'



# Parsed testcases at query #11
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
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '
    var_19 = '^test\\.txt$'
    var_20 = module_0.rex(var_19)
    var_21 = 'test.txt'
    var_22 = 'testxt'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email'
    var_13 = 'user@.com'
    var_14 = '^.*\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'file.txt'
    var_17 = 'file.txt.bak'
    var_18 = 'file'



# Parsed testcases at query #13
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
    var_8 = ''
    var_9 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email@'
    var_13 = 'another.valid-email@sub.domain.co.uk'
    var_14 = '^[A-Z]+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'ABC'
    var_17 = 'abc'
    var_18 = 'AbC'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'ab'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'user@example.org'
    var_22 = 'user@example'



# Parsed testcases at query #15
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@example.com.org'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^test\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test.txt'
    var_12 = 'testxt'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'any'
    var_16 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'invalid-email'



# Parsed testcases at query #17
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
    var_8 = ''
    var_9 = module_0.rex(var_8)
    var_10 = 'any'
    var_11 = '^test\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.txt'
    var_14 = 'testxt'
    var_15 = '^[A-Z]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'ABC'
    var_18 = 'abc'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = 'a'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'ab'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([a-z]+)_(\\d{4})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'file_2023'
    var_21 = 'file_23'
    var_22 = 'FILE_2023'



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
    var_10 = 123
    var_11 = None
    var_12 = [var_9]
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 'test-txt'
    var_18 = '^$'
    var_19 = module_0.rex(var_18)
    var_20 = ''
    var_21 = ' '



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_0.rex(var_9)
    var_11 = 'anything'
    var_12 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'user@example'
    var_16 = 'user@.com'
    var_17 = 'user@example.com.'
    var_18 = '^[\\w\\-]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'valid-identifier'
    var_21 = 'invalid identifier'
    var_22 = 'invalid@identifier'



# Parsed testcases at query #21
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
    var_9 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_10 = module_0.rex(var_9)
    var_11 = '123-45-6789'
    var_12 = '12-34-5678'
    var_13 = '1234-56-7890'
    var_14 = '(?i)^abc$'
    var_15 = module_0.rex(var_14)
    var_16 = 'ABC'
    var_17 = 'AbC'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_bar'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = '^[a-z]+@[a-z]+\\.com$'
    var_7 = module_0.rex(var_6)
    var_8 = 'user@example.com'
    var_9 = 'user@example.org'
    var_10 = 'user@example.com.'
    var_11 = '\\d+'
    var_12 = module_0.rex(var_11)
    var_13 = 123
    var_14 = '123'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '[A-Z][a-z]+'
    var_19 = module_0.rex(var_18)
    var_20 = 'Hello'
    var_21 = 'hello'



# Parsed testcases at query #23
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
    var_9 = module_0.rex(var_8)
    var_10 = 'xyzabc'
    var_11 = 'abcxyz'
    var_12 = 'xyz'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = '\\d+'
    var_18 = module_0.rex(var_17)
    var_19 = 123
    var_20 = '123'
    var_21 = ''
    var_22 = module_0.rex(var_21)
    var_23 = 'anything'
    var_24 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_25 = module_0.rex(var_24)
    var_26 = 'user@example.com'
    var_27 = 'invalid-email'



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
    var_8 = module_0.rex(var_7)
    var_9 = 'anything'
    var_10 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid.email'
    var_14 = 'another@valid.co.uk'



# Parsed testcases at query #25
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^([a-z]+)_(\\d{4})$'
    var_17 = module_0.rex(var_16)
    var_18 = 'file_2023'
    var_19 = 'FILE_2023'
    var_20 = 'file_23'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabc'
    var_6 = 'abcx'
    var_7 = 'xabcy'
    var_8 = 'ab'
    var_9 = 123
    var_10 = None
    var_11 = [var_2]
    var_12 = '^a.c$'
    var_13 = module_0.rex(var_12)
    var_14 = 'axc'
    var_15 = 'ac'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'a'



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
    var_6 = ''
    var_7 = 123
    var_8 = None
    var_9 = []
    var_10 = {}
    var_11 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'invalid.email@'
    var_15 = 'noatsign.com'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 123
    var_5 = '^abc.*'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc123'
    var_8 = 'abx123'
    var_9 = '^ABC$'
    var_10 = module_0.rex(var_9)
    var_11 = 'ABC'
    var_12 = '^a\\.b$'
    var_13 = module_0.rex(var_12)
    var_14 = 'a.b'
    var_15 = 'ab'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'a'



# Parsed testcases at query #29
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'aXb'
    var_20 = 'ab'
    var_21 = '^(\\w+)-(\\d+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'test-123'
    var_24 = 'test-abc'
    var_25 = '^a{2,3}$'
    var_26 = module_0.rex(var_25)
    var_27 = 'aa'
    var_28 = 'aaa'
    var_29 = 'a'
    var_30 = 'aaaa'



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'user@example.com'
    var_9 = 'invalid.email'
    var_10 = 'another@test.org'
    var_11 = '.*'
    var_12 = module_0.rex(var_11)
    var_13 = 'any string'
    var_14 = ''
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = 'not empty'



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
    var_8 = '^[a-zA-Z]+_\\d{2,4}$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_1234'
    var_11 = 'ABC_12'
    var_12 = 'abc_1'
    var_13 = '123_abc'
    var_14 = '^[a-z]+\\.$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.'
    var_17 = 'test'
    var_18 = 'test..'



# Parsed testcases at query #32
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
    var_7 = 'abc'
    var_8 = module_0.rex(var_7)
    var_9 = 'xabcy'
    var_10 = 'xyz'
    var_11 = '^test\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.txt'
    var_14 = 'testxt'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^[A-Z]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'ABC'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'testing'
    var_4 = 'notest'
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
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '



# Parsed testcases at query #34
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^\\W+$'
    var_17 = module_0.rex(var_16)
    var_18 = '!@#'
    var_19 = 'abc'



# Parsed testcases at query #35
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
    var_10 = 'a.c'
    var_11 = module_0.rex(var_10)
    var_12 = 'aXc'
    var_13 = 'ac'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = '^[a-z]+@[a-z]+\\.[a-z]{2,3}$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test@example.com'
    var_19 = 'test@example.co.uk'
    var_20 = 'test@example'
    var_21 = 'test@example.c'



# Parsed testcases at query #36
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
    var_11 = '^hello\\.world$'
    var_12 = module_0.rex(var_11)
    var_13 = 'hello.world'
    var_14 = 'helloworld'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = ' '
    var_19 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'user@example.com'
    var_22 = 'invalid.email'



# Parsed testcases at query #37
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
    var_10 = '^test\\.\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.456'
    var_13 = 'test/456'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_18 = module_0.rex(var_17)
    var_19 = 'user@example.com'
    var_20 = 'invalid.email@'



# Parsed testcases at query #38
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
    var_11 = '^\\w+@\\w+\\.\\w+$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'invalid@email'
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'not empty'



# Parsed testcases at query #39
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
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'helloworld'
    var_4 = module_0.rex(var_2)
    var_5 = 'goodbye'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'test@example.com'
    var_11 = 'test@example'
    var_12 = 'test@example.com.org'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test'
    var_6 = module_0.rex(var_5)
    var_7 = 'test123'
    var_8 = '123test'
    var_9 = '[A-Z]+'
    var_10 = module_0.rex(var_9)
    var_11 = 'ABC'
    var_12 = 'abc'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 123
    var_18 = None



# Parsed testcases at query #42
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
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = 'a'
    var_20 = '^([a-z]+)_(\\d{4})-(\\d{2})-(\\d{2})$'
    var_21 = module_0.rex(var_20)
    var_22 = 'event_2023-01-15'
    var_23 = 'event_23-01-15'
    var_24 = 'Event_2023-01-15'



# Parsed testcases at query #43
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #44
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
    var_7 = '\\d+'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc123def'
    var_10 = 'abcdef'
    var_11 = '[A-Z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'ABC'
    var_14 = 'abc'
    var_15 = '^\\W+$'
    var_16 = module_0.rex(var_15)
    var_17 = '!@#'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'



# Parsed testcases at query #45
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #46
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 'test_123_extra'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^test\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test.txt'
    var_12 = 'testxt'
    var_13 = '123'
    var_14 = None
    var_15 = 123
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'
    var_19 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_20 = module_0.rex(var_19)
    var_21 = 'user@example.com'
    var_22 = 'invalid.email'



# Parsed testcases at query #47
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
    var_8 = '^[a-z]+@[a-z]+\\.com$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example.org'
    var_12 = 'user@example'
    var_13 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_14 = module_0.rex(var_13)
    var_15 = '123-45-6789'
    var_16 = '12-34-5678'



# Parsed testcases at query #48
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'not_test'
    var_5 = 123
    var_6 = None
    var_7 = '^[A-Z].*'
    var_8 = module_0.rex(var_7)
    var_9 = 'Abc'
    var_10 = 'abc'
    var_11 = '^\\d{3}-[a-zA-Z]{2}-\\d{4}$'
    var_12 = module_0.rex(var_11)
    var_13 = '123-ab-4567'
    var_14 = '12-ab-4567'
    var_15 = '123-abcd-4567'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



# Parsed testcases at query #49
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = 123
    var_5 = '^hello.*world$'
    var_6 = module_0.rex(var_5)
    var_7 = 'hello beautiful world'
    var_8 = 'world'
    var_9 = '^HELLO$'
    var_10 = module_0.rex(var_9)
    var_11 = 'HELLO'
    var_12 = '^a\\.b$'
    var_13 = module_0.rex(var_12)
    var_14 = 'a.b'
    var_15 = 'ab'
    var_16 = 'aXb'
    var_17 = None
    var_18 = [var_14]



# Parsed testcases at query #50
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
    var_14 = '^\\w+@\\w+\\.\\w+$'
    var_15 = module_0.rex(var_14)
    var_16 = 'user@example.com'
    var_17 = 'invalid@email'



# Parsed testcases at query #51
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^[A-Z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'ABC'
    var_16 = 'abc'
    var_17 = '^[^@]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test'
    var_20 = 'test@'



# Parsed testcases at query #52
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
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'invalid.email@com'
    var_15 = 'another.user@sub.domain.co.uk'
    var_16 = '^CaseSensitive$'
    var_17 = module_0.rex(var_16)
    var_18 = 'CaseSensitive'
    var_19 = 'casesensitive'



# Parsed testcases at query #53
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
    var_9 = 'a.c'
    var_10 = module_0.rex(var_9)
    var_11 = 'axc'
    var_12 = 'ac'
    var_13 = '^[A-Z][a-z]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'Hello'
    var_16 = 'hello'
    var_17 = 'Hello123'



# Parsed testcases at query #54
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'



# Parsed testcases at query #55
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 123
    var_6 = '^Test$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^([a-z]+)_(\\d{3})$'
    var_19 = module_0.rex(var_18)
    var_20 = 'abc_123'
    var_21 = 'ABC_123'
    var_22 = 'abc_12'



# Parsed testcases at query #56
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = '^\\w+@\\w+\\.\\w+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'user@example.com'
    var_16 = 'user@example'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'



# Parsed testcases at query #57
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'foo_test'
    var_4 = 'test'
    var_5 = '^[a-z]+_[0-9]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc_123'
    var_8 = 'ABC_123'
    var_9 = 'abc_123_'
    var_10 = 123
    var_11 = None
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^[a-z]+\\.$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.'
    var_18 = 'test..'



# Parsed testcases at query #58
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
    var_7 = 'abx123'
    var_8 = module_0.rex(var_0)
    var_9 = 123
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'



# Parsed testcases at query #59
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
    var_12 = 'user@example.co.uk'
    var_13 = 'USER@example.com'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'



# Parsed testcases at query #60
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
    var_7 = '\\d+'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc123def'
    var_10 = 'abcdef'
    var_11 = '^Test$'
    var_12 = module_0.rex(var_11)
    var_13 = 'Test'
    var_14 = 'test'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = ''
    var_22 = ' '



# Parsed testcases at query #61
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
    var_10 = ''
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = '^[a-z]+@[a-z]+\\.com$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test@example.com'
    var_16 = 'test@example.org'
    var_17 = 'test@example'



# Parsed testcases at query #62
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
    var_12 = [var_2]
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'any_string'
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'
    var_20 = 'test-txt'
    var_21 = '^(\\w+)_(\\d+)$'
    var_22 = module_0.rex(var_21)



# Parsed testcases at query #63
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
    var_10 = '^[a-zA-Z_][a-zA-Z0-9_]*$'
    var_11 = module_0.rex(var_10)
    var_12 = 'valid_var'
    var_13 = '1invalid'
    var_14 = 'has space'
    var_15 = '^[A-Z]+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'ABC'
    var_18 = 'abc'



# Parsed testcases at query #64
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
    var_20 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_21 = module_0.rex(var_20)
    var_22 = 'user@example.com'
    var_23 = 'invalid.email'
    var_24 = 'another.valid+email@sub.domain.co.uk'



# Parsed testcases at query #65
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



# Parsed testcases at query #66
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^\\w+@\\w+\\.\\w+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'user@example.com'
    var_18 = 'user@example'
    var_19 = 'user@.com'



# Parsed testcases at query #67
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = 'ab'
    var_18 = ''
    var_19 = module_0.rex(var_18)
    var_20 = 'anything'
    var_21 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_22 = module_0.rex(var_21)
    var_23 = 'user@example.com'
    var_24 = 'invalid.email@'
    var_25 = 'another.valid-one@domain.co.uk'



# Parsed testcases at query #68
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'aXb'
    var_20 = '^(\\w+)-(\\w+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'hello-world'
    var_23 = 'hello world'



# Parsed testcases at query #69
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
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = 123
    var_13 = '123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^a\\.b$'
    var_18 = module_0.rex(var_17)
    var_19 = 'a.b'
    var_20 = 'aXb'
    var_21 = '^(\\w+)-(\\w+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'hello-world'
    var_24 = 'hello'



# Parsed testcases at query #70
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^[\\w\\-]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'valid-name'
    var_16 = 'invalid name'
    var_17 = 'invalid@name'



# Parsed testcases at query #71
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
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'a'



# Parsed testcases at query #72
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
    var_14 = '^test\\.$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.'
    var_17 = 'test'
    var_18 = 'test..'
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'abc_123'
    var_22 = 'abc_def'



# Parsed testcases at query #73
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
    var_7 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'invalid.email'
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'any'
    var_14 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_15 = module_0.rex(var_14)
    var_16 = '123-45-6789'
    var_17 = '12-34-5678'



# Parsed testcases at query #74
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_6 = module_0.rex(var_5)
    var_7 = 'user@example.com'
    var_8 = 'invalid.email'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'



# Parsed testcases at query #75
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabcy'
    var_6 = 'ab'
    var_7 = module_0.rex(var_2)
    var_8 = 123
    var_9 = None
    var_10 = 'a.c'
    var_11 = module_0.rex(var_10)
    var_12 = 'aXc'
    var_13 = 'ac'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'a'
    var_17 = '^[A-Z][a-z]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'Hello'
    var_20 = 'hello'
    var_21 = 'H'



# Parsed testcases at query #76
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = 123
    var_5 = module_0.rex(var_2)
    var_6 = 'xabcy'
    var_7 = 'xyz'
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = '123'
    var_11 = '12a34'
    var_12 = None
    var_13 = 'a'
    var_14 = 'b'
    var_15 = [var_13, var_14]



# Parsed testcases at query #77
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
    var_8 = '\\d+'
    var_9 = module_0.rex(var_8)
    var_10 = 123
    var_11 = '123'
    var_12 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'user@example.com'
    var_15 = 'user@example'
    var_16 = 'user@.com'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'
    var_20 = '\\$test'
    var_21 = module_0.rex(var_20)
    var_22 = '$test'



# Parsed testcases at query #78
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
    var_11 = []
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^\\w+@\\w+\\.\\w+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'user@example.com'
    var_19 = 'user@example'
    var_20 = 'user@.com'



# Parsed testcases at query #79
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'abc_123_'
    var_12 = 'ABC_123'
    var_13 = 'abc_def'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = ' '
    var_18 = '^[a-z]+\\.$'
    var_19 = module_0.rex(var_18)
    var_20 = 'abc.'
    var_21 = 'abc'
    var_22 = 'abc.d'



# Parsed testcases at query #80
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
    var_9 = '^\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^a\\.b$'
    var_18 = module_0.rex(var_17)
    var_19 = 'a.b'
    var_20 = 'aXb'



# Parsed testcases at query #81
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = '^a.*c$'
    var_5 = module_0.rex(var_4)
    var_6 = 'a123c'
    var_7 = 'ac'
    var_8 = 123
    var_9 = None
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = '^a\\.c$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.c'



# Parsed testcases at query #82
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 'test_123_extra'
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'any'
    var_14 = '^test\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.txt'
    var_17 = 'testxt'



# Parsed testcases at query #83
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
    var_10 = '^test\\.txt$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.txt'
    var_13 = 'testTxt'
    var_14 = '^$'
    var_15 = module_0.rex(var_14)
    var_16 = ''
    var_17 = 'a'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #84
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = '^[A-Z][a-z]+$'
    var_7 = module_0.rex(var_6)
    var_8 = 'Hello'
    var_9 = 'hello'
    var_10 = 'HELLO'
    var_11 = '^user@\\w+\\.com$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example.org'
    var_15 = 'user@.com'
    var_16 = '123'
    var_17 = None
    var_18 = 123
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'test.user@example.com'
    var_25 = 'invalid@email'



# Parsed testcases at query #85
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
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^[a-z]+\\.$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.'
    var_20 = 'test'
    var_21 = 'test!'



# Parsed testcases at query #86
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
    var_11 = 'goodbye'
    var_12 = '\\.txt$'
    var_13 = module_0.rex(var_12)
    var_14 = 'file.txt'
    var_15 = 'file.txt.bak'
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'



# Parsed testcases at query #87
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
    var_8 = ''
    var_9 = module_0.rex(var_8)
    var_10 = 'any'
    var_11 = '^[a-zA-Z][a-zA-Z0-9_]*$'
    var_12 = module_0.rex(var_11)
    var_13 = 'valid_name'
    var_14 = '1invalid'
    var_15 = 'valid name'



# Parsed testcases at query #88
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
    var_11 = [var_8]
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'anything'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'file_42'
    var_22 = 'file_42_extra'



# Parsed testcases at query #89
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
    var_8 = ''
    var_9 = module_0.rex(var_8)
    var_10 = 'anything'
    var_11 = '^test\\.txt$'
    var_12 = module_0.rex(var_11)
    var_13 = 'test.txt'
    var_14 = 'testxt'
    var_15 = '^Test$'
    var_16 = module_0.rex(var_15)
    var_17 = 'Test'
    var_18 = 'test'



# Parsed testcases at query #90
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
    var_7 = 'abx123'
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



# Parsed testcases at query #91
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
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '



# Parsed testcases at query #92
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
    var_7 = '^[a-z]+_[0-9]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'abc_123'
    var_10 = 'ABC_123'
    var_11 = 'abc_123_'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^[a-z]+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'ABC'
    var_19 = 'ABC123'



# Parsed testcases at query #93
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_'
    var_6 = '^HELLO$'
    var_7 = module_0.rex(var_6)
    var_8 = 'HELLO'
    var_9 = 'hello'
    var_10 = '^a\\.b$'
    var_11 = module_0.rex(var_10)
    var_12 = 'a.b'
    var_13 = 'aXb'
    var_14 = 123
    var_15 = None
    var_16 = ''
    var_17 = module_0.rex(var_16)
    var_18 = 'anything'
    var_19 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_20 = module_0.rex(var_19)
    var_21 = 'user@example.com'
    var_22 = 'invalid-email'



# Parsed testcases at query #94
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = '^[A-Z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^(\\w+)-(\\w+)$'
    var_18 = module_0.rex(var_17)
    var_19 = 'hello-world'
    var_20 = 'hello'



# Parsed testcases at query #95
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
    var_10 = '^test\\.txt$'
    var_11 = module_0.rex(var_10)
    var_12 = 'test.txt'
    var_13 = 'testxt'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'any'
    var_17 = '^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})$'
    var_18 = module_0.rex(var_17)
    var_19 = 'user@example.com'
    var_20 = 'invalid.email'



# Parsed testcases at query #96
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^HELLO$'
    var_6 = module_0.rex(var_5)
    var_7 = 'HELLO'
    var_8 = 'hello'
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'aXb'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email'



# Parsed testcases at query #97
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^\\w+@\\w+\\.\\w+$'
    var_16 = module_0.rex(var_15)
    var_17 = 'user@example.com'
    var_18 = 'user@example'



# Parsed testcases at query #98
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
    var_14 = '^test\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.txt'
    var_17 = 'testxt'
    var_18 = '^(\\d{3})-(\\d{3})-(\\d{4})$'
    var_19 = module_0.rex(var_18)
    var_20 = '123-456-7890'
    var_21 = '12-345-6789'



# Parsed testcases at query #99
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
    var_16 = '^.*\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'file.txt'
    var_19 = 'file.txt.bak'



# Parsed testcases at query #100
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
    var_8 = [var_2]
    var_9 = ''
    var_10 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'invalid.email@'
    var_14 = 'another.valid-one@sub.domain.co.uk'
    var_15 = '^hello$'
    var_16 = module_0.rex(var_15)
    var_17 = 'hello'
    var_18 = 'HELLO'
    var_19 = '^$'
    var_20 = module_0.rex(var_19)
    var_21 = '$'
    var_22 = 'dollar'



# Parsed testcases at query #101
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
    var_8 = 'key'
    var_9 = {var_8: var_2}
    var_10 = ''
    var_11 = module_0.rex(var_10)
    var_12 = 'anything'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'user@example.com'
    var_16 = 'invalid.email@'
    var_17 = 'another.valid-one@domain.co.uk'
    var_18 = '^[A-Z]+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'ABC'
    var_21 = 'abc'



# Parsed testcases at query #102
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
    var_9 = '^\\w+@\\w+\\.\\w+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid@email'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^(?P<name>\\w+)-(?P<value>\\d+)$'
    var_17 = module_0.rex(var_16)
    var_18 = 'count-42'
    var_19 = 'invalid-42'



# Parsed testcases at query #103
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
    var_7 = 'test'
    var_8 = module_0.rex(var_7)
    var_9 = 'prefix_test'
    var_10 = 'no_match'
    var_11 = '[A-Z]+'
    var_12 = module_0.rex(var_11)
    var_13 = 'ABC'
    var_14 = 'abc'
    var_15 = 'test\\.txt'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'



# Parsed testcases at query #104
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
    var_15 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_16 = module_0.rex(var_15)
    var_17 = '123-45-6789'
    var_18 = '12-345-6789'



# Parsed testcases at query #105
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc123def'
    var_12 = 'abcdef'
    var_13 = '^test\\.txt$'
    var_14 = module_0.rex(var_13)
    var_15 = 'test.txt'
    var_16 = 'testxt'
    var_17 = 'test-txt'
    var_18 = '^Test$'
    var_19 = module_0.rex(var_18)
    var_20 = 'Test'
    var_21 = 'test'
    var_22 = '^(\\w+)_(\\d+)$'
    var_23 = module_0.rex(var_22)
    var_24 = 'abc_123'
    var_25 = 'abc_def'
    var_26 = '^a{2,4}$'
    var_27 = module_0.rex(var_26)
    var_28 = 'aa'
    var_29 = 'aaa'
    var_30 = 'aaaa'
    var_31 = 'a'
    var_32 = 'aaaaa'
    var_33 = '^[A-Z][a-z]+$'
    var_34 = module_0.rex(var_33)
    var_35 = 'Abc'
    var_36 = 'abc'
    var_37 = 'ABC'
    var_38 = 'A1bc'



# Parsed testcases at query #106
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
    var_16 = '123'
    var_17 = None
    var_18 = 123
    var_19 = ''
    var_20 = module_0.rex(var_19)
    var_21 = 'anything'
    var_22 = '^([a-zA-Z]+)-(\\d{4})$'
    var_23 = module_0.rex(var_22)
    var_24 = 'Report-2023'
    var_25 = 'report-23'
    var_26 = 'Report-2023-extra'



# Parsed testcases at query #107
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
    var_9 = '\\d+'
    var_10 = module_0.rex(var_9)
    var_11 = 123
    var_12 = '123'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '
    var_17 = '^[a-z]+\\.$'
    var_18 = module_0.rex(var_17)
    var_19 = 'hello.'
    var_20 = 'hello'
    var_21 = 'hello!'



# Parsed testcases at query #108
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
    var_9 = 'any'
    var_10 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]{2,}$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@example'
    var_14 = 'user@.com'
    var_15 = '^\\$\\d+\\.\\d{2}$'
    var_16 = module_0.rex(var_15)
    var_17 = '$123.45'
    var_18 = '$123,45'



# Parsed testcases at query #109
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
    var_13 = 'any'
    var_14 = '^test\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test.txt'
    var_17 = 'testxt'
    var_18 = '^(\\w+)_(\\d+)$'
    var_19 = module_0.rex(var_18)
    var_20 = 'file_42'
    var_21 = 'file_'



# Parsed testcases at query #110
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email'
    var_13 = 'another.user@domain.co.uk'
    var_14 = "^[a-zA-Z0-9_!#$%&\\'*+/=?`{|}~^-]+(\\.[a-zA-Z0-9_!#$%&\\'*+/=?`{|}~^-]+)*@[a-zA-Z0-9-]+(\\.[a-zA-Z0-9-]+)*$"
    var_15 = module_0.rex(var_14)
    var_16 = 'user+tag@example.com'
    var_17 = 'user@sub.domain.com'



# Parsed testcases at query #111
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^test\\.txt$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.txt'
    var_19 = 'testxt'
    var_20 = '^(\\w+)_(\\d+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'file_123'
    var_23 = 'file_123_extra'



# Parsed testcases at query #112
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
    var_7 = '^[a-z]+@[a-z]+\\.[a-z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@example.com.'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = '^test\\.$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.'



# Parsed testcases at query #113
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = 123
    var_5 = 'other'
    var_6 = 1
    var_7 = 2
    var_8 = {var_2: var_6, var_5: var_7}
    var_9 = module_0.rex(var_0)
    var_10 = '^no_match$'
    var_11 = module_0.rex(var_10)
    var_12 = 'no_match'
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'ab'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'anything'



# Parsed testcases at query #114
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
    var_9 = '^a\\.b$'
    var_10 = module_0.rex(var_9)
    var_11 = 'a.b'
    var_12 = 'ab'
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '(?i)^hello$'
    var_17 = module_0.rex(var_16)
    var_18 = 'HELLO'
    var_19 = 'hello'
    var_20 = 'Hello'



# Parsed testcases at query #115
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = 'test'
    var_3 = 'test123'
    var_4 = '123test'
    var_5 = 123
    var_6 = '^[A-Z]'
    var_7 = module_0.rex(var_6)
    var_8 = 'ABC'
    var_9 = 'abc'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = '12a3'
    var_14 = None
    var_15 = [var_2]
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



# Parsed testcases at query #116
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'ABC_123'
    var_12 = 'abc_'
    var_13 = '_123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'abc'
    var_17 = '^[a-z]+\\.$'
    var_18 = module_0.rex(var_17)
    var_19 = 'abc.'
    var_20 = 'abc.d'



# Parsed testcases at query #117
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
    var_7 = 'test'
    var_8 = [var_7]
    var_9 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email@'
    var_13 = 'another.valid+email@example.co.uk'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^[\\w\\-]+$'
    var_18 = module_0.rex(var_17)
    var_19 = 'valid-chars_123'
    var_20 = 'invalid chars'



# Parsed testcases at query #118
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'abc_test'
    var_4 = 123
    var_5 = '^[a-z]+_\\d+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc_123'
    var_8 = 'ABC_123'
    var_9 = 'abc_123_'
    var_10 = '^$'
    var_11 = module_0.rex(var_10)
    var_12 = ''
    var_13 = 'a'
    var_14 = '^[a-z]+\\.$'
    var_15 = module_0.rex(var_14)
    var_16 = 'abc.'
    var_17 = 'abc'



# Parsed testcases at query #119
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
    var_13 = '^a\\.b$'
    var_14 = module_0.rex(var_13)
    var_15 = 'a.b'
    var_16 = 'aXb'
    var_17 = '^$'
    var_18 = module_0.rex(var_17)
    var_19 = ''
    var_20 = ' '



# Parsed testcases at query #120
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
    var_8 = '^[a-z]+_[0-9]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'ABC_123'
    var_12 = 'abc_123_'
    var_13 = '^$'
    var_14 = module_0.rex(var_13)
    var_15 = ''
    var_16 = ' '



# Parsed testcases at query #121
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^test\\.$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.'
    var_19 = 'test..'
    var_20 = '^(\\w+)_(\\d+)$'
    var_21 = module_0.rex(var_20)
    var_22 = 'abc_123'
    var_23 = 'abc_def'



# Parsed testcases at query #122
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
    var_15 = '^$'
    var_16 = module_0.rex(var_15)
    var_17 = ''
    var_18 = 'a'



# Parsed testcases at query #123
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
    var_13 = 'user@example.com.'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'any'
    var_17 = '^(?P<name>\\w+)-(?P<id>\\d{3})$'
    var_18 = module_0.rex(var_17)
    var_19 = 'john-123'
    var_20 = 'john-12'



# Parsed testcases at query #124
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_[0-9]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'ABC_123'
    var_12 = 'abc_123_'
    var_13 = '_abc_123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'abc'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'
    var_21 = 'test-txt'



# Parsed testcases at query #125
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
    var_13 = 'test'
    var_14 = [var_13]
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^\\w+@\\w+\\.\\w+$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'user@.com'
    var_22 = 'user@example'



# Parsed testcases at query #126
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
    var_11 = [var_8]
    var_12 = ''
    var_13 = module_0.rex(var_12)
    var_14 = 'any'
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'
    var_19 = '^(\\w+)_(\\d+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'file_42'
    var_22 = 'file_42_extra'



# Parsed testcases at query #127
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = '^[a-z]+_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'abc_123'
    var_11 = 'abc_123_'
    var_12 = '_123'
    var_13 = 'ABC_123'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'abc'
    var_17 = '^test\\.txt$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test.txt'
    var_20 = 'testxt'
    var_21 = 'test_txt'



# Parsed testcases at query #128
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
    var_12 = '^[A-Z]+$'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = '^[a-z]+\\.$'
    var_17 = module_0.rex(var_16)
    var_18 = 'test.'
    var_19 = 'test'



# Parsed testcases at query #129
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^[\\w\\-]+$'
    var_14 = module_0.rex(var_13)
    var_15 = 'valid-key'
    var_16 = 'invalid key'
    var_17 = 'invalid@key'



# Parsed testcases at query #130
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
    var_9 = '^[a-z]+@[a-z]+\\.com$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'user@example.org'
    var_13 = 'USER@EXAMPLE.COM'
    var_14 = ''
    var_15 = module_0.rex(var_14)
    var_16 = 'anything'
    var_17 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_18 = module_0.rex(var_17)
    var_19 = '123-45-6789'
    var_20 = '12-345-6789'



# Parsed testcases at query #131
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
    var_7 = 'abx123'
    var_8 = 123
    var_9 = '^$'
    var_10 = module_0.rex(var_9)
    var_11 = ''
    var_12 = 'a'



# Parsed testcases at query #132
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
    var_7 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'user@example'
    var_11 = 'user@.com'
    var_12 = '^$'
    var_13 = module_0.rex(var_12)
    var_14 = ''
    var_15 = ' '
    var_16 = 'test'
    var_17 = module_0.rex(var_16)
    var_18 = 'this is a test'
    var_19 = 'testing'
    var_20 = 'contest'
    var_21 = '[A-Z]+'
    var_22 = module_0.rex(var_21)
    var_23 = 'ABC'
    var_24 = 'abc'



# Parsed testcases at query #133
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_foo'
    var_3 = 'test_123'
    var_4 = 'foo_test'
    var_5 = 'test'
    var_6 = 123
    var_7 = None
    var_8 = '^user_\\d+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user_42'
    var_11 = 'user_abc'
    var_12 = 'user_42_extra'
    var_13 = '^Test$'
    var_14 = module_0.rex(var_13)
    var_15 = 'Test'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'aXb'



# Parsed testcases at query #134
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
    var_8 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_9 = module_0.rex(var_8)
    var_10 = 'user@example.com'
    var_11 = 'user@example'
    var_12 = 'user@.com'
    var_13 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_14 = module_0.rex(var_13)
    var_15 = 'user.name+tag@example.com'
    var_16 = 'user@sub.example.com'



# Parsed testcases at query #135
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
    var_9 = '^.*\\.txt$'
    var_10 = module_0.rex(var_9)
    var_11 = 'file.txt'
    var_12 = 'file.txt.bak'
    var_13 = 123
    var_14 = None
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = 'anything'
    var_18 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_19 = module_0.rex(var_18)
    var_20 = 'user@example.com'
    var_21 = 'invalid.email@'



# Parsed testcases at query #136
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'hello world'
    var_4 = module_0.rex(var_2)
    var_5 = 'world hello'
    var_6 = 'goodbye'
    var_7 = 123
    var_8 = None
    var_9 = '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
    var_10 = module_0.rex(var_9)
    var_11 = 'test@example.com'
    var_12 = 'invalid-email'
    var_13 = 'another.test@domain.co.uk'



# Parsed testcases at query #137
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
    var_9 = 'any_string'
    var_10 = '^[a-zA-Z_]\\w*$'
    var_11 = module_0.rex(var_10)
    var_12 = 'valid_var'
    var_13 = '1invalid'
    var_14 = 'valid-var'
    var_15 = '^Case$'
    var_16 = module_0.rex(var_15)
    var_17 = 'Case'
    var_18 = 'case'



# Parsed testcases at query #138
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
    var_8 = ''
    var_9 = module_0.rex(var_8)
    var_10 = 'anything'
    var_11 = '^([a-zA-Z]+)@([a-zA-Z]+)\\.com$'
    var_12 = module_0.rex(var_11)
    var_13 = 'user@example.com'
    var_14 = 'user@example.org'
    var_15 = 'user@example'
    var_16 = '^\\w+$'
    var_17 = module_0.rex(var_16)
    var_18 = 'hello_world'
    var_19 = 'hello world'
    var_20 = 'hello-world'



# Parsed testcases at query #139
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
    var_11 = '123-45-678'
    var_12 = '^[A-Z]'
    var_13 = module_0.rex(var_12)
    var_14 = 'ABC'
    var_15 = 'abc'
    var_16 = '^$'
    var_17 = module_0.rex(var_16)
    var_18 = ''
    var_19 = ' '



# Parsed testcases at query #140
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '[A-Z]+'
    var_6 = module_0.rex(var_5)
    var_7 = 'ABC'
    var_8 = 'abc'
    var_9 = 123
    var_10 = None
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '\\.txt$'
    var_15 = module_0.rex(var_14)
    var_16 = 'file.txt'
    var_17 = 'file.txt.bak'



# Parsed testcases at query #141
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
    var_7 = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    var_8 = module_0.rex(var_7)
    var_9 = 'user@example.com'
    var_10 = 'invalid.email'
    var_11 = ''
    var_12 = module_0.rex(var_11)
    var_13 = 'anything'
    var_14 = '^a\\.b$'
    var_15 = module_0.rex(var_14)
    var_16 = 'a.b'
    var_17 = 'ab'



# Parsed testcases at query #142
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
    var_7 = module_0.rex(var_2)
    var_8 = 123
    var_9 = None
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = '^[a-z]+@[a-z]+\\.com$'
    var_15 = module_0.rex(var_14)
    var_16 = 'test@example.com'
    var_17 = 'test@example.org'
    var_18 = 'test@example'
    var_19 = 'test@.com'



# Parsed testcases at query #143
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
    var_13 = ''
    var_14 = module_0.rex(var_13)
    var_15 = 'anything'
    var_16 = '^a\\.b$'
    var_17 = module_0.rex(var_16)
    var_18 = 'a.b'
    var_19 = 'aXb'
    var_20 = 'ab'
    var_21 = '^(\\w+)-(\\d+)$'
    var_22 = module_0.rex(var_21)
    var_23 = 'test-123'
    var_24 = 'test-abc'
    var_25 = '^a{2,3}$'
    var_26 = module_0.rex(var_25)
    var_27 = 'aa'
    var_28 = 'aaa'
    var_29 = 'a'
    var_30 = 'aaaa'



# Parsed testcases at query #144
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'abc_test'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '^[a-z]+_\\d+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'abc_123'
    var_12 = 'ABC_123'
    var_13 = 'abc_'
    var_14 = '_123'
    var_15 = '^[a-z]+\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'file.txt'
    var_18 = 'file.txt.bak'



# Parsed testcases at query #145
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = 'test_123_extra'
    var_6 = 123
    var_7 = None
    var_8 = [var_2]
    var_9 = ''
    var_10 = '^[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]+$'
    var_11 = module_0.rex(var_10)
    var_12 = 'user@example.com'
    var_13 = 'user@example'
    var_14 = 'user@.com'
    var_15 = 'user@example.com.'
    var_16 = '^.*\\$test.*$'
    var_17 = module_0.rex(var_16)
    var_18 = '$test'
    var_19 = 'prefix$testsuffix'
    var_20 = 'test'



# Parsed testcases at query #146
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^abc$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'abcd'
    var_4 = module_0.rex(var_2)
    var_5 = 'xabc'
    var_6 = 'abcx'
    var_7 = 'xabcy'
    var_8 = 123
    var_9 = None
    var_10 = [var_2]
    var_11 = '^a.c$'
    var_12 = module_0.rex(var_11)
    var_13 = 'axc'
    var_14 = 'ac'
    var_15 = ''
    var_16 = module_0.rex(var_15)
    var_17 = '^[a-z]+@[a-z]+\\.com$'
    var_18 = module_0.rex(var_17)
    var_19 = 'test@example.com'
    var_20 = 'test@example.org'
    var_21 = 'test@example'



# Parsed testcases at query #147
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
    var_7 = 'ab123'
    var_8 = module_0.rex(var_0)
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = 'a'



# Parsed testcases at query #148
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_123'
    var_3 = 'test_abc'
    var_4 = '123_test'
    var_5 = '^Hello$'
    var_6 = module_0.rex(var_5)
    var_7 = 'Hello'
    var_8 = 'hello'
    var_9 = 123
    var_10 = None
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^a\\.b$'
    var_16 = module_0.rex(var_15)
    var_17 = 'a.b'
    var_18 = 'aXb'
    var_19 = '^(\\w+)-(\\w+)$'
    var_20 = module_0.rex(var_19)
    var_21 = 'foo-bar'
    var_22 = 'foo_bar'



# Parsed testcases at query #149
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'test_abc'
    var_3 = 'test_123'
    var_4 = 'test'
    var_5 = 'not_test'
    var_6 = 123
    var_7 = None
    var_8 = ''
    var_9 = '^[a-zA-Z0-9_]+@[a-zA-Z0-9_]+\\.[a-zA-Z0-9_]+$'
    var_10 = module_0.rex(var_9)
    var_11 = 'user@example.com'
    var_12 = 'invalid.email'
    var_13 = 'another@test.co.uk'



# Parsed testcases at query #150
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
    var_11 = '^$'
    var_12 = module_0.rex(var_11)
    var_13 = ''
    var_14 = ' '
    var_15 = '^test\\.txt$'
    var_16 = module_0.rex(var_15)
    var_17 = 'test.txt'
    var_18 = 'testxt'



# Parsed testcases at query #151
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
    var_8 = '^\\d{3}-\\d{2}-\\d{4}$'
    var_9 = module_0.rex(var_8)
    var_10 = '123-45-6789'
    var_11 = '12-34-5678'
    var_12 = '1234-56-7890'
    var_13 = '^[A-Z]'
    var_14 = module_0.rex(var_13)
    var_15 = 'Hello'
    var_16 = 'hello'
    var_17 = ''
    var_18 = module_0.rex(var_17)
    var_19 = 'any'



