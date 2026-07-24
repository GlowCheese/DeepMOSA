####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = module_0.Position(var_3, var_4, var_5)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_8)
    var_10 = [var_9]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'parent'
    var_7 = 'key2'
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8)
    var_10 = [var_3, var_9]
    var_11 = module_0.BaseError(messages=var_10)



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Different message'
    var_4 = module_0.BaseError(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'test2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'test'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 2
    var_5 = module_0.Position()
    var_6 = 6
    var_7 = module_0.Position()
    var_8 = 'Error'
    var_9 = 'test'
    var_10 = module_0.Message(text=var_8, code=var_9, start_position=var_1, end_position=var_3)
    var_11 = module_0.Message(text=var_8, code=var_9, start_position=var_5, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = []
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #5
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code_1'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = 'error_code_2'
    var_5 = module_0.BaseError(text=var_3, code=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Another error'
    var_5 = 'another_code'
    var_6 = module_0.Message(text=var_4, code=var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.BaseError(messages=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Another error'
    var_4 = 'another_code'
    var_5 = module_0.Message(text=var_3, code=var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = module_0.Message(text=var_3, code=var_4)
    var_9 = module_0.Message(text=var_0, code=var_1)
    var_10 = [var_8, var_9]
    var_11 = module_0.BaseError(messages=var_10)



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code1'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = 'error_code2'
    var_5 = module_0.BaseError(text=var_3, code=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.BaseError(messages=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = [var_1]
    var_3 = module_0.BaseError(messages=var_2)
    var_4 = module_0.Message(text=var_0)
    var_5 = 'Error 2'
    var_6 = module_0.Message(text=var_5)
    var_7 = [var_4, var_6]
    var_8 = module_0.BaseError(messages=var_7)



# Parsed testcases at query #9
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.Message(text=var_0)
    var_7 = module_0.Message(text=var_2)
    var_8 = [var_6, var_7]
    var_9 = module_0.BaseError(messages=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = module_0.Message(text=var_0)
    var_2 = [var_1]
    var_3 = module_0.BaseError(messages=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = module_0.Message(text=var_0)
    var_2 = [var_1]
    var_3 = module_0.BaseError(messages=var_2)



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.Message(text=var_0)
    var_7 = module_0.Message(text=var_2)
    var_8 = [var_6, var_7]
    var_9 = module_0.BaseError(messages=var_8)



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)



# Parsed testcases at query #14
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = module_0.BaseError(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error_code'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 1
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = module_0.BaseError(messages=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error_code'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = 'Error 2'
    var_7 = 1
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_6, code=var_1, index=var_8)
    var_10 = [var_9]
    var_11 = module_0.BaseError(messages=var_5)
    var_12 = module_0.BaseError(messages=var_10)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_4]
    var_6 = dict(var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'key2'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_8 = [var_3, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    var_10 = dict(var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error with position'
    var_5 = module_0.BaseError(text=var_4, position=var_3)
    var_6 = module_0.Message(text=var_4, position=var_3)
    var_7 = [var_6]
    var_8 = dict(var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.BaseError(messages=var_0)



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = module_0.BaseError(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.BaseError(messages=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error 1'
    var_4 = module_0.Message(text=var_3)
    var_5 = 'Error 2'
    var_6 = module_0.Message(text=var_5)
    var_7 = [var_4, var_6]
    var_8 = module_0.BaseError(messages=var_7)



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 10
    var_3 = module_0.Position()
    var_4 = 2
    var_5 = module_0.Position()
    var_6 = 20
    var_7 = module_0.Position()
    var_8 = 'Error'
    var_9 = 'custom'
    var_10 = module_0.Message(text=var_8, code=var_9, start_position=var_1, end_position=var_3)
    var_11 = module_0.Message(text=var_8, code=var_9, start_position=var_5, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Invalid value', code='invalid')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 2
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = repr(var_11)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = repr(var_6)



# Parsed testcases at query #5
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'test2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 3
    var_7 = [var_2, var_6]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'test'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 3
    var_5 = module_0.Position()
    var_6 = 4
    var_7 = module_0.Position()
    var_8 = 'Error'
    var_9 = 'test'
    var_10 = module_0.Message(text=var_8, code=var_9, start_position=var_1, end_position=var_3)
    var_11 = module_0.Message(text=var_8, code=var_9, start_position=var_5, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Error message'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'First error'
    var_1 = 'error1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Second error'
    var_6 = 'error2'
    var_7 = 1
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = str(var_11)
    assert var_12 == "{0: 'First error', 1: 'Second error'}"



# Parsed testcases at query #9
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'users'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 2
    var_7 = [var_2, var_6]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = []
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_3)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 5
    var_5 = module_0.Position()
    var_6 = 'Error'
    var_7 = 'custom'
    var_8 = module_0.Message(text=var_6, code=var_7, start_position=var_1, end_position=var_5)
    var_9 = module_0.Message(text=var_6, code=var_7, start_position=var_3, end_position=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 10
    var_5 = module_0.Position()
    var_6 = 'Error'
    var_7 = 'custom'
    var_8 = module_0.Message(text=var_6, code=var_7, start_position=var_1, end_position=var_3)
    var_9 = module_0.Message(text=var_6, code=var_7, start_position=var_1, end_position=var_5)



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'code2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position()
    var_3 = 2
    var_4 = module_0.Position()
    var_5 = 'Error'
    var_6 = 'custom'
    var_7 = module_0.Message(text=var_5, code=var_6, position=var_2)
    var_8 = module_0.Message(text=var_5, code=var_6, position=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 4
    var_8 = [var_2, var_7, var_4]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 10
    var_3 = module_0.Position()
    var_4 = 2
    var_5 = module_0.Position()
    var_6 = 20
    var_7 = module_0.Position()
    var_8 = 'Error'
    var_9 = 'max_length'
    var_10 = module_0.Message(text=var_8, code=var_9, start_position=var_1, end_position=var_3)
    var_11 = module_0.Message(text=var_8, code=var_9, start_position=var_5, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Different message'
    var_4 = module_0.BaseError(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = 'code2'
    var_5 = module_0.Message(text=var_3, code=var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = module_0.BaseError(messages=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = 'code2'
    var_5 = module_0.Message(text=var_3, code=var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Message(text=var_0, code=var_1)
    var_8 = 'Error3'
    var_9 = 'code3'
    var_10 = module_0.Message(text=var_8, code=var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.BaseError(messages=var_6)
    var_13 = module_0.BaseError(messages=var_11)



# Parsed testcases at query #14
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)



# Parsed testcases at query #15
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 4
    var_8 = [var_2, var_7, var_4]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_3, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #16
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_1)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, end_position=var_1)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 'Error'
    var_3 = 'max_length'
    var_4 = 'username'
    var_5 = module_0.Message(text=var_2, code=var_3, key=var_4, position=var_1)
    var_6 = module_0.Message(text=var_2, code=var_3, key=var_4, position=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_1, end_position=var_3)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_1, end_position=var_3)



# Parsed testcases at query #17
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #19
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)



# Parsed testcases at query #21
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_1)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, start_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, end_position=var_1)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 'Error'
    var_3 = 'max_length'
    var_4 = 'username'
    var_5 = module_0.Message(text=var_2, code=var_3, key=var_4, position=var_1)
    var_6 = module_0.Message(text=var_2, code=var_3, key=var_4, position=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 'Error'
    var_3 = 'max_length'
    var_4 = 'username'
    var_5 = module_0.Message(text=var_2, code=var_3, key=var_4, position=var_1)
    var_6 = module_0.Message(text=var_2, code=var_3, key=var_4, start_position=var_1, end_position=var_1)



# Parsed testcases at query #23
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.Message(text=var_0)
    var_7 = module_0.Message(text=var_2)
    var_8 = [var_6, var_7]
    var_9 = module_0.BaseError(messages=var_8)



# Parsed testcases at query #24
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #25
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #26
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 4
    var_8 = [var_2, var_7, var_4]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_3, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 'Error'
    var_3 = 'max_length'
    var_4 = module_0.Message(text=var_2, code=var_3, position=var_1)
    var_5 = module_0.Message(text=var_2, code=var_3, start_position=var_1, end_position=var_1)



# Parsed testcases at query #27
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #28
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)



# Parsed testcases at query #29
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)



# Parsed testcases at query #30
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'users'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 2
    var_7 = [var_2, var_6]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, start_position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, start_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = 10
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, end_position=var_4)
    var_6 = 20
    var_7 = module_0.Position()
    var_8 = module_0.Message(text=var_0, code=var_1, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1)
    var_3 = module_0.Message(text=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, start_position=var_1, end_position=var_1)
    var_3 = module_0.Message(text=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #31
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = 'Error2'
    var_7 = module_0.Position()
    var_8 = module_0.Message(text=var_6, code=var_1, key=var_2, position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = 'test2'
    var_7 = module_0.Position()
    var_8 = module_0.Message(text=var_0, code=var_6, key=var_2, position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = 'field2'
    var_7 = module_0.Position()
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_6, position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_4, end_position=var_6)
    var_8 = module_0.Position()
    var_9 = module_0.Position()
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_8, end_position=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_4, end_position=var_6)
    var_8 = module_0.Position()
    var_9 = 3
    var_10 = module_0.Position()
    var_11 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_8, end_position=var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, key=var_1, position=var_3)
    var_5 = 'custom'
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_5, key=var_1, position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'code2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_4]

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error message'
    var_5 = 'error_code'
    var_6 = 'error_key'
    var_7 = module_0.BaseError(text=var_4, code=var_5, key=var_6, position=var_3)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, position=var_3)
    var_9 = [var_8]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code_1'
    var_2 = 'error_key_1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error message 2'
    var_5 = 'error_code_2'
    var_6 = 'error_key_2'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_8 = [var_3, var_7]
    var_9 = module_0.BaseError(messages=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code_1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error message 2'
    var_8 = 'error_code_2'
    var_9 = 1
    var_10 = 'email'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_1, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #5
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'code2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 2
    var_5 = module_0.Position()
    var_6 = module_0.Position()
    var_7 = 'Error'
    var_8 = 'custom'
    var_9 = module_0.Message(text=var_7, code=var_8, start_position=var_1, end_position=var_3)
    var_10 = module_0.Message(text=var_7, code=var_8, start_position=var_5, end_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'test2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 3
    var_7 = [var_2, var_6]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'test'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 5
    var_3 = module_0.Position()
    var_4 = 2
    var_5 = module_0.Position()
    var_6 = 6
    var_7 = module_0.Position()
    var_8 = 'Error'
    var_9 = 'test'
    var_10 = module_0.Message(text=var_8, code=var_9, start_position=var_1, end_position=var_3)
    var_11 = module_0.Message(text=var_8, code=var_9, start_position=var_5, end_position=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = []
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #9
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_1, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a Position'



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = 5
    var_5 = module_0.Position()
    var_6 = module_0.Message(text=var_0, code=var_1, start_position=var_3, end_position=var_5)
    var_7 = 2
    var_8 = module_0.Position()
    var_9 = module_0.Position()
    var_10 = module_0.Message(text=var_0, code=var_1, start_position=var_8, end_position=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = 5
    var_5 = module_0.Position()
    var_6 = module_0.Message(text=var_0, code=var_1, start_position=var_3, end_position=var_5)
    var_7 = module_0.Position()
    var_8 = 10
    var_9 = module_0.Position()
    var_10 = module_0.Message(text=var_0, code=var_1, start_position=var_7, end_position=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, position=var_3)
    var_5 = module_0.Position()
    var_6 = module_0.Message(text=var_0, code=var_1, position=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, position=var_6)



# Parsed testcases at query #14
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #15
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a Position object'



# Parsed testcases at query #16
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #17
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'test2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, start_position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, start_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, end_position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, end_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'custom'
    var_3 = module_0.Message(text=var_0, code=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'specific'
    var_3 = module_0.Message(text=var_0, code=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #18
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_3, end_position=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position()
    var_2 = 2
    var_3 = module_0.Position()
    var_4 = 'Error'
    var_5 = 'max_length'
    var_6 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_1)
    var_7 = module_0.Message(text=var_4, code=var_5, start_position=var_1, end_position=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)



# Parsed testcases at query #19
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #21
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_4, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 5
    var_5 = module_0.Position(var_0, var_4, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 6
    var_5 = module_0.Position(var_0, var_1, var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #23
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #24
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = module_0.Position()
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'test2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'a'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, start_position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, start_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = module_0.Position()
    var_4 = module_0.Message(text=var_0, code=var_1, end_position=var_3)
    var_5 = 2
    var_6 = module_0.Position()
    var_7 = module_0.Message(text=var_0, code=var_1, end_position=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #25
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #26
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_1, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #27
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #28
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)



# Parsed testcases at query #29
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



