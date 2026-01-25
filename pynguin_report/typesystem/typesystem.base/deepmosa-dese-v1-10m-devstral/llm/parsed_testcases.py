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



