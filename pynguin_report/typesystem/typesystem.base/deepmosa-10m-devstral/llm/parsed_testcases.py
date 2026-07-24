####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_message_eq_with_different_start_position. Retrieved 5/11 statements.
# Partially parsed test_message_eq_with_different_end_position. Retrieved 5/11 statements.
# Partially parsed test_message_eq_with_position_vs_start_end. Retrieved 3/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'test2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 0
    var_3 = 5
    var_4 = 1

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 0
    var_3 = 5
    var_4 = 6

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 0

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(not var_3 == 'not a message')
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(not var_3 == None)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = var_1.text
    assert var_2 == 'Error message'
    var_3 = var_1.code
    assert var_3 == 'custom'
    var_4 = var_1.index
    var_5 = bool(var_1.index == [])
    assert var_5 is True
    var_6 = var_1.start_position
    assert var_6 is None
    var_7 = var_1.end_position
    assert var_7 is None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'max_length'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = var_2.text
    assert var_3 == 'Error message'
    var_4 = var_2.code
    assert var_4 == 'max_length'
    var_5 = var_2.index
    var_6 = bool(var_2.index == [])
    assert var_6 is True
    var_7 = var_2.start_position
    assert var_7 is None
    var_8 = var_2.end_position
    assert var_8 is None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = var_2.text
    assert var_3 == 'Error message'
    var_4 = var_2.code
    assert var_4 == 'custom'
    var_5 = var_2.index
    var_6 = bool(var_2.index == ['username'])
    assert var_6 is True
    var_7 = var_2.start_position
    assert var_7 is None
    var_8 = var_2.end_position
    assert var_8 is None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'users'
    var_2 = 3
    var_3 = 'username'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Message(text=var_0, index=var_4)
    var_6 = var_5.text
    assert var_6 == 'Error message'
    var_7 = var_5.code
    assert var_7 == 'custom'
    var_8 = var_5.index
    var_9 = bool(var_5.index == ['users', 3, 'username'])
    assert var_9 is True
    var_10 = var_5.start_position
    assert var_10 is None
    var_11 = var_5.end_position
    assert var_11 is None

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error message'
    var_5 = module_0.Message(text=var_4, position=var_3)
    var_6 = var_5.text
    assert var_6 == 'Error message'
    var_7 = var_5.code
    assert var_7 == 'custom'
    var_8 = var_5.index
    var_9 = bool(var_5.index == [])
    assert var_9 is True
    var_10 = var_5.start_position
    var_11 = bool(var_5.start_position == var_3)
    assert var_11 is True
    var_12 = var_5.end_position
    var_13 = bool(var_5.end_position == var_3)
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = module_0.Position(var_4, var_5, var_6)
    var_8 = 'Error message'
    var_9 = module_0.Message(text=var_8, start_position=var_3, end_position=var_7)
    var_10 = var_9.text
    assert var_10 == 'Error message'
    var_11 = var_9.code
    assert var_11 == 'custom'
    var_12 = var_9.index
    var_13 = bool(var_9.index == [])
    assert var_13 is True
    var_14 = var_9.start_position
    var_15 = bool(var_9.start_position == var_3)
    assert var_15 is True
    var_16 = var_9.end_position
    var_17 = bool(var_9.end_position == var_7)
    assert var_17 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = bool(var_3 == var_4)
    assert var_9 is True
    var_10 = bool(not var_3 == var_8)
    assert var_10 is True
    var_11 = bool(not var_3 == 'not a Position')
    assert var_11 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = var_3._messages
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_3._messages[0].text
    assert var_6 == 'Error message'
    var_7 = var_3._messages[0].code
    assert var_7 == 'error_code'
    var_8 = var_3._messages[0].index
    var_9 = bool(var_3._messages[0].index == ['error_key'])
    assert var_9 is True
    var_10 = dict(var_3)
    var_11 = bool(var_10 == {'error_key': 'Error message'})
    assert var_11 is True

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
    var_8 = var_7._messages
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7._messages[0].text
    assert var_10 == 'Error message'
    var_11 = var_7._messages[0].code
    assert var_11 == 'error_code'
    var_12 = var_7._messages[0].index
    var_13 = bool(var_7._messages[0].index == ['error_key'])
    assert var_13 is True
    var_14 = var_7._messages[0].start_position
    var_15 = bool(var_7._messages[0].start_position == var_3)
    assert var_15 is True
    var_16 = var_7._messages[0].end_position
    var_17 = bool(var_7._messages[0].end_position == var_3)
    assert var_17 is True
    var_18 = dict(var_7)
    var_19 = bool(var_18 == {'error_key': 'Error message'})
    assert var_19 is True

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
    var_10 = var_9._messages
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = var_9._messages[0].text
    assert var_12 == 'Error 1'
    var_13 = var_9._messages[0].code
    assert var_13 == 'code1'
    var_14 = var_9._messages[0].index
    var_15 = bool(var_9._messages[0].index == ['key1'])
    assert var_15 is True
    var_16 = var_9._messages[1].text
    assert var_16 == 'Error 2'
    var_17 = var_9._messages[1].code
    assert var_17 == 'code2'
    var_18 = var_9._messages[1].index
    var_19 = bool(var_9._messages[1].index == ['key2'])
    assert var_19 is True
    var_20 = dict(var_9)
    var_21 = bool(var_20 == {'key1': 'Error 1', 'key2': 'Error 2'})
    assert var_21 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = 1
    var_10 = 'email'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = var_14._messages
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14._messages[0].text
    assert var_17 == 'Error 1'
    var_18 = var_14._messages[0].code
    assert var_18 == 'code1'
    var_19 = var_14._messages[0].index
    var_20 = bool(var_14._messages[0].index == ['users', 0, 'username'])
    assert var_20 is True
    var_21 = var_14._messages[1].text
    assert var_21 == 'Error 2'
    var_22 = var_14._messages[1].code
    assert var_22 == 'code2'
    var_23 = var_14._messages[1].index
    var_24 = bool(var_14._messages[1].index == ['users', 1, 'email'])
    assert var_24 is True
    var_25 = dict(var_14)
    var_26 = bool(var_25 == {'users': {0: {'username': 'Error 1'}, 1: {'email': 'Error 2'}}})
    assert var_26 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_iter_with_error. Retrieved 1/4 statements.
# Partially parsed test_iter_with_none. Retrieved 1/2 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'

import typesystem.base as module_0

def test_case_0():
    var_0 = module_0.ValidationResult()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ValidationResult___iter___with_value. Retrieved 2/3 statements.
# Partially parsed test_ValidationResult___iter___with_error. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid data'
    var_1 = module_0.ValidationResult(error=var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 1
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = str(var_11)
    assert var_12 == "{0: 'Error 1', 1: 'Error 2'}"



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = 'Error2'
    var_6 = 'code2'
    var_7 = module_0.Message(text=var_5, code=var_6)
    var_8 = [var_7]
    var_9 = module_0.BaseError(messages=var_8)
    var_10 = bool(not var_4 == var_9)
    assert var_10 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = bool(not var_4 == 'not an error')
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.BaseError(messages=var_0)
    var_2 = []
    var_3 = module_0.BaseError(messages=var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "BaseError([Message(text='Error message', code='error_code', index=['field'])]))"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = repr(var_11)
    var_13 = bool(var_12 == f'BaseError({var_10!r})')
    assert var_13 is True



# Parsed testcases at query #4
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
    var_10 = bool(var_5 == var_9)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.
# Partially parsed test_validation_result_iter_with_none. Retrieved 1/2 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'

import typesystem.base as module_0

def test_case_0():
    var_0 = module_0.ValidationResult()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_message_inequality_with_different_start_position. Retrieved 4/8 statements.
# Partially parsed test_message_inequality_with_different_end_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_position_vs_start_end. Retrieved 2/5 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'test2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'test'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'custom'
    var_3 = module_0.Message(text=var_0, code=var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(not var_1 == 'not a message')
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_iter_with_error. Retrieved 1/4 statements.
# Partially parsed test_iter_with_none. Retrieved 1/2 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'

import typesystem.base as module_0

def test_case_0():
    var_0 = module_0.ValidationResult()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__iter__returns_iterator_of_value_and_error. Retrieved 6/11 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'test_error'
    var_2 = module_0.ValidationResult(value=var_0)
    var_3 = iter(var_2)
    var_4 = next(var_3)
    var_5 = bool(var_4 == var_0)
    assert var_5 is True
    var_6 = next(var_3)
    assert var_6 is None



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = module_0.BaseError(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = bool(var_2 != 'not an error')
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = module_0.BaseError(messages=var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error 1'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = 'Error 2'
    var_6 = module_0.Message(text=var_5, code=var_1)
    var_7 = [var_4, var_6]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = bool(var_2 != var_8)
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'Error message 2'
    var_4 = module_0.BaseError(text=var_3, code=var_1)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = 'not an error'
    var_4 = bool(not var_2 == var_3)
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = module_0.BaseError(messages=var_4)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

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
    var_9 = bool(not var_2 == var_8)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_message_equality_with_same_attributes. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_different_text. Retrieved 5/9 statements.
# Partially parsed test_message_equality_with_different_code. Retrieved 5/9 statements.
# Partially parsed test_message_equality_with_different_index. Retrieved 5/9 statements.
# Partially parsed test_message_equality_with_different_start_position. Retrieved 6/12 statements.
# Partially parsed test_message_equality_with_different_end_position. Retrieved 6/12 statements.
# Partially parsed test_message_equality_with_non_message_object. Retrieved 4/6 statements.
# Partially parsed test_message_equality_with_default_code. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = 'Error2'

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = 1
    var_4 = 'test2'

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field1'
    var_3 = 1
    var_4 = 'field2'

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = 5
    var_5 = 2

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1
    var_4 = 5
    var_5 = 10

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = 1

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = 1
    var_3 = 'custom'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_message_equality_with_different_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_different_start_end_positions. Retrieved 6/12 statements.
# Partially parsed test_message_equality_with_same_position_and_start_end. Retrieved 3/6 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

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
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'custom'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 2
    var_3 = 6
    var_4 = 'Error'
    var_5 = 'custom'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = bool(not var_2 == None)
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = bool(not var_2 == 'not a message')
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'Error'
    var_2 = 'custom'



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = [var_1]
    var_3 = module_0.BaseError(messages=var_2)
    var_4 = module_0.Message(text=var_0)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_3 == var_6)
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



