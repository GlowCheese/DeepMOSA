####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_message_equality_with_different_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_start_end_positions. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_different_start_position. Retrieved 5/10 statements.
# Partially parsed test_message_equality_with_different_end_position. Retrieved 5/10 statements.


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
    var_2 = 'a'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'test'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_1)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'Error'
    var_3 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = 'Error'
    var_4 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 'Error'
    var_4 = 'test'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = None
    var_3 = module_0.Message(text=var_0, code=var_1, start_position=var_2, end_position=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, position=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(not var_3 == 'not a message')
    assert var_4 is True



# Parsed testcases at query #2
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
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "BaseError([Message(text='Invalid value', code='invalid', index=['field'])]"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = repr(var_11)
    assert var_12 == "BaseError([Message(text='Error 1', code='code1', index=['key1']), Message(text='Error 2', code='code2', index=['key2'])]"



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = '__next__'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #5
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
    var_1 = 'error1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'error2'
    var_7 = 1
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = str(var_11)
    assert var_12 == "{0: 'Error 1', 1: 'Error 2'}"



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'Invalid data'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == ['test', None])
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_eq_with_different_position. Retrieved 4/8 statements.
# Partially parsed test_eq_with_different_start_position. Retrieved 4/8 statements.
# Partially parsed test_eq_with_different_end_position. Retrieved 4/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error1'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'min_length'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'email'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'max_length'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'max_length'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'max_length'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(not var_3 == 'not a message')
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(not var_3 == None)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_message_equality_with_different_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_different_start_and_end_position. Retrieved 6/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error message 2'
    var_5 = module_0.Message(text=var_4, code=var_1, key=var_2)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'error2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 3
    var_7 = [var_2, var_6]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error message'
    var_3 = 'error'

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = 20
    var_4 = 'Error message'
    var_5 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = None
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_4, end_position=var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = bool(var_3 != 'not a message')
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_eq_different_start_position. Retrieved 3/11 statements.
# Partially parsed test_eq_different_end_position. Retrieved 3/11 statements.
# Partially parsed test_eq_different_position_vs_start_end. Retrieved 2/9 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error message 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'max_length'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'min_length'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'users'
    var_2 = 3
    var_3 = 'username'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Message(text=var_0, index=var_4)
    var_6 = 4
    var_7 = [var_1, var_6, var_3]
    var_8 = module_0.Message(text=var_0, index=var_7)
    var_9 = bool(not var_5 == var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error message'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'Error message'

def test_case_0():
    var_0 = 1
    var_1 = 'Error message'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(not var_1 == 'not a message')
    assert var_2 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(not var_1 == None)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



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
    var_6 = var_3._messages
    var_7 = bool(var_3._messages == var_5)
    assert var_7 is True
    var_8 = var_3._message_dict
    var_9 = bool(var_3._message_dict == {'error_key': 'Error message'})
    assert var_9 is True

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
    var_11 = bool(var_9._messages == var_8)
    assert var_11 is True
    var_12 = var_9._message_dict
    var_13 = bool(var_9._message_dict == {'key1': 'Error 1', 'key2': 'Error 2'})
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
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
    var_16 = bool(var_14._messages == var_13)
    assert var_16 is True
    var_17 = var_14._message_dict
    var_18 = bool(var_14._message_dict == {'users': {0: {'name': 'Error 1'}, 1: {'email': 'Error 2'}}})
    assert var_18 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error message'
    var_5 = 'error_code'
    var_6 = module_0.BaseError(text=var_4, code=var_5, position=var_3)
    var_7 = module_0.Message(text=var_4, code=var_5, position=var_3)
    var_8 = [var_7]
    var_9 = var_6._messages
    var_10 = bool(var_6._messages == var_8)
    assert var_10 is True
    var_11 = var_6._message_dict
    var_12 = bool(var_6._message_dict == {'': 'Error message'})
    assert var_12 is True



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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = var_7.text
    assert var_8 == 'Error message'
    var_9 = var_7.code
    assert var_9 == 'max_length'
    var_10 = var_7.index
    var_11 = bool(var_7.index == ['username'])
    assert var_11 is True
    var_12 = module_0.Position(var_3, var_4, var_5)
    var_13 = var_7.start_position
    var_14 = bool(var_7.start_position == var_12)
    assert var_14 is True
    var_15 = module_0.Position(var_3, var_4, var_5)
    var_16 = var_7.end_position
    var_17 = bool(var_7.end_position == var_15)
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
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_1, var_1, var_2)
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = bool(not var_3 == 'not a position')
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_message_equality_with_different_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_none_position. Retrieved 5/9 statements.
# Partially parsed test_message_equality_with_start_and_end_position. Retrieved 4/8 statements.
# Partially parsed test_message_equality_with_different_start_position. Retrieved 5/10 statements.
# Partially parsed test_message_equality_with_different_end_position. Retrieved 5/10 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 'Error'
    var_3 = 'test'

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
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 'Error'
    var_3 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 'Error'
    var_4 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 20
    var_3 = 'Error'
    var_4 = 'test'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ValidationResult___iter__. Retrieved 3/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'error'



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = str(var_3)
    assert var_4 == "{'field': 'Invalid value'}"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Missing value'
    var_6 = 'missing'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = str(var_11)
    assert var_12 == "{'field1': 'Invalid value', 'field2': 'Missing value'}"



# Parsed testcases at query #7
#--------------------------




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
    var_2 = 'a'
    var_3 = 1
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = 'b'
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_8)
    var_10 = bool(not var_5 == var_9)
    assert var_10 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = module_0.Message(text=var_0, key=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = 'Error'
    var_7 = 'test'
    var_8 = module_0.Message(text=var_6, code=var_7, position=var_2)
    var_9 = module_0.Message(text=var_6, code=var_7, position=var_5)
    var_10 = bool(not var_8 == var_9)
    assert var_10 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, start_position=var_4, end_position=var_7)
    var_9 = (var_2, var_3)
    var_10 = (var_5, var_6)
    var_11 = module_0.Message(text=var_0, code=var_1, start_position=var_9, end_position=var_10)
    var_12 = bool(var_8 == var_11)
    assert var_12 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, start_position=var_4, end_position=var_7)
    var_9 = 5
    var_10 = 6
    var_11 = (var_9, var_10)
    var_12 = (var_5, var_6)
    var_13 = module_0.Message(text=var_0, code=var_1, start_position=var_11, end_position=var_12)
    var_14 = bool(not var_8 == var_13)
    assert var_14 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, start_position=var_4, end_position=var_7)
    var_9 = (var_2, var_3)
    var_10 = 5
    var_11 = 6
    var_12 = (var_10, var_11)
    var_13 = module_0.Message(text=var_0, code=var_1, start_position=var_9, end_position=var_12)
    var_14 = bool(not var_8 == var_13)
    assert var_14 is True

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
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'Invalid data'



# Parsed testcases at query #9
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
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "BaseError([Message(text='Invalid value', code='invalid', index=['field'])]"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Missing value'
    var_6 = 'missing'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = repr(var_11)
    assert var_12 == "BaseError([Message(text='Invalid value', code='invalid', index=['field1']), Message(text='Missing value', code='missing', index=['field2'])]"



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_message_repr_with_position. Retrieved 4/7 statements.
# Partially parsed test_message_repr_with_start_and_end_position. Retrieved 5/9 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "Message(text='Error message', code='error_code', index=['username'])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = repr(var_6)
    assert var_7 == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'Error message'
    var_3 = 'error_code'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 'Error message'
    var_4 = 'error_code'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='Error message', code='custom')"



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_message_repr_with_text_code_and_position. Retrieved 3/6 statements.
# Partially parsed test_message_repr_with_text_code_start_and_end_position. Retrieved 4/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "Message(text='Error message', code='error_code', index=['username'])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = repr(var_6)
    assert var_7 == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_case_0():
    var_0 = 1
    var_1 = 'Error message'
    var_2 = 'error_code'

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 'Error message'
    var_3 = 'error_code'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__iter__returns_iterator_with_value_and_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'test_error'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_start_position_not_equal_to_end_position. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Error'
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_start_position_not_equal_to_end_position. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Error message'
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__iter__returns_iterator_with_value_and_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'test_error'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_message_repr_with_position. Retrieved 3/6 statements.
# Partially parsed test_message_repr_with_start_and_end_position. Retrieved 4/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='custom', index=['username'])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'users'
    var_2 = 3
    var_3 = 'username'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Message(text=var_0, index=var_4)
    var_6 = repr(var_5)
    assert var_6 == "Message(text='Error message', code='custom', index=['users', 3, 'username'])"

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'Error message'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 'Error message'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_repr_with_text_code_and_position. Retrieved 4/8 statements.
# Partially parsed test_repr_with_text_code_start_and_end_position. Retrieved 5/10 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "Message(text='Error message', code='error_code', index=['username'])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = repr(var_6)
    assert var_7 == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'Error message'
    var_3 = 'error_code'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 'Error message'
    var_4 = 'error_code'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='Error message', code='custom')"



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_ValidationResult___iter___with_value. Retrieved 2/3 statements.
# Partially parsed test_ValidationResult___iter___with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_message_repr_with_text_code_and_position. Retrieved 4/7 statements.
# Partially parsed test_message_repr_with_text_code_and_start_end_positions. Retrieved 5/9 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error message', code='error_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = repr(var_3)
    assert var_4 == "Message(text='Error message', code='error_code', index=['username'])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'users'
    var_3 = 3
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = repr(var_6)
    assert var_7 == "Message(text='Error message', code='error_code', index=['users', 3, 'username'])"

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'Error message'
    var_3 = 'error_code'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = 'Error message'
    var_4 = 'error_code'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='Error message', code='custom')"



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validation_result_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_validation_result_iter_with_error. Retrieved 1/4 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'test error'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_iter_with_value. Retrieved 2/3 statements.
# Partially parsed test_iter_with_error. Retrieved 1/4 statements.
# Partially parsed test_iter_with_none. Retrieved 1/2 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)

def test_case_0():
    var_0 = 'Invalid data'

import typesystem.base as module_0

def test_case_0():
    var_0 = module_0.ValidationResult()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_start_position_not_equal_to_end_position. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'Test message'
    var_1 = 'test_code'
    var_2 = 1
    var_3 = 2



