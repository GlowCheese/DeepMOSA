####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_message_eq_different_positions. Retrieved 3/14 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'user'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validation_result_iter_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'success'
    var_4 = next(var_2)
    assert var_4 is None

import builtins as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'invalid data'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Exception(*var_1, **var_2)
    var_4 = module_1.ValidationResult(error=var_3)
    var_5 = iter(var_4)
    var_6 = next(var_5)
    assert var_6 is None
    var_7 = next(var_5)
    var_8 = bool(var_7 == var_3)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #4
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
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = (var_0, var_1, var_2)
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = bool(var_3 != None)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_repr_with_position_single. Retrieved 1/7 statements.
# Partially parsed test_repr_with_start_and_end_position. Retrieved 1/8 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='error', code='err_code')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='error', code='custom')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'users'
    var_2 = 0
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = repr(var_4)
    assert var_5 == "Message(text='error', code='custom', index=['users', 0])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='error', code='custom', index=['username'])"

def test_case_0():
    var_0 = 'error'

def test_case_0():
    var_0 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = repr(var_2)
    var_5 = repr(var_3)
    var_6 = bool(var_4 == var_5)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_message_eq_different_positions. Retrieved 3/12 statements.
# Partially parsed test_message_eq_different_start_end_positions. Retrieved 4/13 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'
    var_3 = None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 != 'not a message')
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = 1
    var_4 = 5
    var_5 = module_0.Position(var_3, var_4, var_4)
    var_6 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = var_6.messages()
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_6.messages()[var_9]
    var_11 = var_10.text
    var_12 = bool(var_11 == var_0)
    assert var_12 is True
    var_13 = var_6.messages()[var_9]
    var_14 = var_13.code
    var_15 = bool(var_14 == var_1)
    assert var_15 is True
    var_16 = var_6.messages()[var_9]
    var_17 = var_16.index
    var_18 = bool(var_17 == [var_2])
    assert var_18 is True
    var_19 = var_6.messages()[var_9]
    var_20 = var_19.start_position
    var_21 = bool(var_20 == var_5)
    assert var_21 is True
    var_22 = var_6.messages()[var_9]
    var_23 = var_22.end_position
    var_24 = bool(var_23 == var_5)
    assert var_24 is True
    var_25 = var_6['username']
    var_26 = bool(var_6['username'] == var_0)
    assert var_26 is True

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
    var_10 = 'age'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = var_14.messages()
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14['users'][0]['name']
    assert var_17 == 'Error 1'
    var_18 = var_14['users'][1]['age']
    assert var_18 == 'Error 2'
    var_19 = len(var_14)
    assert var_19 == 1

import typesystem.base as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'msg'
    var_2 = module_0.Message(text=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(text=var_0, messages=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.BaseError(messages=var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_base_error_str_single_message. Retrieved 5/16 statements.
# Partially parsed test_base_error_str_multiple_messages. Retrieved 12/23 statements.
# Partially parsed test_base_error_str_nested_messages. Retrieved 8/19 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'error_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = module_0.ValidationError(text=var_0, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Field error'
    var_1 = 'err1'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Other error'
    var_6 = 'err2'
    var_7 = 'other'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.ValidationError(messages=var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Nested error'
    var_1 = 'err1'
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = [var_6]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'code1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = 'code2'
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_5, code=var_6, index=var_7)
    var_9 = [var_4]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = [var_8]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = bool(var_10 != var_12)
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_6 != 'not an error object')
    assert var_7 is True
    var_8 = bool(var_6 != 123)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'other'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_6)
    var_8 = [var_7]
    var_9 = module_0.BaseError(messages=var_8)
    var_10 = [var_4]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = bool(var_11 != var_9)
    assert var_12 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = [var_6]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'code1'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = 'code2'
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_5, code=var_6, index=var_7)
    var_9 = [var_4]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = [var_8]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = bool(var_10 != var_12)
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_6 != 'not an error')
    assert var_7 is True
    var_8 = bool(var_6 != 123)
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = [var_6]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'code1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = 'code2'
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_5, code=var_6, index=var_7)
    var_9 = [var_4]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = [var_8]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = bool(var_10 != var_12)
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_6 != 'not an error object')
    assert var_7 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_6 != None)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




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
    var_11 = module_0.ValidationError(messages=var_10)
    var_12 = [var_4, var_9]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = bool(var_11 == var_13)
    assert var_14 is True



# Parsed testcases at query #15
#--------------------------




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
    var_12 = var_11._messages
    var_13 = len(var_12)
    var_14 = bool(var_13 != 1)
    assert var_14 is True
    var_15 = str(var_11)
    var_16 = dict(var_11)
    var_17 = str(var_16)
    var_18 = bool(var_15 == var_17)
    assert var_18 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error with index'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = var_6._messages
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_6._messages[var_9]
    var_11 = var_10.index
    var_12 = len(var_11)
    var_13 = bool(var_12 > 0)
    assert var_13 is True
    var_14 = str(var_6)
    var_15 = dict(var_6)
    var_16 = str(var_15)
    var_17 = bool(var_14 == var_16)
    assert var_17 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.
# Partially parsed test_message_eq_different_start_end_position. Retrieved 4/13 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 3
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)
    var_8 = bool(var_4 != var_7)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'
    var_3 = None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 != 'Not a message')
    assert var_2 is True
    var_3 = bool(var_1 != None)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    assert var_3 is False
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_message_eq_different_positions. Retrieved 3/12 statements.
# Partially parsed test_message_eq_different_start_end_positions. Retrieved 4/13 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = 'users'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_2, var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'
    var_3 = None

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 != 'Not a message')
    assert var_2 is True
    var_3 = bool(var_1 != None)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple error'
    var_1 = 'err_01'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Simple error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Field error'
    var_1 = 'field'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'Nested error'
    var_5 = 'parent'
    var_6 = 'child'
    var_7 = [var_5, var_6]
    var_8 = module_0.Message(text=var_4, index=var_7)
    var_9 = [var_3, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = {var_6: var_4}
    var_12 = {var_1: var_0, var_5: var_11}
    var_13 = str(var_12)
    var_14 = str(var_10)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = [var_6]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'code1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = 'code2'
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_5, code=var_6, index=var_7)
    var_9 = [var_4]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = [var_8]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = bool(var_10 != var_12)
    assert var_13 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = bool(var_6 != 'not an error')
    assert var_7 is True
    var_8 = bool(var_6 != 123)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = [var_4]
    var_8 = module_0.ValidationError(messages=var_7)
    var_9 = bool(var_6 != var_8)
    assert var_9 is True



# Parsed testcases at query #4
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
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_2, var_2)
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 4
    var_5 = module_0.Position(var_0, var_1, var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = (var_0, var_1, var_2)
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = bool(var_3 != None)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = 'field'
    var_3 = 0
    var_4 = 'sub'
    var_5 = [var_3, var_4]
    var_6 = None
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5, start_position=var_6, end_position=var_6)
    var_8 = [var_3, var_4]
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, start_position=var_6, end_position=var_6)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 1
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = None
    var_3 = module_0.Message(text=var_0, code=var_2)
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'err2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    var_12 = [var_4, var_9]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = bool(var_11 == var_13)
    assert var_14 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validation_result_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == ['success', None])
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'invalid input'
    var_1 = module_0.ValidationResult(error=var_0)
    var_2 = list(var_1)
    var_3 = bool(var_2 == [None, 'invalid input'])
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2)
    assert var_3 is False
    var_4 = bool(var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_message_eq_false_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'error'
    var_1 = 1
    var_2 = 2

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    assert var_4 is False
    var_5 = bool(var_2 == var_4)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    assert var_4 is False
    var_5 = bool(var_2 == var_4)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



