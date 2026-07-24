####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    var_3 = 'error'
    var_4 = 'err_code'
    var_5 = 'field'
    var_6 = module_0.Message(text=var_3, code=var_4, key=var_5, position=var_2)
    var_7 = module_0.BaseError(text=var_3, code=var_4, key=var_5, position=var_2)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7['field']
    assert var_9 == 'error'
    var_10 = var_7.messages()[var_1]
    var_11 = var_10.text
    assert var_11 == 'error'
    var_12 = var_7.messages()[var_1]
    var_13 = var_12.code
    assert var_13 == 'err_code'
    var_14 = var_7.messages()[var_1]
    var_15 = var_14.index
    var_16 = bool(var_15 == ['field'])
    assert var_16 is True
    var_17 = var_7.messages()[var_1]
    var_18 = var_17.start_position
    var_19 = bool(var_18 == var_2)
    assert var_19 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'err1'
    var_1 = 'c1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'err2'
    var_8 = 'c2'
    var_9 = 1
    var_10 = 'age'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'users'
    var_17 = bool('users' in var_14)
    assert var_17 is True
    var_18 = var_14['users']['0']
    assert var_18 == 'err1'
    var_19 = var_14['users']['1']
    assert var_19 == 'err2'
    var_20 = var_14.messages()
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_14.messages()[var_3]
    var_23 = var_22.text
    assert var_23 == 'err1'
    var_24 = var_14.messages()[var_9]
    var_25 = var_24.text
    assert var_25 == 'err2'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'err1'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Message(text=var_0, index=var_4)
    var_6 = 'err2'
    var_7 = 'd'
    var_8 = [var_1, var_7]
    var_9 = module_0.Message(text=var_6, index=var_8)
    var_10 = [var_5, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = var_11['a']['b']['c']
    assert var_12 == 'err1'
    var_13 = var_11['a']['d']
    assert var_13 == 'err2'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'root_error'
    var_1 = []
    var_2 = module_0.Message(text=var_0, index=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = ''
    var_6 = bool('' in var_4)
    assert var_6 is True
    var_7 = var_4['']
    assert var_7 == 'root_error'
    var_8 = len(var_4)
    assert var_8 == 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = 1
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
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'b'
    var_6 = [var_5, var_2]
    var_7 = module_0.Message(text=var_0, index=var_6)
    var_8 = bool(var_4 != var_7)
    assert var_8 is True

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



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2)
    var_4 = bool(var_1 != var_3)
    assert var_4 is True



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



# Parsed testcases at query #5
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error text'
    var_1 = 'error_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='Error text', code='error_code', index=[])])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error text'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = repr(var_6)
    var_8 = bool(var_7 == f'BaseError([{var_4!r}])')
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'key'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    var_9 = [var_3, var_8]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = repr(var_10)
    var_12 = bool(var_11 == f'BaseError([{var_3!r}, {var_8!r}])')
    assert var_12 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple error'
    var_1 = 'simple_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1)
    var_5 = repr(var_4)
    assert var_5 == "BaseError(text='Simple error', code='simple_code')"



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4)
    var_6 = [var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_1 = 1
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 2
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'field'
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
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validation_result_iter_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_2[0]
    assert var_4 == 'success'
    var_5 = var_2[1]
    assert var_5 is None

import builtins as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'validation failed'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Exception(*var_1, **var_2)
    var_4 = module_1.ValidationResult(error=var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    assert var_7 is None
    var_8 = var_5[1]
    var_9 = bool(var_5[1] == var_3)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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
    var_0 = 'Error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
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
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 != 'Not a message object')
    assert var_2 is True
    var_3 = bool(var_1 != None)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 4/14 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 3
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)
    var_8 = bool(var_4 != var_7)
    assert var_8 is True

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 2
    var_3 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1)
    var_3 = []
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = bool(var_2 == var_4)
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
    var_5 = bool(var_2 != var_4)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_1'
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



# Parsed testcases at query #2
#--------------------------




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
    var_0 = 'Error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
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
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3)
    var_5 = bool(var_2 == var_4)
    assert var_5 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a Message object'
    var_3 = bool(var_1 != var_2)
    assert var_3 is True



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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validation_result_iter_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = tuple(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_2[0]
    assert var_4 == 'success'
    var_5 = var_2[1]
    assert var_5 is None

import builtins as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'validation failed'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Exception(*var_1, **var_2)
    var_4 = module_1.ValidationResult(error=var_3)
    var_5 = tuple(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_5[0]
    assert var_7 is None
    var_8 = var_5[1]
    var_9 = bool(var_5[1] == var_3)
    assert var_9 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = 'username'
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
    var_1 = 'err_01'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 1
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_6)
    var_8 = bool(var_4 != var_7)
    assert var_8 is True

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = 'user'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = [var_2]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = bool(var_1 != 'not a message')
    assert var_2 is True
    var_3 = bool(var_1 != None)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = 'code2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4]
    var_11 = module_0.ValidationError(messages=var_10)
    var_12 = [var_4]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = [var_9]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = 'text'
    var_17 = 'code'
    var_18 = 'key'
    var_19 = module_0.BaseError(text=var_16, code=var_17, key=var_18)
    var_20 = bool(var_11 == var_13)
    assert var_20 is True
    var_21 = bool(var_11 != var_15)
    assert var_21 is True
    var_22 = bool(var_11 != var_19)
    assert var_22 is True



