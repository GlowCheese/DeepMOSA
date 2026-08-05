####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 0
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #2
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='error', code='custom')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'val_err'
    var_2 = 'users'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = repr(var_5)
    assert var_6 == "Message(text='error', code='val_err', index=['users', 0])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'line1:col1'
    var_1 = 'error'
    var_2 = module_0.Message(text=var_1, position=var_0)
    var_3 = repr(var_2)
    var_4 = f"Message(text='error', code='custom', position={repr(var_0)})"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'line1:col1'
    var_1 = 'line1:col5'
    var_2 = 'error'
    var_3 = module_0.Message(text=var_2, start_position=var_0, end_position=var_1)
    var_4 = repr(var_3)
    var_5 = f"Message(text='error', code='custom', start_position={repr(var_0)}, end_position={repr(var_1)})"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='error', code='custom', index=['username'])"



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validation_result_iter_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import typesystem.base as module_0

def test_case_0():
    var_0 = 'invalid input'
    var_1 = module_0.ValidationResult(error=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #5
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
    var_4 = (var_0, var_1, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #6
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = 1
    var_4 = 5
    var_5 = 10
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = error.messages()[var_9]
    var_11 = var_10.text
    var_12 = error.messages()[var_9]
    var_13 = var_12.code
    var_14 = error.messages()[var_9]
    var_15 = var_14.index
    var_16 = error.messages()[var_9]
    var_17 = var_16.start_position
    var_18 = error.messages()[var_9]
    var_19 = var_18.end_position

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
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = var_14.messages()
    var_17 = len(var_16)
    assert var_17 == 2

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple Error'
    var_1 = module_0.BaseError(text=var_0)
    var_2 = len(var_1)
    assert var_2 == 1
    var_3 = 0
    var_4 = error.messages()[var_3]
    var_5 = var_4.code
    assert var_5 == 'custom'
    var_6 = error.messages()[var_3]
    var_7 = var_6.index

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.BaseError(text=var_0)
    var_2 = 0
    var_3 = error.messages()[var_2]
    var_4 = var_3.text
    assert var_4 == 'test'



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #8
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_1)
    var_3 = 'custom'
    var_4 = []
    var_5 = module_0.Message(text=var_0, code=var_3, index=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_message_eq_with_positions. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 0
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'key1'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'key2'
    var_4 = module_0.Message(text=var_0, key=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'custom'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'

def test_case_0():
    var_0 = 10
    var_1 = 'error'
    var_2 = None



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'err_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1)
    var_5 = var_4.__str__()
    assert var_5 == 'Invalid input'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'c1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'c2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = var_11.__str__()
    assert var_12 == "{'field1': 'Error 1', 'field2': 'Error 2'}"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Nested Error'
    var_1 = 'c1'
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = var_7.__str__()
    assert var_8 == "{'parent': {'child': 'Nested Error'}}"



# Parsed testcases at query #12
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a position object'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #15
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #16
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'user'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = 'user'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = module_0.Message(text=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #19
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a position object'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #21
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #22
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #23
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a position object'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #24
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #25
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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'b'
    var_6 = [var_5, var_2]
    var_7 = module_0.Message(text=var_0, index=var_6)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = None
    var_3 = module_0.Message(text=var_0, code=var_2)



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
    var_4 = (var_0, var_1, var_2)

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
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #28
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error B'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #29
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_1)
    var_3 = 'custom'
    var_4 = []
    var_5 = module_0.Message(text=var_0, code=var_3, index=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #31
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'err_code'
    var_2 = 'field'
    var_3 = 'users'
    var_4 = 0
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 0
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_1)
    var_3 = 'custom'
    var_4 = []
    var_5 = module_0.Message(text=var_0, code=var_3, index=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'



# Parsed testcases at query #33
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Position(1, 2, 3)'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #34
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #35
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a position object'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #36
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #37
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #38
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #39
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #40
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'not a position'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False



# Parsed testcases at query #41
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error text'
    var_1 = 'err_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='error text', code='err_code', index=[])])"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error text'
    var_1 = 'err_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = repr(var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'text1'
    var_1 = 'code1'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = 'text2'
    var_5 = 'code2'
    var_6 = 'key'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    var_9 = [var_3, var_8]
    var_10 = module_0.BaseError(messages=var_9)
    var_11 = repr(var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'simple error'
    var_1 = 'simple_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1)
    var_5 = repr(var_4)
    assert var_5 == "BaseError(text='simple error', code='simple_code')"



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 0
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 2
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'other'
    var_6 = module_0.Message(text=var_0, key=var_5)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = None
    var_3 = module_0.Message(text=var_0, code=var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error'



# Parsed testcases at query #3
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='success')"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'invalid input'
    var_1 = module_0.ValidationResult(error=var_0)
    var_2 = repr(var_1)
    var_3 = f'ValidationResult(error={repr(var_0)})'

import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = repr(var_1)
    assert var_2 == 'ValidationResult(value=None)'

import typesystem.base as module_0

def test_case_0():
    var_0 = module_0.ValidationResult()
    var_1 = repr(var_0)
    assert var_1 == 'ValidationResult(value=None)'



# Parsed testcases at query #4
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'c1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)
    var_9 = [var_6]
    var_10 = module_0.ValidationError(messages=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'c1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'error2'
    var_6 = [var_2]
    var_7 = module_0.Message(text=var_5, code=var_1, index=var_6)
    var_8 = [var_4]
    var_9 = module_0.BaseError(messages=var_8)
    var_10 = [var_7]
    var_11 = module_0.ValidationError(messages=var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'c1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = {var_2: var_0}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'c1'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    var_7 = [var_4]
    var_8 = module_0.BaseError(messages=var_7)



# Parsed testcases at query #5
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err_1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 2'
    var_6 = 'err_2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    var_12 = [var_4, var_9]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = 'Single error'
    var_15 = 'err_single'
    var_16 = module_0.BaseError(text=var_14, code=var_15)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validation_result_iter_with_error. Retrieved 1/5 statements.
# Partially parsed test_validation_result_unpacking. Retrieved 2/3 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

def test_case_0():
    var_0 = 'invalid error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.ValidationResult(value=var_0)



# Parsed testcases at query #7
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple error'
    var_1 = 'err_code'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1)
    var_5 = str(var_4)
    assert var_5 == 'Simple error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Err 1'
    var_1 = 'c1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Err 2'
    var_6 = 'c2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = str(var_11)
    assert var_12 == "{'field1': 'Err 1', 'field2': 'Err 2'}"

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Nested error'
    var_1 = 'c1'
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = str(var_7)
    assert var_8 == "{'parent': {'child': 'Nested error'}}"



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #9
#--------------------------




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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'b'
    var_6 = [var_5, var_2]
    var_7 = module_0.Message(text=var_0, index=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = module_0.Position(var_0, var_1)
    var_3 = 2
    var_4 = module_0.Position(var_3, var_1)
    var_5 = 'Error'
    var_6 = module_0.Message(text=var_5, position=var_2)
    var_7 = module_0.Message(text=var_5, position=var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #10
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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 3
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'other'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #11
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = [var_2, var_1]
    var_6 = module_0.Message(text=var_0, index=var_5)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #13
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_message_eq_different_positions. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = 'field'
    var_3 = 0
    var_4 = 'sub'
    var_5 = [var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5)
    var_7 = [var_3, var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'other'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)



# Parsed testcases at query #15
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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)

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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field_a'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'field_b'
    var_4 = module_0.Message(text=var_0, key=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'text'
    var_3 = {var_2: var_0}



# Parsed testcases at query #16
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_code'
    var_2 = 'user'
    var_3 = 0
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4)
    var_6 = [var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_code_1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_code_2'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 0
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'field_a'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = 'field_b'
    var_4 = module_0.Message(text=var_0, key=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'users'
    var_2 = 0
    var_3 = 'name'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Message(text=var_0, index=var_4)
    var_6 = [var_1, var_2, var_3]
    var_7 = module_0.Message(text=var_0, index=var_6)
    var_8 = 1
    var_9 = [var_1, var_8, var_3]
    var_10 = module_0.Message(text=var_0, index=var_9)



# Parsed testcases at query #18
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'user'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_message_eq_different_position. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_01'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = 'b'
    var_6 = [var_5, var_2]
    var_7 = module_0.Message(text=var_0, index=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = []
    var_2 = module_0.Message(text=var_0, index=var_1)
    var_3 = None
    var_4 = module_0.Message(text=var_0, index=var_3)

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 'Error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Not a message'



# Parsed testcases at query #21
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'Error B'
    var_3 = module_0.Message(text=var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_message_eq_different_positions. Retrieved 3/12 statements.


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

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error 1'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'error 2'
    var_4 = module_0.Message(text=var_3, code=var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 0
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    var_3 = [var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    var_5 = [var_1]
    var_6 = module_0.Message(text=var_0, index=var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'error'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'not a message'



# Parsed testcases at query #24
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error A'
    var_1 = 'err'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error B'
    var_4 = module_0.Message(text=var_3, code=var_1)



