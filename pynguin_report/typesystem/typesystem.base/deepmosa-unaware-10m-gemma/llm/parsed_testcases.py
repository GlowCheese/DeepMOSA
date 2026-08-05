####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple error'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = str(var_3)
    assert var_4 == 'Simple error'
    var_5 = 'Field error'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_5, key=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = str(var_9)
    assert var_10 == "{'username': 'Field error'}"
    var_11 = 'Nested error'
    var_12 = 'users'
    var_13 = 0
    var_14 = 'email'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Message(text=var_11, index=var_15)
    var_17 = 'Root error'
    var_18 = 'root'
    var_19 = module_0.Message(text=var_17, key=var_18)
    var_20 = [var_16, var_19]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = "{'users': {0: {'email': 'Nested error'}}, 'root': 'Root error'}"
    var_23 = str(var_21)
    var_24 = 'Empty index error'
    var_25 = []
    var_26 = module_0.Message(text=var_24, index=var_25)
    var_27 = [var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = str(var_28)
    assert var_29 == 'Empty index error'
    var_30 = 'Deep error'
    var_31 = 1
    var_32 = 'sub_key'
    var_33 = [var_31, var_32]
    var_34 = module_0.Message(text=var_30, index=var_33)
    var_35 = [var_34]
    var_36 = module_0.ValidationError(messages=var_35)
    var_37 = str(var_36)
    assert var_37 == "{1: {'sub_key': 'Deep error'}}"



# Parsed testcases at query #2
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, end_position=var_8)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'error text'
    var_7 = 'err_code'
    var_8 = module_0.Message(text=var_6, code=var_7)
    var_9 = [var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = module_0.ValidationResult(error=var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = module_0.ValidationResult()
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2



# Parsed testcases at query #5
#--------------------------


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
    var_8 = module_0.Position(var_0, var_1, var_2)
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'list'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_8)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_7)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, start_position=var_3, end_position=var_7)



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 20
    var_7 = module_0.Position(var_5, var_0, var_6)
    var_8 = 'error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'different error'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_23)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, index=var_27, start_position=var_3, end_position=var_7)



# Parsed testcases at query #8
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, start_position=var_3, end_position=var_8)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = 'admin'
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_24, position=var_3)
    var_26 = [var_12, var_0]
    var_27 = module_0.Message(text=var_9, code=var_10, index=var_26, position=var_3)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, position=var_8)
    var_30 = module_0.Message(text=var_9, code=var_10, start_position=var_3, end_position=var_8)
    var_31 = module_0.Message(text=var_9, code=var_10, start_position=var_3, end_position=var_3)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 11
    var_7 = module_0.Position(var_5, var_1, var_6)
    var_8 = 'error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, index=var_27, start_position=var_3, end_position=var_4)
    var_29 = [var_11, var_12]
    var_30 = module_0.Message(text=var_8, code=var_9, index=var_29, start_position=var_3, end_position=var_4)
    var_31 = [var_11, var_12]
    var_32 = module_0.Message(text=var_8, code=var_9, index=var_31, start_position=var_3, end_position=var_7)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 6
    var_7 = 11
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_3, end_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, position=var_3)



# Parsed testcases at query #16
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22)
    var_24 = 'admin'
    var_25 = 'admins'
    var_26 = [var_25, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_24, index=var_26)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, index=var_28)
    var_30 = [var_12, var_13]
    var_31 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_30, start_position=var_8)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 20
    var_7 = module_0.Position(var_5, var_0, var_6)
    var_8 = 'Error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'Different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, position=var_3)
    var_25 = 'different_key'
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_25)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, start_position=var_7)
    var_29 = [var_11, var_12]
    var_30 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_29, start_position=var_3, end_position=var_7)



# Parsed testcases at query #18
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, index=var_28, start_position=var_3, end_position=var_8)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 11
    var_7 = module_0.Position(var_5, var_1, var_6)
    var_8 = 'error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, start_position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, end_position=var_7)
    var_29 = [var_11, var_12]
    var_30 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_29, start_position=var_3, end_position=var_3)



# Parsed testcases at query #20
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, end_position=var_8)



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 6
    var_7 = 11
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_9, code=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_9, code=var_10, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_10, index=var_18, position=var_3)
    var_20 = [var_11, var_12]
    var_21 = module_0.Message(text=var_9, index=var_20, position=var_3)
    var_22 = 'custom'
    var_23 = [var_11, var_12]
    var_24 = module_0.Message(text=var_9, code=var_22, index=var_23, position=var_3)
    var_25 = 'other'
    var_26 = [var_11, var_12]
    var_27 = module_0.Message(text=var_9, code=var_25, index=var_26, position=var_3)
    var_28 = [var_11, var_0]
    var_29 = module_0.Message(text=var_9, code=var_10, index=var_28, position=var_3)
    var_30 = []
    var_31 = module_0.Message(text=var_9, code=var_10, index=var_30, position=var_3)
    var_32 = [var_11, var_12]
    var_33 = module_0.Message(text=var_9, code=var_10, index=var_32, position=var_8)
    var_34 = [var_11, var_12]
    var_35 = module_0.Message(text=var_9, code=var_10, index=var_34, start_position=var_3, end_position=var_8)



# Parsed testcases at query #23
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, index=var_26, start_position=var_3, end_position=var_8)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = module_0.Position(var_0, var_1, var_1)
    var_3 = module_0.Position(var_0, var_1, var_1)
    var_4 = 2
    var_5 = 10
    var_6 = module_0.Position(var_4, var_0, var_5)
    var_7 = 'error'
    var_8 = 'err_code'
    var_9 = 'user'
    var_10 = 'users'
    var_11 = 0
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_12, position=var_2)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_14, position=var_3)
    var_16 = 'different error'
    var_17 = [var_10, var_11]
    var_18 = module_0.Message(text=var_16, code=var_8, key=var_9, index=var_17, position=var_2)
    var_19 = 'other_code'
    var_20 = [var_10, var_11]
    var_21 = module_0.Message(text=var_7, code=var_19, key=var_9, index=var_20, position=var_2)
    var_22 = [var_10, var_0]
    var_23 = module_0.Message(text=var_7, code=var_8, index=var_22, position=var_2)
    var_24 = [var_11, var_9]
    var_25 = module_0.Message(text=var_7, code=var_8, index=var_24, position=var_2)
    var_26 = [var_10, var_11]
    var_27 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_26, start_position=var_2, end_position=var_6)
    var_28 = module_0.Message(text=var_7)
    var_29 = None
    var_30 = module_0.Message(text=var_7, code=var_29, index=var_29)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 11
    var_7 = module_0.Position(var_5, var_1, var_6)
    var_8 = 'Error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'Different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, start_position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, end_position=var_7)
    var_29 = module_0.Message(text=var_8)
    var_30 = None
    var_31 = module_0.Message(text=var_8, code=var_30)
    var_32 = module_0.Message(text=var_8, position=var_3)



# Parsed testcases at query #26
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_24, position=var_3)
    var_26 = module_0.Message(text=var_9, code=var_10, start_position=var_3, end_position=var_8)
    var_27 = module_0.Message(text=var_9, code=var_10, start_position=var_4, end_position=var_8)
    var_28 = module_0.Message(text=var_9, code=var_10, start_position=var_3, end_position=var_3)



# Parsed testcases at query #27
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22)
    var_24 = 'other'
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_9, code=var_10, index=var_25)
    var_27 = [var_12, var_13]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, start_position=var_8)



# Parsed testcases at query #28
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, start_position=var_3, end_position=var_8)
    var_30 = [var_12, var_13]
    var_31 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_30, start_position=var_3, end_position=var_4)



# Parsed testcases at query #29
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, position=var_3)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, position=var_8)
    var_30 = [var_12, var_13]
    var_31 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_30, start_position=var_3, end_position=var_8)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 11
    var_7 = module_0.Position(var_5, var_1, var_6)
    var_8 = 'error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'different error'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, start_position=var_3, end_position=var_7)



# Parsed testcases at query #31
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = [var_12, var_0]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, position=var_3)
    var_26 = [var_12, var_13]
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, start_position=var_3, end_position=var_8)



# Parsed testcases at query #32
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_9, code=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_9, code=var_10, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_10, index=var_18, position=var_3)
    var_20 = [var_11, var_12]
    var_21 = module_0.Message(text=var_9, code=var_17, index=var_20, position=var_3)
    var_22 = [var_11, var_0]
    var_23 = module_0.Message(text=var_9, code=var_10, index=var_22, position=var_3)
    var_24 = [var_11, var_12]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, start_position=var_3, end_position=var_8)
    var_26 = [var_11, var_12]
    var_27 = module_0.Message(text=var_9, code=var_10, index=var_26, position=var_8)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 3
    var_6 = 6
    var_7 = module_0.Position(var_0, var_5, var_6)
    var_8 = 'Error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'Different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, start_position=var_3, end_position=var_3)
    var_29 = [var_11, var_12]
    var_30 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_29, start_position=var_3, end_position=var_7)



# Parsed testcases at query #34
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'users'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different error'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19)
    var_21 = 'different_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22)
    var_24 = 'other'
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_9, code=var_10, index=var_25)
    var_27 = [var_12, var_13]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, position=var_8)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, start_position=var_8, end_position=var_8)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 11
    var_7 = module_0.Position(var_5, var_1, var_6)
    var_8 = 'error'
    var_9 = 'err_code'
    var_10 = 'user'
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_9, key=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_8, code=var_20, key=var_10, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_25, start_position=var_7, end_position=var_7)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_27, start_position=var_3, end_position=var_4)



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
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = module_0.Position(var_4, var_5, var_6)
    var_8 = 'error 1'
    var_9 = 'err1'
    var_10 = 'a'
    var_11 = 0
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_8, code=var_9, index=var_12, position=var_3)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_8, code=var_9, index=var_14, position=var_3)
    var_16 = 'error 2'
    var_17 = 'err2'
    var_18 = 'b'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_16, code=var_17, index=var_19, position=var_7)
    var_21 = [var_13, var_20]
    var_22 = module_0.ValidationError(messages=var_21)
    var_23 = [var_13, var_20]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = [var_13]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = [var_20]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = [var_13, var_20]



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.ValidationResult(value=var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'Invalid input'
    var_9 = 'error_code'
    var_10 = 'field'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = [var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = module_0.ValidationResult(error=var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 10
    var_18 = None
    var_19 = module_0.ValidationResult(value=var_17, error=var_18)
    var_20 = module_0.ValidationResult(value=var_18, error=var_18)
    var_21 = list(var_20)



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Simple error'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = str(var_3)
    assert var_4 == 'Simple error'
    var_5 = 'Field error'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_5, key=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = str(var_9)
    assert var_10 == "{'username': 'Field error'}"
    var_11 = 'Error A'
    var_12 = 'users'
    var_13 = 0
    var_14 = 'name'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Message(text=var_11, index=var_15)
    var_17 = 'Error B'
    var_18 = 1
    var_19 = 'age'
    var_20 = [var_12, var_18, var_19]
    var_21 = module_0.Message(text=var_17, index=var_20)
    var_22 = 'Error C'
    var_23 = 'global_key'
    var_24 = module_0.Message(text=var_22, key=var_23)
    var_25 = [var_16, var_21, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = {var_14: var_11}
    var_28 = {var_19: var_17}
    var_29 = {var_13: var_27, var_18: var_28}
    var_30 = {var_12: var_29, var_23: var_22}
    var_31 = str(var_30)
    var_32 = str(var_26)
    var_33 = 'Root error'
    var_34 = []
    var_35 = module_0.Message(text=var_33, index=var_34)
    var_36 = [var_35]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = str(var_37)
    assert var_38 == "{'': 'Root error'}"



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success_data'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'error text'
    var_5 = 'err_code'
    var_6 = module_0.ValidationError(text=var_4, code=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = module_0.ValidationResult()
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #5
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'user'
    var_12 = 'meta'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = 'other'
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_25, position=var_3)
    var_27 = [var_12, var_13]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, position=var_8)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, start_position=var_3, end_position=var_8)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error 1'
    var_5 = 'err1'
    var_6 = 'user'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7, position=var_3)
    var_9 = 'Error 2'
    var_10 = 'err2'
    var_11 = 'admin'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    var_14 = [var_6]
    var_15 = module_0.Message(text=var_4, code=var_5, index=var_14, position=var_3)
    var_16 = [var_8, var_13]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_8, var_13]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_8, var_15]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_8, var_13]
    var_23 = [var_8]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = [var_6]
    var_26 = module_0.Message(text=var_4, code=var_5, index=var_25, position=var_3)
    var_27 = [var_11]
    var_28 = module_0.Message(text=var_9, code=var_10, index=var_27)
    var_29 = [var_26, var_28]
    var_30 = module_0.ValidationError(messages=var_29)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = 'err2'
    var_6 = 'nested'
    var_7 = 'item'
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_11 = [var_3, var_9]
    var_12 = module_0.ValidationError(messages=var_11)
    var_13 = [var_3, var_9]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = [var_10, var_9]
    var_16 = module_0.ValidationError(messages=var_15)
    var_17 = [var_3]
    var_18 = module_0.ValidationError(messages=var_17)
    var_19 = [var_3, var_9]
    var_20 = module_0.ParseError(messages=var_19)
    var_21 = 'Single'
    var_22 = 'code'
    var_23 = module_0.ValidationError(text=var_21, code=var_22)
    var_24 = module_0.ValidationError(text=var_21, code=var_22)
    var_25 = 'Different'
    var_26 = module_0.ValidationError(text=var_25, code=var_22)
    var_27 = [var_3, var_9]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = [var_9, var_3]
    var_30 = module_0.ValidationError(messages=var_29)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 6
    var_5 = 11
    var_6 = module_0.Position(var_0, var_4, var_5)
    var_7 = 'Error 1'
    var_8 = 'err1'
    var_9 = 'users'
    var_10 = 0
    var_11 = [var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11, position=var_3)
    var_13 = 'Error 2'
    var_14 = 'err2'
    var_15 = [var_9, var_0]
    var_16 = module_0.Message(text=var_13, code=var_14, index=var_15, position=var_6)
    var_17 = [var_9, var_10]
    var_18 = module_0.Message(text=var_7, code=var_8, index=var_17, position=var_3)
    var_19 = [var_12, var_16]
    var_20 = module_0.ValidationError(messages=var_19)
    var_21 = [var_12, var_16]
    var_22 = module_0.ValidationError(messages=var_21)
    var_23 = [var_12, var_18]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = [var_12, var_16]
    var_26 = [var_12]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = 'Single'
    var_29 = 'single'
    var_30 = module_0.Message(text=var_28, code=var_29)
    var_31 = [var_30]
    var_32 = module_0.ValidationError(messages=var_31)
    var_33 = module_0.Message(text=var_28, code=var_29)
    var_34 = [var_33]
    var_35 = module_0.ValidationError(messages=var_34)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = 'err2'
    var_6 = 'parent'
    var_7 = 'child'
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8)
    var_10 = 'Error 3'
    var_11 = 'err3'
    var_12 = module_0.Message(text=var_10, code=var_11)
    var_13 = [var_3, var_9]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = [var_3, var_9]
    var_16 = module_0.ValidationError(messages=var_15)
    var_17 = [var_3, var_12]
    var_18 = module_0.ValidationError(messages=var_17)
    var_19 = [var_9, var_3]
    var_20 = module_0.ValidationError(messages=var_19)
    var_21 = [var_3, var_9]
    var_22 = module_0.ParseError(messages=var_21)
    var_23 = [var_3, var_9]



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error 1'
    var_5 = 'err1'
    var_6 = 'key1'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7, position=var_3)
    var_9 = 'Error 2'
    var_10 = 'err2'
    var_11 = 'key2'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    var_14 = [var_8, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = [var_8, var_13]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = 'Different text'
    var_19 = [var_6]
    var_20 = module_0.Message(text=var_18, code=var_5, index=var_19, position=var_3)
    var_21 = [var_20, var_13]
    var_22 = module_0.ValidationError(messages=var_21)
    var_23 = [var_13, var_8]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = [var_8, var_13]
    var_26 = [var_8]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = [var_6]
    var_29 = module_0.Message(text=var_4, code=var_5, index=var_28, position=var_3)
    var_30 = [var_29, var_13]
    var_31 = module_0.ValidationError(messages=var_30)



# Parsed testcases at query #11
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
    var_9 = 'error'
    var_10 = 'err_code'
    var_11 = 'field'
    var_12 = 'a'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = 'other_code'
    var_22 = [var_12, var_13]
    var_23 = module_0.Message(text=var_9, code=var_21, key=var_11, index=var_22, position=var_3)
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_25, position=var_3)
    var_27 = [var_12, var_13]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, start_position=var_8, end_position=var_8)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, index=var_29, position=var_3)
    var_31 = [var_12, var_13]
    var_32 = module_0.Message(text=var_9, code=var_10, index=var_31, position=var_3)
    var_33 = module_0.Message(text=var_9, code=var_10, start_position=var_3, end_position=var_8)
    var_34 = module_0.Message(text=var_9, code=var_10, start_position=var_4, end_position=var_8)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 5
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 11
    var_6 = module_0.Position(var_0, var_5, var_2)
    var_7 = 2
    var_8 = module_0.Position(var_7, var_1, var_2)
    var_9 = 6
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 'not a position'



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'error 1'
    var_5 = 'err1'
    var_6 = 'a'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    var_9 = 'error 2'
    var_10 = 'err2'
    var_11 = 'b'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    var_14 = [var_6]
    var_15 = module_0.Message(text=var_4, code=var_5, index=var_14)
    var_16 = [var_8, var_13]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_8, var_13]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_8, var_15]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_13, var_8]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'single'
    var_25 = 'code'
    var_26 = module_0.ValidationError(text=var_24, code=var_25)
    var_27 = module_0.ValidationError(text=var_24, code=var_25)
    var_28 = 'different'
    var_29 = module_0.ValidationError(text=var_28, code=var_25)
    var_30 = [var_8, var_13]
    var_31 = module_0.ParseError(messages=var_30)
    var_32 = 'p'
    var_33 = module_0.Message(text=var_32, position=var_3)
    var_34 = 0
    var_35 = module_0.Position(var_34, var_34, var_34)
    var_36 = module_0.Message(text=var_32, position=var_35)
    var_37 = [var_33]
    var_38 = module_0.ValidationError(messages=var_37)
    var_39 = [var_36]
    var_40 = module_0.ValidationError(messages=var_39)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'error 1'
    var_5 = 'err1'
    var_6 = 'key1'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_8 = 'error 2'
    var_9 = 'err2'
    var_10 = 'parent'
    var_11 = 'child'
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_8, code=var_9, index=var_12)
    var_14 = 'error 3'
    var_15 = module_0.Message(text=var_14, position=var_3)
    var_16 = [var_7, var_13]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_7, var_13]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_7, var_15]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_13, var_7]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = [var_7, var_13]



