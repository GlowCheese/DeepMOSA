####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)
    var_11 = None
    var_12 = module_0.ValidationResult(value=var_11, error=var_11)
    var_13 = iter(var_12)
    var_14 = next(var_13)
    assert var_14 is None
    var_15 = next(var_13)
    assert var_15 is None



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'field'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='Error message', code='error_code', index=['field'])])"
    var_7 = 'Error 1'
    var_8 = 'code1'
    var_9 = 'field1'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    var_11 = 'Error 2'
    var_12 = 'code2'
    var_13 = 'field2'
    var_14 = module_0.Message(text=var_11, code=var_12, key=var_13)
    var_15 = [var_10, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = repr(var_16)
    assert var_17 == "BaseError([Message(text='Error 1', code='code1', index=['field1']), Message(text='Error 2', code='code2', index=['field2'])])"
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.BaseError(text=var_0, code=var_1, position=var_21)
    var_23 = repr(var_22)
    assert var_23 == "BaseError([Message(text='Error message', code='error_code', position=Position(line_no=1, column_no=2, char_index=3))])"



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'field_name'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='Error message', code='error_code', index=['field_name'])]))"
    var_7 = 'Error 1'
    var_8 = 'error1'
    var_9 = 'field1'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    var_11 = 'Error 2'
    var_12 = 'error2'
    var_13 = 'field2'
    var_14 = module_0.Message(text=var_11, code=var_12, key=var_13)
    var_15 = [var_10, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = repr(var_16)
    assert var_17 == "BaseError([Message(text='Error 1', code='error1', index=['field1']), Message(text='Error 2', code='error2', index=['field2'])]))"
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.BaseError(text=var_0, code=var_1, position=var_21)
    var_23 = repr(var_22)
    assert var_23 == "BaseError([Message(text='Error message', code='error_code', position=Position(line_no=1, column_no=2, char_index=3))]))"



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = 8
    var_30 = module_0.Position(var_18, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_33 = module_0.Position(var_18, var_20, var_24)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 6
    var_36 = 9
    var_37 = module_0.Position(var_18, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_37)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = module_0.Position(var_18, var_19, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_24)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_24)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_24, end_position=var_24)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 'c'
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = 4
    var_27 = module_0.Position(var_20, var_22, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)
    var_29 = module_0.Position(var_20, var_21, var_22)
    var_30 = 5
    var_31 = module_0.Position(var_20, var_26, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_31)
    var_34 = module_0.Position(var_21, var_21, var_22)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = module_0.Position(var_20, var_30, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_37)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_4 = var_3._messages
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'Message 1'
    var_7 = 'code1'
    var_8 = 'key1'
    var_9 = module_0.Message(text=var_6, code=var_7, key=var_8)
    var_10 = 'Message 2'
    var_11 = 'code2'
    var_12 = 'key2'
    var_13 = [var_12]
    var_14 = module_0.Message(text=var_10, code=var_11, index=var_13)
    var_15 = [var_9, var_14]
    var_16 = module_0.ValidationError(messages=var_15)
    var_17 = var_16._messages
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = dict(var_16)
    var_20 = module_0.ValidationError(messages=var_15)
    var_21 = hash(var_16)
    var_22 = hash(var_20)
    var_23 = 'Simple error'
    var_24 = module_0.ValidationError(text=var_23)
    var_25 = repr(var_24)
    assert var_25 == "ValidationError(text='Simple error', code='custom')"
    var_26 = str(var_24)
    assert var_26 == 'Simple error'



# Parsed testcases at query #10
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
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = module_0.Position(var_6, var_7, var_8)
    var_10 = 'Error with position'
    var_11 = module_0.BaseError(text=var_10, position=var_9)
    var_12 = 'Message 1'
    var_13 = 'code1'
    var_14 = 'key1'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = 'Message 2'
    var_17 = 'code2'
    var_18 = 'nested'
    var_19 = 'key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Message(text=var_16, code=var_17, index=var_20)
    var_22 = [var_15, var_21]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = var_23._messages
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_23.messages()
    var_27 = 'prefix'
    var_28 = var_23.messages(add_prefix=var_27)
    var_29 = dict(var_3)
    var_30 = list(var_3)
    var_31 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_32 = str(var_3)
    assert var_32 == 'Error message'
    var_33 = repr(var_3)
    assert var_33 == "BaseError(text='Error message', code='error_code')"



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'nested'
    var_12 = [var_2, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_20 = 4
    var_21 = module_0.Position(var_15, var_16, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Position(var_14, var_15, var_16)
    var_24 = 5
    var_25 = 8
    var_26 = module_0.Position(var_14, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_26)
    var_29 = module_0.Position(var_15, var_16, var_20)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_26)
    var_31 = 6
    var_32 = 9
    var_33 = module_0.Position(var_14, var_31, var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_33)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = module_0.Position(var_15, var_16, var_17)
    var_22 = 5
    var_23 = 8
    var_24 = module_0.Position(var_15, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_24)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'other'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_7)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = 'c'
    var_17 = [var_10, var_16]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_19, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)
    var_11 = next(var_8)
    assert var_11 is None
    var_12 = next(var_8)
    assert var_12 is None



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)



# Parsed testcases at query #3
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
    var_6 = dict(var_3)
    var_7 = 'Error 1'
    var_8 = 'code1'
    var_9 = 'key1'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    var_11 = 'Error 2'
    var_12 = 'code2'
    var_13 = 'key2'
    var_14 = module_0.Message(text=var_11, code=var_12, key=var_13)
    var_15 = [var_10, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = var_16._messages
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = dict(var_16)
    var_20 = 'users'
    var_21 = 0
    var_22 = 'username'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.Message(text=var_7, code=var_8, index=var_23)
    var_25 = 1
    var_26 = 'email'
    var_27 = [var_20, var_25, var_26]
    var_28 = module_0.Message(text=var_11, code=var_12, index=var_27)
    var_29 = [var_24, var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = var_30._messages
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = dict(var_30)
    var_34 = 2
    var_35 = 3
    var_36 = module_0.Position(var_25, var_34, var_35)
    var_37 = 'Error with position'
    var_38 = module_0.BaseError(text=var_37, position=var_36)
    var_39 = module_0.Position(var_25, var_34, var_35)
    var_40 = 5
    var_41 = 8
    var_42 = module_0.Position(var_25, var_40, var_41)
    var_43 = 'Error with positions'
    var_44 = module_0.BaseError(text=var_43)
    var_45 = 'Error without code'
    var_46 = module_0.BaseError(text=var_45)
    var_47 = 'Error'
    var_48 = 'code'
    var_49 = module_0.BaseError(text=var_47, code=var_48)
    var_50 = module_0.BaseError(text=var_47, code=var_48)
    var_51 = module_0.BaseError(text=var_47, code=var_48)
    var_52 = module_0.BaseError(text=var_47, code=var_48)
    var_53 = hash(var_51)
    var_54 = hash(var_52)
    var_55 = module_0.BaseError(text=var_47, code=var_48)
    var_56 = repr(var_55)
    assert var_56 == "BaseError(text='Error', code='code')"
    var_57 = module_0.BaseError(text=var_47, code=var_48)
    var_58 = str(var_57)
    assert var_58 == 'Error'



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = 'test_code'
    var_7 = module_0.ValidationError(text=var_5, code=var_6)
    var_8 = module_0.ValidationResult(error=var_7)
    var_9 = iter(var_8)
    var_10 = next(var_9)
    assert var_10 is None
    var_11 = next(var_9)
    var_12 = next(var_9)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = module_0.Position(var_13, var_14, var_15)
    var_17 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_19 = 4
    var_20 = module_0.Position(var_14, var_15, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Position(var_13, var_14, var_15)
    var_23 = 5
    var_24 = 8
    var_25 = module_0.Position(var_13, var_23, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_25)
    var_28 = module_0.Position(var_14, var_15, var_19)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_25)
    var_30 = 6
    var_31 = 9
    var_32 = module_0.Position(var_14, var_30, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_32)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_12, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_12, var_13, var_17)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_12, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_30)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_18, var_19)
    var_29 = module_0.Position(var_23, var_24, var_25)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_29)
    var_32 = 7
    var_33 = 8
    var_34 = 9
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_29)
    var_37 = 10
    var_38 = 11
    var_39 = 12
    var_40 = module_0.Position(var_37, var_38, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_40)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_17, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_11 = 'other_field'
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_11)
    var_13 = 'list'
    var_14 = 0
    var_15 = [var_13, var_14]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = [var_13, var_14]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 1
    var_20 = [var_13, var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 2
    var_23 = 3
    var_24 = module_0.Position(var_19, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_27 = 4
    var_28 = module_0.Position(var_22, var_23, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_28)
    var_30 = module_0.Position(var_19, var_22, var_23)
    var_31 = 5
    var_32 = 8
    var_33 = module_0.Position(var_19, var_31, var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_33)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_33)
    var_36 = module_0.Position(var_22, var_23, var_27)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_36, end_position=var_33)
    var_38 = 6
    var_39 = 9
    var_40 = module_0.Position(var_22, var_38, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_40)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different error'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_19, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = 8
    var_30 = module_0.Position(var_18, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_33 = module_0.Position(var_19, var_20, var_24)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 6
    var_36 = 9
    var_37 = module_0.Position(var_18, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_37)



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_20, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different error'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_17, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, position=var_20)
    var_22 = module_0.Message(text=var_0, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, position=var_26)
    var_28 = module_0.Position(var_17, var_18, var_19)
    var_29 = module_0.Position(var_23, var_24, var_25)
    var_30 = module_0.Message(text=var_0, start_position=var_28, end_position=var_29)
    var_31 = module_0.Message(text=var_0, start_position=var_28, end_position=var_29)
    var_32 = 7
    var_33 = 8
    var_34 = 9
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = module_0.Message(text=var_0, start_position=var_35, end_position=var_29)
    var_37 = 10
    var_38 = 11
    var_39 = 12
    var_40 = module_0.Position(var_37, var_38, var_39)
    var_41 = module_0.Message(text=var_0, start_position=var_28, end_position=var_40)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = module_0.Position(var_13, var_14, var_15)
    var_17 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_19 = 4
    var_20 = module_0.Position(var_14, var_15, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Position(var_13, var_14, var_15)
    var_23 = 5
    var_24 = 8
    var_25 = module_0.Position(var_13, var_23, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_25)
    var_28 = module_0.Position(var_14, var_15, var_19)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_25)
    var_30 = 6
    var_31 = 9
    var_32 = module_0.Position(var_13, var_30, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_32)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_12, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_12, var_12, var_13)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_11, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_30)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_20, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_20)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_20)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_14)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, position=var_14)
    var_16 = module_0.Message(text=var_0, position=var_14)
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, position=var_20)
    var_22 = module_0.Position(var_11, var_12, var_13)
    var_23 = 8
    var_24 = module_0.Position(var_11, var_18, var_23)
    var_25 = module_0.Message(text=var_0, start_position=var_22, end_position=var_24)
    var_26 = module_0.Message(text=var_0, start_position=var_22, end_position=var_24)
    var_27 = module_0.Position(var_12, var_13, var_17)
    var_28 = module_0.Message(text=var_0, start_position=var_27, end_position=var_24)
    var_29 = 9
    var_30 = module_0.Position(var_11, var_19, var_29)
    var_31 = module_0.Message(text=var_0, start_position=var_22, end_position=var_30)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 'c'
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = 4
    var_27 = module_0.Position(var_20, var_22, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_27)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_27)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_27)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_23)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 'c'
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = 4
    var_27 = module_0.Position(var_20, var_22, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)
    var_29 = module_0.Position(var_20, var_21, var_22)
    var_30 = 5
    var_31 = 8
    var_32 = module_0.Position(var_20, var_30, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_32)
    var_35 = module_0.Position(var_21, var_21, var_22)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_32)
    var_37 = 6
    var_38 = 9
    var_39 = module_0.Position(var_20, var_37, var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_39)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = 5
    var_23 = 6
    var_24 = module_0.Position(var_21, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_24)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_24)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_24, end_position=var_24)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Position(var_11, var_12, var_13)
    var_23 = module_0.Position(var_17, var_18, var_19)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_23)
    var_26 = 7
    var_27 = 8
    var_28 = 9
    var_29 = module_0.Position(var_26, var_27, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_23)
    var_31 = 10
    var_32 = 11
    var_33 = 12
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_34)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'other'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_7)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = 'c'
    var_17 = [var_10, var_16]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_19, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = 'Error'
    var_16 = module_0.Message(text=var_15, position=var_14)
    var_17 = module_0.Message(text=var_15, position=var_14)
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_15, position=var_21)
    var_23 = module_0.Position(var_11, var_12, var_13)
    var_24 = 8
    var_25 = module_0.Position(var_11, var_19, var_24)
    var_26 = module_0.Message(text=var_15, start_position=var_23, end_position=var_25)
    var_27 = module_0.Message(text=var_15, start_position=var_23, end_position=var_25)
    var_28 = module_0.Position(var_12, var_13, var_18)
    var_29 = module_0.Message(text=var_15, start_position=var_28, end_position=var_25)
    var_30 = 9
    var_31 = module_0.Position(var_11, var_20, var_30)
    var_32 = module_0.Message(text=var_15, start_position=var_23, end_position=var_31)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = 10
    var_30 = module_0.Position(var_18, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_33 = module_0.Position(var_18, var_20, var_24)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 6
    var_36 = 11
    var_37 = module_0.Position(var_18, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_37)



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 3
    var_13 = [var_11, var_12, var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12, var_2]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 4
    var_18 = [var_11, var_17, var_2]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = module_0.Position(var_20, var_21, var_12)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = module_0.Position(var_21, var_12, var_17)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_20, var_21, var_12)
    var_28 = 5
    var_29 = 8
    var_30 = module_0.Position(var_20, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_33 = module_0.Position(var_21, var_12, var_17)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 6
    var_36 = 9
    var_37 = module_0.Position(var_20, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_37)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = 5
    var_23 = 6
    var_24 = module_0.Position(var_21, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_24)



# Parsed testcases at query #36
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_11 = 'other_field'
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_11)
    var_13 = 'list'
    var_14 = 0
    var_15 = [var_13, var_14]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = [var_13, var_14]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 1
    var_20 = [var_13, var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 2
    var_23 = 3
    var_24 = module_0.Position(var_19, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_27 = 4
    var_28 = 5
    var_29 = 6
    var_30 = module_0.Position(var_27, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_24, end_position=var_30)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_24, end_position=var_30)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_30)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_24, end_position=var_24)



# Parsed testcases at query #37
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_19, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = 10
    var_30 = module_0.Position(var_18, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_30)
    var_33 = module_0.Position(var_18, var_19, var_24)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 11
    var_36 = module_0.Position(var_18, var_28, var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_36)



# Parsed testcases at query #38
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #39
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = 'Error'
    var_16 = module_0.Message(text=var_15, position=var_14)
    var_17 = module_0.Message(text=var_15, position=var_14)
    var_18 = 4
    var_19 = module_0.Position(var_12, var_13, var_18)
    var_20 = module_0.Message(text=var_15, position=var_19)
    var_21 = 'a'
    var_22 = 'b'
    var_23 = [var_21, var_22]
    var_24 = module_0.Message(text=var_15, index=var_23)
    var_25 = [var_21, var_22]
    var_26 = module_0.Message(text=var_15, index=var_25)
    var_27 = 'c'
    var_28 = [var_21, var_27]
    var_29 = module_0.Message(text=var_15, index=var_28)
    var_30 = module_0.Position(var_11, var_11, var_11)
    var_31 = 5
    var_32 = module_0.Position(var_11, var_31, var_31)
    var_33 = module_0.Message(text=var_15, start_position=var_30, end_position=var_32)
    var_34 = module_0.Message(text=var_15, start_position=var_30, end_position=var_32)
    var_35 = 6
    var_36 = module_0.Position(var_12, var_11, var_35)
    var_37 = module_0.Message(text=var_15, start_position=var_36, end_position=var_32)
    var_38 = module_0.Position(var_11, var_35, var_35)
    var_39 = module_0.Message(text=var_15, start_position=var_30, end_position=var_38)



# Parsed testcases at query #40
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = 5
    var_27 = 6
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_28)
    var_30 = module_0.Position(var_17, var_20, var_21)
    var_31 = 8
    var_32 = module_0.Position(var_17, var_26, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_32)
    var_35 = module_0.Position(var_20, var_21, var_25)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_32)
    var_37 = 9
    var_38 = module_0.Position(var_17, var_27, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_38)



# Parsed testcases at query #41
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error text'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different text'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_21)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)



# Parsed testcases at query #42
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_11, var_12, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_11, var_12, var_17)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 9
    var_29 = module_0.Position(var_11, var_21, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_29)



# Parsed testcases at query #43
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #44
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_25)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)



# Parsed testcases at query #45
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'other_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)



# Parsed testcases at query #46
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12, var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12, var_2]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17, var_2]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_20, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #47
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_26)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_26)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)



# Parsed testcases at query #48
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = 'Error'
    var_16 = module_0.Message(text=var_15, position=var_14)
    var_17 = module_0.Message(text=var_15, position=var_14)
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_15, position=var_21)
    var_23 = module_0.Message(text=var_15, start_position=var_14, end_position=var_21)
    var_24 = module_0.Message(text=var_15, start_position=var_14, end_position=var_21)
    var_25 = module_0.Message(text=var_15, start_position=var_21, end_position=var_21)
    var_26 = module_0.Message(text=var_15, start_position=var_14, end_position=var_14)



# Parsed testcases at query #49
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, position=var_14)
    var_16 = module_0.Message(text=var_0, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_11, var_12, var_17)
    var_19 = module_0.Message(text=var_0, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 10
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_11, var_13, var_13)
    var_27 = module_0.Message(text=var_0, start_position=var_26, end_position=var_23)
    var_28 = 11
    var_29 = module_0.Position(var_11, var_21, var_28)
    var_30 = module_0.Message(text=var_0, start_position=var_20, end_position=var_29)



# Parsed testcases at query #50
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'email'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = module_0.Position(var_16, var_17, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)



# Parsed testcases at query #51
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = module_0.Position(var_18, var_24, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_32 = module_0.Position(var_18, var_20, var_24)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_29)
    var_34 = 6
    var_35 = module_0.Position(var_18, var_28, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_35)



# Parsed testcases at query #52
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_11 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = module_0.Position(var_12, var_13, var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_17 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Position(var_12, var_13, var_14)
    var_24 = module_0.Position(var_18, var_19, var_20)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_24)
    var_27 = 7
    var_28 = 8
    var_29 = 9
    var_30 = module_0.Position(var_27, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_24)
    var_32 = 10
    var_33 = 11
    var_34 = 12
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_35)



# Parsed testcases at query #53
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_14)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_14)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_20)
    var_25 = 'users'
    var_26 = 0
    var_27 = 'name'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.Message(text=var_0, code=var_1, index=var_28)
    var_30 = [var_25, var_26, var_27]
    var_31 = module_0.Message(text=var_0, code=var_1, index=var_30)
    var_32 = [var_25, var_11, var_27]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)



# Parsed testcases at query #54
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_17, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #55
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 0
    var_12 = [var_2, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = 4
    var_19 = 5
    var_20 = 6
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_17, end_position=var_21)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_17)



# Parsed testcases at query #56
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_12, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_12, var_13, var_17)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_11, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_30)



# Parsed testcases at query #57
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_22)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)



# Parsed testcases at query #58
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = [var_8]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 'other_field'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = module_0.Position(var_16, var_17, var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_19)
    var_22 = 4
    var_23 = module_0.Position(var_17, var_18, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Position(var_16, var_17, var_18)
    var_26 = 5
    var_27 = 8
    var_28 = module_0.Position(var_16, var_26, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_28)
    var_31 = module_0.Position(var_17, var_18, var_22)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_31, end_position=var_28)
    var_33 = 6
    var_34 = 9
    var_35 = module_0.Position(var_16, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_35)



# Parsed testcases at query #59
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = module_0.Position(var_15, var_17, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #60
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'other'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_7)
    var_10 = [var_2]
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_10)
    var_12 = [var_2]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_20 = module_0.Position(var_14, var_15, var_16)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_14, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_15, var_15, var_16)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)



# Parsed testcases at query #61
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_12, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_12, var_13, var_17)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_11, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_30)



# Parsed testcases at query #62
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_12, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_12, var_13, var_17)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_11, var_28, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_30)



# Parsed testcases at query #63
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = module_0.Position(var_17, var_18, var_19)
    var_24 = 5
    var_25 = 8
    var_26 = module_0.Position(var_17, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_26)



# Parsed testcases at query #64
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 0
    var_12 = [var_2, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_20 = 4
    var_21 = module_0.Position(var_15, var_16, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)



# Parsed testcases at query #65
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = [var_11, var_12, var_13]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 4
    var_19 = [var_11, var_12, var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = module_0.Position(var_11, var_12, var_13)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = module_0.Position(var_11, var_12, var_18)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Position(var_11, var_12, var_13)
    var_27 = 5
    var_28 = 10
    var_29 = module_0.Position(var_11, var_27, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_29)
    var_32 = module_0.Position(var_11, var_13, var_13)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_29)
    var_34 = 11
    var_35 = module_0.Position(var_11, var_27, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_35)



# Parsed testcases at query #66
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = module_0.Position(var_16, var_17, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_18)



# Parsed testcases at query #67
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, position=var_14)
    var_16 = module_0.Message(text=var_0, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_11, var_13, var_17)
    var_19 = module_0.Message(text=var_0, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 8
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_11, var_13, var_17)
    var_27 = module_0.Message(text=var_0, start_position=var_26, end_position=var_23)
    var_28 = 6
    var_29 = 9
    var_30 = module_0.Position(var_11, var_28, var_29)
    var_31 = module_0.Message(text=var_0, start_position=var_20, end_position=var_30)



# Parsed testcases at query #68
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Position(var_18, var_19, var_20)
    var_28 = 5
    var_29 = module_0.Position(var_18, var_24, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_32 = module_0.Position(var_19, var_19, var_20)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_29)
    var_34 = 6
    var_35 = module_0.Position(var_18, var_28, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_35)



# Parsed testcases at query #69
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_20, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_21)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)



# Parsed testcases at query #70
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 4
    var_22 = module_0.Position(var_15, var_17, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #71
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_20, var_21)
    var_29 = 5
    var_30 = 8
    var_31 = module_0.Position(var_17, var_29, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_31)
    var_34 = module_0.Position(var_20, var_21, var_25)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_34, end_position=var_31)
    var_36 = 6
    var_37 = 9
    var_38 = module_0.Position(var_20, var_36, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_38)



# Parsed testcases at query #72
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2, var_9]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2, var_9]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 'key3'
    var_16 = [var_2, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_21)
    var_24 = 4
    var_25 = module_0.Position(var_18, var_19, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_25)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)



# Parsed testcases at query #73
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_9]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_23 = 4
    var_24 = 5
    var_25 = 6
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Position(var_17, var_18, var_19)
    var_29 = module_0.Position(var_23, var_24, var_25)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_29)
    var_32 = 7
    var_33 = 8
    var_34 = 9
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_29)
    var_37 = 10
    var_38 = 11
    var_39 = 12
    var_40 = module_0.Position(var_37, var_38, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_40)



# Parsed testcases at query #74
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = module_0.Message(text=var_0, position=var_14)
    var_16 = module_0.Message(text=var_0, position=var_14)
    var_17 = 4
    var_18 = module_0.Position(var_11, var_12, var_17)
    var_19 = module_0.Message(text=var_0, position=var_18)
    var_20 = module_0.Position(var_11, var_12, var_13)
    var_21 = 5
    var_22 = 10
    var_23 = module_0.Position(var_11, var_21, var_22)
    var_24 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_25 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_26 = module_0.Position(var_11, var_13, var_13)
    var_27 = module_0.Message(text=var_0, start_position=var_26, end_position=var_23)
    var_28 = 11
    var_29 = module_0.Position(var_11, var_21, var_28)
    var_30 = module_0.Message(text=var_0, start_position=var_20, end_position=var_29)



# Parsed testcases at query #75
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = module_0.Position(var_15, var_16, var_17)
    var_22 = 5
    var_23 = 8
    var_24 = module_0.Position(var_15, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_21)



# Parsed testcases at query #76
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'list'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_17, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_25 = 4
    var_26 = module_0.Position(var_20, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_26)



# Parsed testcases at query #77
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different error'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_18)
    var_21 = 5
    var_22 = 10
    var_23 = module_0.Position(var_15, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_23)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_23)



# Parsed testcases at query #78
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 'c'
    var_18 = [var_11, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = 4
    var_27 = module_0.Position(var_21, var_22, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_27)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_27)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_27)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_23)



