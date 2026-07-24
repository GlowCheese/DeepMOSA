####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    assert var_3 is None
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = None
    var_6 = module_0.ValidationResult(value=var_5, error=var_5)



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
    var_11 = module_0.ValidationResult()
    var_12 = iter(var_11)
    var_13 = next(var_12)
    assert var_13 is None
    var_14 = next(var_12)
    assert var_14 is None



# Parsed testcases at query #3
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
    var_18 = 'Error'
    var_19 = module_0.Message(text=var_18, position=var_17)
    var_20 = module_0.Message(text=var_18, position=var_17)
    var_21 = 4
    var_22 = module_0.Position(var_14, var_16, var_21)
    var_23 = module_0.Message(text=var_18, position=var_22)
    var_24 = module_0.Position(var_14, var_15, var_16)
    var_25 = 5
    var_26 = 8
    var_27 = module_0.Position(var_14, var_25, var_26)
    var_28 = module_0.Message(text=var_18, start_position=var_24, end_position=var_27)
    var_29 = module_0.Message(text=var_18, start_position=var_24, end_position=var_27)
    var_30 = module_0.Position(var_14, var_16, var_21)
    var_31 = module_0.Message(text=var_18, start_position=var_30, end_position=var_27)
    var_32 = 6
    var_33 = 9
    var_34 = module_0.Position(var_14, var_32, var_33)
    var_35 = module_0.Message(text=var_18, start_position=var_24, end_position=var_34)



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
    var_11 = None
    var_12 = module_0.ValidationResult(value=var_11, error=var_11)
    var_13 = iter(var_12)
    var_14 = next(var_13)
    assert var_14 is None
    var_15 = next(var_13)
    assert var_15 is None



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
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationResult(error=var_2)
    var_4 = repr(var_3)
    assert var_4 == "ValidationResult(error=ValidationError(text='Error message', code='error_code'))"
    var_5 = 'valid_value'
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(value='valid_value')"



# Parsed testcases at query #7
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
    var_26 = 5
    var_27 = 6
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_28)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_28)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_28)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



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



# Parsed testcases at query #9
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
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_25)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_25)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_21)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationResult(error=var_2)
    var_4 = repr(var_3)
    assert var_4 == "ValidationResult(error=ValidationError(text='Test error', code='test_code'))"
    var_5 = 'Test value'
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(value='Test value')"



# Parsed testcases at query #11
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
    assert var_17 == "BaseError([Message(text='Error 1', code='code1', index=['field1']), Message(text='Error 2', code='code2', index=['field2'])]))"
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.BaseError(text=var_0, code=var_1, position=var_21)
    var_23 = repr(var_22)
    assert var_23 == "BaseError([Message(text='Error message', code='error_code', position=Position(line_no=1, column_no=2, char_index=3))]))"



# Parsed testcases at query #12
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
    assert var_17 == "BaseError([Message(text='Error 1', code='code1', index=['field1']), Message(text='Error 2', code='code2', index=['field2'])]))"
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = 'Error with position'
    var_23 = 'pos_error'
    var_24 = module_0.BaseError(text=var_22, code=var_23, position=var_21)
    var_25 = repr(var_24)
    assert var_25 == "BaseError([Message(text='Error with position', code='pos_error', index=[], position=Position(line_no=1, column_no=2, char_index=3))]))"



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_23)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_23)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_23)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_27)



# Parsed testcases at query #15
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



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = dict(var_3)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = module_0.Position(var_6, var_7, var_8)
    var_10 = 'Error with position'
    var_11 = module_0.BaseError(text=var_10, position=var_9)
    var_12 = 'First error'
    var_13 = 'first'
    var_14 = module_0.Message(text=var_12, code=var_13)
    var_15 = 'Second error'
    var_16 = 'second'
    var_17 = module_0.Message(text=var_15, code=var_16)
    var_18 = [var_14, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = dict(var_19)
    var_22 = 'Nested error'
    var_23 = 'nested'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = [var_24, var_25]
    var_27 = module_0.Message(text=var_22, code=var_23, index=var_26)
    var_28 = [var_27]
    var_29 = module_0.BaseError(messages=var_28)
    var_30 = dict(var_29)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'Message 1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    var_9 = 'Message 2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    var_13 = [var_8, var_12]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'Nested error'
    var_17 = 'parent'
    var_18 = module_0.ValidationError(text=var_16, key=var_17)
    var_19 = 'Child error'
    var_20 = 'child'
    var_21 = [var_17, var_20]
    var_22 = module_0.Message(text=var_19, index=var_21)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = 'Test message'
    var_26 = 'test'
    var_27 = module_0.ValidationError(text=var_25, key=var_26)
    var_28 = 'custom'
    var_29 = [var_26]
    var_30 = module_0.Message(text=var_25, code=var_28, index=var_29)
    var_31 = [var_30]
    var_32 = module_0.ValidationError(text=var_25, key=var_26)
    var_33 = 'prefix'
    var_34 = [var_33, var_26]
    var_35 = module_0.Message(text=var_25, code=var_28, index=var_34)
    var_36 = [var_35]
    var_37 = 'Same'
    var_38 = 'key'
    var_39 = module_0.ValidationError(text=var_37, key=var_38)
    var_40 = module_0.ValidationError(text=var_37, key=var_38)
    var_41 = 'Hash test'
    var_42 = module_0.ValidationError(text=var_41, key=var_38)
    var_43 = module_0.ValidationError(text=var_41, key=var_38)
    var_44 = hash(var_42)
    var_45 = hash(var_43)
    var_46 = 'Repr test'
    var_47 = module_0.ValidationError(text=var_46)
    var_48 = repr(var_47)
    assert var_48 == "ValidationError(text='Repr test', code='custom')"
    var_49 = 'Msg1'
    var_50 = 'k1'
    var_51 = module_0.Message(text=var_49, key=var_50)
    var_52 = 'Msg2'
    var_53 = 'k2'
    var_54 = module_0.Message(text=var_52, key=var_53)
    var_55 = [var_51, var_54]
    var_56 = module_0.ValidationError(messages=var_55)
    var_57 = repr(var_56)
    var_58 = 'String test'
    var_59 = module_0.ValidationError(text=var_58)
    var_60 = str(var_59)
    assert var_60 == 'String test'
    var_61 = module_0.Message(text=var_49, key=var_50)
    var_62 = module_0.Message(text=var_52, key=var_53)
    var_63 = [var_61, var_62]
    var_64 = module_0.ValidationError(messages=var_63)
    var_65 = str(var_64)
    var_66 = {var_50: var_49, var_53: var_52}
    var_67 = str(var_66)
    var_68 = 'Dict test'
    var_69 = 'dict_key'
    var_70 = module_0.ValidationError(text=var_68, key=var_69)
    var_71 = dict(var_70)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = module_0.Position(var_1, var_1, var_2)
    var_6 = module_0.Position(var_0, var_2, var_2)
    var_7 = 4
    var_8 = module_0.Position(var_0, var_1, var_7)



# Parsed testcases at query #22
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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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
    var_24 = module_0.Position(var_18, var_19, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_24)
    var_26 = module_0.Position(var_17, var_18, var_19)
    var_27 = 5
    var_28 = 8
    var_29 = module_0.Position(var_17, var_27, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_29)
    var_32 = module_0.Position(var_18, var_19, var_23)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_29)
    var_34 = 6
    var_35 = 9
    var_36 = module_0.Position(var_17, var_34, var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_36)



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
    var_21 = module_0.Position(var_15, var_16, var_17)
    var_22 = 5
    var_23 = 8
    var_24 = module_0.Position(var_15, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_27 = 4
    var_28 = module_0.Position(var_16, var_17, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_28)



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



# Parsed testcases at query #28
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
    var_21 = module_0.Position(var_15, var_16, var_17)
    var_22 = 5
    var_23 = 8
    var_24 = module_0.Position(var_15, var_22, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_21, end_position=var_24)
    var_27 = module_0.Position(var_16, var_16, var_17)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)



# Parsed testcases at query #29
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
    var_17 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_16)
    var_19 = 4
    var_20 = module_0.Position(var_14, var_15, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_20)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_16, end_position=var_20)
    var_23 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_16, end_position=var_20)
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_20, end_position=var_20)
    var_25 = module_0.Message(text=var_0)
    var_26 = module_0.Message(text=var_0)
    var_27 = []
    var_28 = module_0.Message(text=var_0, index=var_27)
    var_29 = module_0.Message(text=var_0)



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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
    var_13 = 'name'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = [var_11, var_12, var_13]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = [var_11, var_18, var_13]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_18, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = 4
    var_27 = 5
    var_28 = 6
    var_29 = module_0.Position(var_26, var_27, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_29)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_29)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_29)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_23, end_position=var_23)



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
    var_29 = module_0.Position(var_18, var_24, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_29)
    var_32 = module_0.Position(var_18, var_20, var_24)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_29)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_32)



# Parsed testcases at query #34
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
    var_21 = 4
    var_22 = module_0.Position(var_15, var_16, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_18, end_position=var_22)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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
    var_25 = module_0.Message(text=var_15, start_position=var_21, end_position=var_14)
    var_26 = 0
    var_27 = [var_2, var_26]
    var_28 = module_0.Message(text=var_15, index=var_27)
    var_29 = [var_2, var_26]
    var_30 = module_0.Message(text=var_15, index=var_29)
    var_31 = [var_2, var_11]
    var_32 = module_0.Message(text=var_15, index=var_31)



# Parsed testcases at query #37
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
    var_26 = 5
    var_27 = 6
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_28)
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_28)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_28, end_position=var_28)



# Parsed testcases at query #38
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
    var_37 = module_0.Position(var_19, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_37)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'Error 1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    var_9 = 'Error 2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    var_13 = [var_8, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'Nested error'
    var_17 = 'parent'
    var_18 = module_0.BaseError(text=var_16, key=var_17)
    var_19 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_20 = repr(var_3)
    assert var_20 == "BaseError(text='Error message', code='error_code')"
    var_21 = repr(var_14)
    var_22 = str(var_3)
    assert var_22 == 'Error message'
    var_23 = str(var_14)
    var_24 = dict(var_14)
    var_25 = str(var_24)
    var_26 = var_3.messages()
    var_27 = 0
    var_28 = 'prefix'
    var_29 = error1.messages(add_prefix=var_28)[var_27]
    var_30 = var_29.index



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = 'Error 2'
    var_9 = 'err2'
    var_10 = module_0.Message(text=var_8, code=var_9)
    var_11 = [var_10]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = module_0.Message(text=var_0, code=var_1)
    var_14 = module_0.Message(text=var_8, code=var_9)
    var_15 = [var_13, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = module_0.BaseError(text=var_0, code=var_1)
    var_18 = module_0.BaseError(text=var_0, code=var_1)
    var_19 = module_0.BaseError(text=var_8, code=var_9)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_25 = [var_24]
    var_26 = module_0.BaseError(messages=var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_28 = [var_27]
    var_29 = module_0.BaseError(messages=var_28)
    var_30 = 4
    var_31 = 5
    var_32 = 6
    var_33 = module_0.Position(var_30, var_31, var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, position=var_33)
    var_35 = [var_34]
    var_36 = module_0.BaseError(messages=var_35)



# Parsed testcases at query #3
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
    var_12 = module_0.ValidationResult()
    var_13 = iter(var_12)
    var_14 = next(var_13)
    assert var_14 is None
    var_15 = next(var_13)
    assert var_15 is None



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    assert var_3 is None
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = module_0.ValidationResult()



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



# Parsed testcases at query #7
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
    var_10 = 'Error 3'
    var_11 = module_0.Message(text=var_10)
    var_12 = [var_11]
    var_13 = module_0.BaseError(messages=var_12)
    var_14 = 'Single error'
    var_15 = 'error_code'
    var_16 = module_0.BaseError(text=var_14, code=var_15)
    var_17 = module_0.BaseError(text=var_14, code=var_15)
    var_18 = 'Different error'
    var_19 = module_0.BaseError(text=var_18)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = module_0.Position(var_20, var_21, var_22)
    var_24 = 'Error with pos'
    var_25 = module_0.Message(text=var_24, position=var_23)
    var_26 = [var_25]
    var_27 = module_0.BaseError(messages=var_26)
    var_28 = module_0.Message(text=var_24, position=var_23)
    var_29 = [var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = 4
    var_32 = 5
    var_33 = 6
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = module_0.Message(text=var_24, position=var_34)
    var_36 = [var_35]
    var_37 = module_0.BaseError(messages=var_36)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='Error message', code='custom')"
    var_3 = 'max_length'
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, code=var_3, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "Message(text='Error message', code='max_length', index=['username'])"
    var_7 = 'users'
    var_8 = 3
    var_9 = [var_7, var_8, var_4]
    var_10 = module_0.Message(text=var_0, index=var_9)
    var_11 = repr(var_10)
    assert var_11 == "Message(text='Error message', code='custom', index=['users', 3, 'username'])"
    var_12 = 1
    var_13 = 2
    var_14 = module_0.Position(var_12, var_13, var_8)
    var_15 = module_0.Message(text=var_0, position=var_14)
    var_16 = repr(var_15)
    assert var_16 == "Message(text='Error message', code='custom', position=Position(line_no=1, column_no=2, char_index=3))"
    var_17 = module_0.Position(var_12, var_13, var_8)
    var_18 = 5
    var_19 = 8
    var_20 = module_0.Position(var_12, var_18, var_19)
    var_21 = module_0.Message(text=var_0, start_position=var_17, end_position=var_20)
    var_22 = "Message(text='Error message', code='custom', start_position=Position(line_no=1, column_no=2, char_index=3), end_position=Position(line_no=1, column_no=5, char_index=8))"
    var_23 = repr(var_21)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.BaseError(messages=var_3)
    var_5 = module_0.Message(text=var_0, code=var_1)
    var_6 = [var_5]
    var_7 = module_0.BaseError(messages=var_6)
    var_8 = 'Error 2'
    var_9 = module_0.Message(text=var_8, code=var_1)
    var_10 = [var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = module_0.Message(text=var_0, code=var_1)
    var_13 = module_0.Message(text=var_8, code=var_1)
    var_14 = [var_12, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    var_16 = module_0.BaseError(text=var_0, code=var_1)
    var_17 = module_0.BaseError(text=var_0, code=var_1)
    var_18 = module_0.BaseError(text=var_8, code=var_1)
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = module_0.Position(var_19, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = [var_23]
    var_25 = module_0.BaseError(messages=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_27 = [var_26]
    var_28 = module_0.BaseError(messages=var_27)
    var_29 = 4
    var_30 = module_0.Position(var_20, var_21, var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, position=var_30)
    var_32 = [var_31]
    var_33 = module_0.BaseError(messages=var_32)



# Parsed testcases at query #11
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
    var_8 = 'Error 2'
    var_9 = 'error2'
    var_10 = module_0.Message(text=var_8, code=var_9)
    var_11 = [var_10]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = module_0.Message(text=var_0, code=var_1)
    var_14 = module_0.Message(text=var_8, code=var_9)
    var_15 = [var_13, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = module_0.BaseError(text=var_0, code=var_1)
    var_18 = module_0.BaseError(text=var_0, code=var_1)
    var_19 = module_0.BaseError(text=var_8, code=var_9)



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
    var_31 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)



# Parsed testcases at query #13
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
    var_8 = 'Error 2'
    var_9 = 'error2'
    var_10 = module_0.Message(text=var_8, code=var_9)
    var_11 = [var_10]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = module_0.Message(text=var_0, code=var_1)
    var_14 = module_0.Message(text=var_8, code=var_9)
    var_15 = [var_13, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = module_0.BaseError(text=var_0, code=var_1)
    var_18 = module_0.BaseError(text=var_0, code=var_1)
    var_19 = module_0.BaseError(text=var_8, code=var_9)



# Parsed testcases at query #14
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
    var_20 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_14)
    var_21 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_14)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_18)



# Parsed testcases at query #15
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
    var_10 = module_0.Message(text=var_0)
    var_11 = 'Error 3'
    var_12 = module_0.Message(text=var_11)
    var_13 = [var_10, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = module_0.Message(text=var_0)
    var_16 = [var_15]
    var_17 = module_0.BaseError(messages=var_16)
    var_18 = 'Single error'
    var_19 = 'error_code'
    var_20 = module_0.BaseError(text=var_18, code=var_19)
    var_21 = module_0.BaseError(text=var_18, code=var_19)
    var_22 = 'Different error'
    var_23 = module_0.BaseError(text=var_22, code=var_19)
    var_24 = module_0.Message(text=var_18, code=var_19)
    var_25 = [var_24]
    var_26 = module_0.BaseError(messages=var_25)



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



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'Error 1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    var_9 = 'Error 2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    var_13 = [var_8, var_12]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'Nested error'
    var_17 = 'parent'
    var_18 = module_0.ValidationError(text=var_16, key=var_17)
    var_19 = 'child'
    var_20 = [var_17, var_19]
    var_21 = module_0.Message(text=var_16, index=var_20)
    var_22 = [var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = module_0.Position(var_24, var_25, var_26)
    var_28 = 'Position error'
    var_29 = module_0.ValidationError(text=var_28, position=var_27)
    var_30 = module_0.Position(var_24, var_25, var_26)
    var_31 = 5
    var_32 = 8
    var_33 = module_0.Position(var_24, var_31, var_32)
    var_34 = 'Range error'
    var_35 = module_0.ValidationError(text=var_34)
    var_36 = 'Default code error'
    var_37 = module_0.ValidationError(text=var_36)
    var_38 = module_0.ValidationError()
    var_39 = 'Error'
    var_40 = []
    var_41 = module_0.ValidationError(text=var_39, messages=var_40)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = module_0.BaseError(text=var_0, position=var_8)
    var_10 = 'Error 1'
    var_11 = 'code1'
    var_12 = 'key1'
    var_13 = module_0.Message(text=var_10, code=var_11, key=var_12)
    var_14 = 'Error 2'
    var_15 = 'code2'
    var_16 = 'key2'
    var_17 = module_0.Message(text=var_14, code=var_15, key=var_16)
    var_18 = [var_13, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'Nested error'
    var_22 = 'parent'
    var_23 = 'child'
    var_24 = [var_22, var_23]
    var_25 = module_0.Message(text=var_21, index=var_24)
    var_26 = [var_25]
    var_27 = module_0.BaseError(messages=var_26)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'Error 1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    var_9 = 'Error 2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    var_13 = [var_8, var_12]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = module_0.ValidationError()
    var_17 = 'Error'
    var_18 = module_0.ValidationError(text=var_17, messages=var_13)
    var_19 = []
    var_20 = module_0.ValidationError(messages=var_19)



# Parsed testcases at query #20
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
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = module_0.Position(var_10, var_11, var_12)
    var_14 = module_0.Message(text=var_0, position=var_13)
    var_15 = module_0.Message(text=var_0, position=var_13)
    var_16 = module_0.Position(var_10, var_11, var_12)
    var_17 = 5
    var_18 = 8
    var_19 = module_0.Position(var_10, var_17, var_18)
    var_20 = module_0.Message(text=var_0, start_position=var_16, end_position=var_19)
    var_21 = module_0.Message(text=var_0, start_position=var_16, end_position=var_19)
    var_22 = module_0.Position(var_11, var_11, var_12)
    var_23 = module_0.Message(text=var_0, position=var_22)



# Parsed testcases at query #21
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
    var_33 = module_0.Position(var_18, var_20, var_20)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 11
    var_36 = module_0.Position(var_18, var_28, var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_36)



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



# Parsed testcases at query #24
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
    var_33 = module_0.Position(var_18, var_20, var_20)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_30)
    var_35 = 11
    var_36 = module_0.Position(var_18, var_28, var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_27, end_position=var_36)



# Parsed testcases at query #25
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
    var_9 = 'different_field'
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
    var_11 = 'users'
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



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = module_0.Position(var_10, var_11, var_12)
    var_14 = module_0.Message(text=var_0, code=var_1, position=var_13)
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_13)
    var_16 = 4
    var_17 = module_0.Position(var_10, var_12, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_19 = module_0.Position(var_10, var_11, var_12)
    var_20 = 5
    var_21 = 8
    var_22 = module_0.Position(var_10, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_22)
    var_25 = module_0.Position(var_10, var_12, var_16)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_22)
    var_27 = 6
    var_28 = 9
    var_29 = module_0.Position(var_10, var_27, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_29)



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
    var_9 = 'other_field'
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



