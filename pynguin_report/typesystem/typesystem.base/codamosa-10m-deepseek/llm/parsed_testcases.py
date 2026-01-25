####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = module_0.Position(var_4, var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_7)
    var_9 = repr(var_8)
    assert var_9 == "Message(text='example', code='custom', index=['field'], position=Position(line_no=1, column_no=2, char_index=3))"
    var_10 = [var_2]
    var_11 = module_0.Position(var_4, var_5, var_6)
    var_12 = 4
    var_13 = 5
    var_14 = 6
    var_15 = module_0.Position(var_12, var_13, var_14)
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_10, start_position=var_11, end_position=var_15)
    var_17 = repr(var_16)
    assert var_17 == "Message(text='example', code='custom', index=['field'], start_position=Position(line_no=1, column_no=2, char_index=3), end_position=Position(line_no=4, column_no=5, char_index=6))"
    var_18 = module_0.Message(text=var_0, code=var_1)
    var_19 = repr(var_18)
    assert var_19 == "Message(text='example', code='custom')"



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    var_4 = repr(var_3)
    assert var_4 == "ValidationResult(value={'name': 'John'})"
    var_5 = 'Invalid name'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = repr(var_7)
    assert var_8 == "ValidationResult(error=ValidationError(text='Invalid name', code='custom'))"



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 1
    var_4 = module_0.Position(var_3, var_3, var_3)
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Message 1'
    var_8 = 'code1'
    var_9 = 'key1'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    var_11 = 'Message 2'
    var_12 = 'code2'
    var_13 = 'key2'
    var_14 = module_0.Message(text=var_11, code=var_12, key=var_13)
    var_15 = [var_10, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'Nested message'
    var_19 = 'code3'
    var_20 = 'key3'
    var_21 = 'subkey'
    var_22 = [var_20, var_21]
    var_23 = module_0.Message(text=var_18, code=var_19, index=var_22)
    var_24 = [var_23]
    var_25 = module_0.BaseError(messages=var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = 'Message 3'
    var_28 = module_0.Message(text=var_27, code=var_19, key=var_20)
    var_29 = [var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = [var_28]
    var_32 = module_0.BaseError(messages=var_31)
    var_33 = 'Message 4'
    var_34 = 'code4'
    var_35 = 'key4'
    var_36 = module_0.Message(text=var_33, code=var_34, key=var_35)
    var_37 = [var_36]
    var_38 = module_0.BaseError(messages=var_37)
    var_39 = [var_36]
    var_40 = module_0.BaseError(messages=var_39)
    var_41 = hash(var_38)
    var_42 = hash(var_40)
    var_43 = 'Message 5'
    var_44 = 'code5'
    var_45 = module_0.Message(text=var_43, code=var_44)
    var_46 = module_0.BaseError(text=var_43, code=var_44)
    var_47 = repr(var_46)
    assert var_47 == "BaseError(text='Message 5', code='code5')"
    var_48 = 'Message 6'
    var_49 = 'code6'
    var_50 = module_0.Message(text=var_48, code=var_49)
    var_51 = module_0.BaseError(text=var_48, code=var_49)
    var_52 = str(var_51)
    assert var_52 == 'Message 6'
    var_53 = 'Message 7'
    var_54 = 'code7'
    var_55 = 'key7'
    var_56 = module_0.Message(text=var_53, code=var_54, key=var_55)
    var_57 = 'Message 8'
    var_58 = 'code8'
    var_59 = 'key8'
    var_60 = module_0.Message(text=var_57, code=var_58, key=var_59)
    var_61 = [var_56, var_60]
    var_62 = module_0.BaseError(messages=var_61)
    var_63 = list(var_62)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'Error message'
    var_2 = module_0.ValidationError(text=var_1)
    var_3 = module_0.ValidationResult(value=var_0)
    var_4 = list(var_3)
    var_5 = module_0.ValidationResult(error=var_2)
    var_6 = list(var_5)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = module_0.ValidationError(text=var_2)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'Error 1'
    var_5 = 'error1'
    var_6 = 'field1'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    var_9 = 'Error 2'
    var_10 = 'error2'
    var_11 = 'field2'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    var_14 = [var_8, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    var_16 = repr(var_15)
    var_17 = module_0.BaseError(text=var_0, code=var_1, key=var_6)
    var_18 = repr(var_17)
    assert var_18 == "BaseError([Message(text='Error message', code='error_code', index=['field1'])])"



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'This is an error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='This is an error message', code='error_code', index=['error_key'])])"



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Invalid value', code='invalid')"
    var_4 = 'field1'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Missing field'
    var_8 = 'missing'
    var_9 = 'field2'
    var_10 = [var_9]
    var_11 = module_0.Message(text=var_7, code=var_8, index=var_10)
    var_12 = [var_6, var_11]
    var_13 = module_0.BaseError(messages=var_12)
    var_14 = repr(var_13)
    assert var_14 == "BaseError([Message(text='Invalid value', code='invalid', index=['field1']), Message(text='Missing field', code='missing', index=['field2'])])"
    var_15 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_16 = repr(var_15)
    assert var_16 == "BaseError(text='Invalid value', code='invalid')"
    var_17 = 1
    var_18 = 5
    var_19 = 10
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.BaseError(text=var_0, code=var_1, position=var_20)
    var_22 = repr(var_21)
    assert var_22 == "BaseError(text='Invalid value', code='invalid', position=Position(line_no=1, column_no=5, char_index=10))"



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "ValidationError(text='Error message', code='error_code')"
    var_4 = 'key'
    var_5 = module_0.ValidationError(text=var_0, code=var_1, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "ValidationError([Message(text='Error message', code='error_code', index=['key'])])"
    var_7 = 'Error 1'
    var_8 = 'error_1'
    var_9 = 'key1'
    var_10 = [var_9]
    var_11 = module_0.Message(text=var_7, code=var_8, index=var_10)
    var_12 = 'Error 2'
    var_13 = 'error_2'
    var_14 = 'key2'
    var_15 = [var_14]
    var_16 = module_0.Message(text=var_12, code=var_13, index=var_15)
    var_17 = [var_11, var_16]
    var_18 = module_0.ValidationError(messages=var_17)
    var_19 = repr(var_18)
    assert var_19 == "ValidationError([Message(text='Error 1', code='error_1', index=['key1']), Message(text='Error 2', code='error_2', index=['key2'])])"



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_4)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7, position=var_8)
    var_10 = 'Different message'
    var_11 = [var_2]
    var_12 = module_0.Position(var_4, var_4, var_4)
    var_13 = module_0.Message(text=var_10, code=var_1, key=var_2, index=var_11, position=var_12)
    var_14 = 'different'
    var_15 = [var_2]
    var_16 = module_0.Position(var_4, var_4, var_4)
    var_17 = module_0.Message(text=var_0, code=var_14, key=var_2, index=var_15, position=var_16)
    var_18 = 'key2'
    var_19 = [var_18]
    var_20 = module_0.Position(var_4, var_4, var_4)
    var_21 = module_0.Message(text=var_0, code=var_1, key=var_18, index=var_19, position=var_20)
    var_22 = [var_2]
    var_23 = module_0.Position(var_4, var_4, var_4)
    var_24 = module_0.Position(var_4, var_4, var_4)
    var_25 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_22, start_position=var_23, end_position=var_24)
    var_26 = [var_2]
    var_27 = module_0.Position(var_4, var_4, var_4)
    var_28 = 2
    var_29 = module_0.Position(var_4, var_28, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_26, start_position=var_27, end_position=var_29)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_code'
    var_2 = 'test_key'
    var_3 = 'idx1'
    var_4 = 'idx2'
    var_5 = [var_3, var_4]
    var_6 = 1
    var_7 = module_0.Position(var_6, var_6, var_6)
    var_8 = module_0.Position(var_6, var_6, var_6)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5, start_position=var_7, end_position=var_8)
    var_10 = [var_3, var_4]
    var_11 = module_0.Position(var_6, var_6, var_6)
    var_12 = module_0.Position(var_6, var_6, var_6)
    var_13 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_10, start_position=var_11, end_position=var_12)
    var_14 = 'different'
    var_15 = [var_3, var_4]
    var_16 = module_0.Position(var_6, var_6, var_6)
    var_17 = module_0.Position(var_6, var_6, var_6)
    var_18 = module_0.Message(text=var_14, code=var_1, key=var_2, index=var_15, start_position=var_16, end_position=var_17)
    var_19 = [var_3, var_4]
    var_20 = module_0.Position(var_6, var_6, var_6)
    var_21 = module_0.Position(var_6, var_6, var_6)
    var_22 = module_0.Message(text=var_0, code=var_14, key=var_2, index=var_19, start_position=var_20, end_position=var_21)
    var_23 = [var_3]
    var_24 = module_0.Position(var_6, var_6, var_6)
    var_25 = module_0.Position(var_6, var_6, var_6)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, start_position=var_24, end_position=var_25)
    var_27 = [var_3, var_4]
    var_28 = 2
    var_29 = module_0.Position(var_28, var_6, var_6)
    var_30 = module_0.Position(var_6, var_6, var_6)
    var_31 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_27, start_position=var_29, end_position=var_30)
    var_32 = [var_3, var_4]
    var_33 = module_0.Position(var_6, var_6, var_6)
    var_34 = module_0.Position(var_28, var_6, var_6)
    var_35 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_32, start_position=var_33, end_position=var_34)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = [var_15]
    var_20 = module_0.Position(var_5, var_5, var_5)
    var_21 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_19, position=var_20)
    var_22 = [var_3]
    var_23 = 2
    var_24 = module_0.Position(var_23, var_23, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_22, position=var_24)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    var_2 = 2
    var_3 = module_0.Position(var_2, var_2, var_2)
    var_4 = 'Error'
    var_5 = 'error'
    var_6 = [var_0]
    var_7 = module_0.Message(text=var_4, code=var_5, index=var_6, start_position=var_1, end_position=var_3)
    var_8 = [var_0]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8, start_position=var_1, end_position=var_3)
    var_10 = 'Different'
    var_11 = [var_0]
    var_12 = module_0.Message(text=var_10, code=var_5, index=var_11, start_position=var_1, end_position=var_3)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_4, code=var_5, index=var_13, start_position=var_1, end_position=var_3)
    var_15 = 'different'
    var_16 = [var_0]
    var_17 = module_0.Message(text=var_4, code=var_15, index=var_16, start_position=var_1, end_position=var_3)
    var_18 = [var_0]
    var_19 = module_0.Message(text=var_4, code=var_5, index=var_18, start_position=var_3, end_position=var_3)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'test1'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)
    var_10 = 'test2'
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_10, code=var_1, index=var_11)
    var_13 = 'code1'
    var_14 = [var_2]
    var_15 = module_0.Message(text=var_0, code=var_13, index=var_14)
    var_16 = 'code2'
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, code=var_16, index=var_17)
    var_19 = 'key1'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 'key2'
    var_23 = [var_22]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    var_25 = 1
    var_26 = module_0.Position(var_25, var_25, var_25)
    var_27 = 2
    var_28 = module_0.Position(var_27, var_27, var_27)
    var_29 = [var_2]
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_29, position=var_26)
    var_31 = [var_2]
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_31, position=var_28)
    var_33 = [var_2]
    var_34 = module_0.Message(text=var_0, code=var_1, index=var_33)
    var_35 = 'test'



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error message'
    var_5 = 'error_code'
    var_6 = 'key'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, position=var_3)



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
    var_9 = 'Error'
    var_10 = 'custom'
    var_11 = 'key'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12, start_position=var_3, end_position=var_3)
    var_14 = [var_11]
    var_15 = module_0.Message(text=var_9, code=var_10, index=var_14, start_position=var_4, end_position=var_4)
    var_16 = 'Different'
    var_17 = [var_11]
    var_18 = module_0.Message(text=var_16, code=var_10, index=var_17, start_position=var_3, end_position=var_3)
    var_19 = 'different'
    var_20 = [var_11]
    var_21 = module_0.Message(text=var_9, code=var_19, index=var_20, start_position=var_3, end_position=var_3)
    var_22 = [var_19]
    var_23 = module_0.Message(text=var_9, code=var_10, index=var_22, start_position=var_3, end_position=var_3)
    var_24 = [var_11]
    var_25 = module_0.Message(text=var_9, code=var_10, index=var_24, start_position=var_8, end_position=var_8)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_4)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7, position=var_8)
    var_10 = 'different'
    var_11 = [var_2]
    var_12 = module_0.Position(var_4, var_4, var_4)
    var_13 = module_0.Message(text=var_10, code=var_1, key=var_2, index=var_11, position=var_12)
    var_14 = [var_2]
    var_15 = module_0.Position(var_4, var_4, var_4)
    var_16 = module_0.Message(text=var_0, code=var_10, key=var_2, index=var_14, position=var_15)
    var_17 = [var_10]
    var_18 = module_0.Position(var_4, var_4, var_4)
    var_19 = module_0.Message(text=var_0, code=var_1, key=var_10, index=var_17, position=var_18)
    var_20 = [var_2]
    var_21 = 2
    var_22 = module_0.Position(var_21, var_21, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_22)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4)
    var_6 = [var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_6)
    var_8 = 'Different message'
    var_9 = [var_3]
    var_10 = module_0.Message(text=var_8, code=var_1, key=var_2, index=var_9)
    var_11 = 'different'
    var_12 = [var_3]
    var_13 = module_0.Message(text=var_0, code=var_11, key=var_2, index=var_12)
    var_14 = [var_3]
    var_15 = module_0.Message(text=var_0, code=var_1, key=var_11, index=var_14)
    var_16 = [var_11]
    var_17 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_16)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Test message'
    var_1 = 'test_code'
    var_2 = 'test_key'
    var_3 = 'test_index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = module_0.Position(var_5, var_5, var_5)
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = [var_3]
    var_11 = module_0.Position(var_5, var_5, var_5)
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_10, position=var_11)
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_13)
    var_15 = module_0.Position(var_5, var_5, var_5)
    var_16 = 2
    var_17 = module_0.Position(var_16, var_16, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_15, end_position=var_17)
    var_19 = module_0.Position(var_5, var_5, var_5)
    var_20 = module_0.Position(var_16, var_16, var_16)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test message'
    var_1 = 'test_code'
    var_2 = 'test_index'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_code'
    var_2 = 'test_key'
    var_3 = 'test_index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'different'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = [var_3]
    var_16 = module_0.Position(var_5, var_5, var_5)
    var_17 = module_0.Message(text=var_0, code=var_11, key=var_2, index=var_15, position=var_16)
    var_18 = [var_11]
    var_19 = module_0.Position(var_5, var_5, var_5)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_18, position=var_19)
    var_21 = [var_3]
    var_22 = 2
    var_23 = module_0.Position(var_22, var_22, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_21, position=var_23)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Different Error'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'index'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_4)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_7, position=var_8)
    var_10 = 'Different message'
    var_11 = [var_2]
    var_12 = module_0.Position(var_4, var_4, var_4)
    var_13 = module_0.Message(text=var_10, code=var_1, index=var_11, position=var_12)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = 'different_index'
    var_20 = [var_19]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_21)
    var_23 = [var_3]
    var_24 = 2
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_25)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = '\n    Keyword arguments:\n    No keyword arguments.\n    \n    Expected return:\n    No return value.\n    \n    Expected side effects:\n    Assertions should pass if the method works correctly.\n    '
    var_1 = 'Error'
    var_2 = 'custom'
    var_3 = 'key'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_1, code=var_2, index=var_4)
    var_6 = [var_3]
    var_7 = module_0.Message(text=var_1, code=var_2, index=var_6)
    var_8 = [var_3]
    var_9 = module_0.Message(text=var_1, code=var_2, index=var_8)
    var_10 = 'Different Error'
    var_11 = [var_3]
    var_12 = module_0.Message(text=var_10, code=var_2, index=var_11)
    var_13 = [var_3]
    var_14 = module_0.Message(text=var_1, code=var_2, index=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Message(text=var_1, code=var_15, index=var_16)
    var_18 = [var_3]
    var_19 = module_0.Message(text=var_1, code=var_2, index=var_18)
    var_20 = 'different_key'
    var_21 = [var_20]
    var_22 = module_0.Message(text=var_1, code=var_2, index=var_21)
    var_23 = 1
    var_24 = module_0.Position(var_23, var_23, var_23)
    var_25 = 2
    var_26 = module_0.Position(var_25, var_25, var_25)
    var_27 = [var_3]
    var_28 = module_0.Message(text=var_1, code=var_2, index=var_27, start_position=var_24)
    var_29 = [var_3]
    var_30 = module_0.Message(text=var_1, code=var_2, index=var_29, start_position=var_26)
    var_31 = [var_3]
    var_32 = module_0.Message(text=var_1, code=var_2, index=var_31, end_position=var_24)
    var_33 = [var_3]
    var_34 = module_0.Message(text=var_1, code=var_2, index=var_33, end_position=var_26)
    var_35 = [var_3]
    var_36 = module_0.Message(text=var_1, code=var_2, index=var_35)
    var_37 = 'text'
    var_38 = 'code'
    var_39 = 'index'
    var_40 = [var_3]
    var_41 = {var_37: var_1, var_38: var_2, var_39: var_40}



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = 'index1'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = 'different_index'
    var_20 = [var_19]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_21)
    var_23 = [var_3]
    var_24 = 2
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_25)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test message'
    var_1 = 'test_code'
    var_2 = 'test_key'
    var_3 = 'test_index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'different_error'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'different_field'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = 'subfield'
    var_12 = [var_2, var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = module_0.Position(var_14, var_14, var_14)
    var_16 = 2
    var_17 = module_0.Position(var_16, var_16, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_19 = module_0.Message(text=var_0, code=var_1, position=var_17)
    var_20 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_21 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_17)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_15)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_17)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 0
    var_6 = module_0.Position(var_2, var_2, var_5)
    var_7 = 10
    var_8 = 9
    var_9 = module_0.Position(var_2, var_7, var_8)
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_4, start_position=var_6, end_position=var_9)
    var_11 = [var_2, var_3]
    var_12 = module_0.Position(var_2, var_2, var_5)
    var_13 = module_0.Position(var_2, var_7, var_8)
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_11, start_position=var_12, end_position=var_13)
    var_15 = 'Different message'
    var_16 = [var_2, var_3]
    var_17 = module_0.Position(var_2, var_2, var_5)
    var_18 = module_0.Position(var_2, var_7, var_8)
    var_19 = module_0.Message(text=var_15, code=var_1, index=var_16, start_position=var_17, end_position=var_18)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Position(var_2, var_2, var_2)
    var_5 = 5
    var_6 = module_0.Position(var_2, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_4, end_position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_2, var_2, var_2)
    var_10 = module_0.Position(var_2, var_5, var_5)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_8, start_position=var_9, end_position=var_10)
    var_12 = 'Different message'
    var_13 = [var_2]
    var_14 = module_0.Position(var_2, var_2, var_2)
    var_15 = module_0.Position(var_2, var_5, var_5)
    var_16 = module_0.Message(text=var_12, code=var_1, index=var_13, start_position=var_14, end_position=var_15)
    var_17 = 'different_code'
    var_18 = [var_2]
    var_19 = module_0.Position(var_2, var_2, var_2)
    var_20 = module_0.Position(var_2, var_5, var_5)
    var_21 = module_0.Message(text=var_0, code=var_17, index=var_18, start_position=var_19, end_position=var_20)
    var_22 = 2
    var_23 = [var_22]
    var_24 = module_0.Position(var_2, var_2, var_2)
    var_25 = module_0.Position(var_2, var_5, var_5)
    var_26 = module_0.Message(text=var_0, code=var_1, index=var_23, start_position=var_24, end_position=var_25)
    var_27 = [var_2]
    var_28 = module_0.Position(var_22, var_2, var_2)
    var_29 = module_0.Position(var_2, var_5, var_5)
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_27, start_position=var_28, end_position=var_29)
    var_31 = [var_2]
    var_32 = module_0.Position(var_2, var_2, var_2)
    var_33 = module_0.Position(var_22, var_5, var_5)
    var_34 = module_0.Message(text=var_0, code=var_1, index=var_31, start_position=var_32, end_position=var_33)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test message'
    var_1 = 'test_code'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'different message'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_2, var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_6)
    var_8 = 'Different error'
    var_9 = [var_2, var_3]
    var_10 = module_0.Message(text=var_8, code=var_1, index=var_9)
    var_11 = 'different_code'
    var_12 = [var_2, var_3]
    var_13 = module_0.Message(text=var_0, code=var_11, index=var_12)
    var_14 = 'key3'
    var_15 = [var_2, var_14]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = 'index1'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4)
    var_6 = [var_3]
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_6)
    var_8 = 'Different message'
    var_9 = [var_3]
    var_10 = module_0.Message(text=var_8, code=var_1, key=var_2, index=var_9)
    var_11 = 'different_code'
    var_12 = [var_3]
    var_13 = module_0.Message(text=var_0, code=var_11, key=var_2, index=var_12)
    var_14 = 'different_index'
    var_15 = [var_14]
    var_16 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_15)
    var_17 = 1
    var_18 = module_0.Position(var_17, var_17, var_17)
    var_19 = 2
    var_20 = module_0.Position(var_19, var_19, var_19)
    var_21 = [var_3]
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_21, start_position=var_18)
    var_23 = [var_3]
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, start_position=var_20)
    var_25 = [var_3]
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_25, end_position=var_18)
    var_27 = [var_3]
    var_28 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_27, end_position=var_20)



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    var_2 = 'test'
    var_3 = 'test_code'
    var_4 = 'test_index'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_2, code=var_3, index=var_5, position=var_1)
    var_7 = [var_4]
    var_8 = module_0.Message(text=var_2, code=var_3, index=var_7, position=var_1)
    var_9 = 'different'
    var_10 = [var_4]
    var_11 = module_0.Message(text=var_9, code=var_3, index=var_10, position=var_1)
    var_12 = [var_4]
    var_13 = module_0.Message(text=var_2, code=var_9, index=var_12, position=var_1)
    var_14 = [var_9]
    var_15 = module_0.Message(text=var_2, code=var_3, index=var_14, position=var_1)
    var_16 = [var_4]
    var_17 = None
    var_18 = module_0.Message(text=var_2, code=var_3, index=var_16, position=var_17)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = module_0.Position(var_4, var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_7)
    var_9 = [var_2]
    var_10 = module_0.Position(var_4, var_5, var_6)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_9, position=var_10)
    var_12 = 'Different Error'
    var_13 = [var_2]
    var_14 = module_0.Position(var_4, var_5, var_6)
    var_15 = module_0.Message(text=var_12, code=var_1, index=var_13, position=var_14)
    var_16 = 'different_error'
    var_17 = [var_2]
    var_18 = module_0.Position(var_4, var_5, var_6)
    var_19 = module_0.Message(text=var_0, code=var_16, index=var_17, position=var_18)
    var_20 = 'different_key'
    var_21 = [var_20]
    var_22 = module_0.Position(var_4, var_5, var_6)
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_21, position=var_22)
    var_24 = [var_2]
    var_25 = 4
    var_26 = 5
    var_27 = 6
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, index=var_24, position=var_28)



# Parsed testcases at query #36
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Test message'
    var_1 = 'test_code'
    var_2 = 'test_key'
    var_3 = 'test_index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_8)
    var_10 = [var_3]
    var_11 = module_0.Position(var_5, var_6, var_7)
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_10, position=var_11)
    var_13 = 'Different message'
    var_14 = [var_3]
    var_15 = module_0.Position(var_5, var_6, var_7)
    var_16 = module_0.Message(text=var_13, code=var_1, key=var_2, index=var_14, position=var_15)



# Parsed testcases at query #37
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 'Different Error'
    var_10 = [var_2]
    var_11 = module_0.Message(text=var_9, code=var_1, index=var_10)
    var_12 = [var_2]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)



# Parsed testcases at query #38
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error #1'
    var_1 = 'error1'
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Position(var_4, var_4, var_4)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_4)
    var_10 = module_0.Position(var_4, var_4, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_8, start_position=var_9, end_position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_4, var_4, var_4)
    var_14 = module_0.Position(var_4, var_4, var_4)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_12, start_position=var_13, end_position=var_14)
    var_16 = [var_2]
    var_17 = 2
    var_18 = module_0.Position(var_4, var_17, var_4)
    var_19 = module_0.Position(var_4, var_4, var_4)
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_16, start_position=var_18, end_position=var_19)
    var_21 = [var_2]
    var_22 = module_0.Position(var_4, var_4, var_4)
    var_23 = module_0.Position(var_4, var_4, var_4)
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_21, start_position=var_22, end_position=var_23)
    var_25 = [var_2]
    var_26 = module_0.Position(var_4, var_4, var_4)
    var_27 = module_0.Position(var_4, var_17, var_4)
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_25, start_position=var_26, end_position=var_27)
    var_29 = [var_2]
    var_30 = module_0.Position(var_4, var_4, var_4)
    var_31 = module_0.Position(var_4, var_4, var_4)
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_29, start_position=var_30, end_position=var_31)
    var_33 = [var_2]
    var_34 = module_0.Position(var_4, var_4, var_4)
    var_35 = module_0.Position(var_4, var_4, var_17)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_33, start_position=var_34, end_position=var_35)
    var_37 = [var_2]
    var_38 = module_0.Position(var_4, var_4, var_4)
    var_39 = module_0.Position(var_4, var_4, var_4)
    var_40 = module_0.Message(text=var_0, code=var_1, index=var_37, start_position=var_38, end_position=var_39)
    var_41 = [var_2]
    var_42 = module_0.Position(var_4, var_4, var_4)
    var_43 = module_0.Position(var_4, var_4, var_4)
    var_44 = module_0.Message(text=var_0, code=var_1, index=var_41, start_position=var_42, end_position=var_43)
    var_45 = [var_2]
    var_46 = module_0.Position(var_4, var_4, var_4)
    var_47 = module_0.Position(var_4, var_4, var_4)
    var_48 = module_0.Message(text=var_0, code=var_1, index=var_45, start_position=var_46, end_position=var_47)
    var_49 = [var_2]
    var_50 = module_0.Position(var_4, var_4, var_4)
    var_51 = module_0.Position(var_4, var_4, var_4)
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_49, start_position=var_50, end_position=var_51)
    var_53 = [var_2]
    var_54 = module_0.Position(var_4, var_4, var_4)
    var_55 = module_0.Position(var_4, var_4, var_4)
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_53, start_position=var_54, end_position=var_55)
    var_57 = [var_2]
    var_58 = module_0.Position(var_4, var_4, var_4)
    var_59 = module_0.Position(var_4, var_4, var_4)
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_57, start_position=var_58, end_position=var_59)
    var_61 = [var_2]
    var_62 = module_0.Position(var_4, var_4, var_4)
    var_63 = module_0.Position(var_4, var_4, var_4)
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_61, start_position=var_62, end_position=var_63)
    var_65 = [var_2]
    var_66 = module_0.Position(var_4, var_4, var_4)
    var_67 = module_0.Position(var_4, var_4, var_4)
    var_68 = module_0.Message(text=var_0, code=var_1, index=var_65, start_position=var_66, end_position=var_67)
    var_69 = [var_2]
    var_70 = module_0.Position(var_4, var_4, var_4)
    var_71 = module_0.Position(var_4, var_4, var_4)
    var_72 = module_0.Message(text=var_0, code=var_1, index=var_69, start_position=var_70, end_position=var_71)
    var_73 = [var_2]
    var_74 = module_0.Position(var_4, var_4, var_4)
    var_75 = module_0.Position(var_4, var_4, var_4)
    var_76 = module_0.Message(text=var_0, code=var_1, index=var_73, start_position=var_74, end_position=var_75)
    var_77 = [var_2]
    var_78 = module_0.Position(var_4, var_4, var_4)
    var_79 = module_0.Position(var_4, var_4, var_4)
    var_80 = module_0.Message(text=var_0, code=var_1, index=var_77, start_position=var_78, end_position=var_79)
    var_81 = [var_2]
    var_82 = module_0.Position(var_4, var_4, var_4)
    var_83 = module_0.Position(var_4, var_4, var_4)
    var_84 = module_0.Message(text=var_0, code=var_1, index=var_81, start_position=var_82, end_position=var_83)
    var_85 = [var_2]
    var_86 = module_0.Position(var_4, var_4, var_4)
    var_87 = module_0.Position(var_4, var_4, var_4)
    var_88 = module_0.Message(text=var_0, code=var_1, index=var_85, start_position=var_86, end_position=var_87)
    var_89 = [var_2]
    var_90 = module_0.Position(var_4, var_4, var_4)
    var_91 = module_0.Position(var_4, var_4, var_4)
    var_92 = module_0.Message(text=var_0, code=var_1, index=var_89, start_position=var_90, end_position=var_91)
    var_93 = [var_2]
    var_94 = module_0.Position(var_4, var_4, var_4)
    var_95 = module_0.Position(var_4, var_4, var_4)
    var_96 = module_0.Message(text=var_0, code=var_1, index=var_93, start_position=var_94, end_position=var_95)
    var_97 = [var_2]
    var_98 = module_0.Position(var_4, var_4, var_4)
    var_99 = module_0.Position(var_4, var_4, var_4)
    var_100 = module_0.Message(text=var_0, code=var_1, index=var_97, start_position=var_98, end_position=var_99)
    var_101 = [var_2]
    var_102 = module_0.Position(var_4, var_4, var_4)
    var_103 = module_0.Position(var_4, var_4, var_4)
    var_104 = module_0.Message(text=var_0, code=var_1, index=var_101, start_position=var_102, end_position=var_103)
    var_105 = [var_2]
    var_106 = module_0.Position(var_4, var_4, var_4)
    var_107 = module_0.Position(var_4, var_4, var_4)
    var_108 = module_0.Message(text=var_0, code=var_1, index=var_105, start_position=var_106, end_position=var_107)
    var_109 = [var_2]
    var_110 = module_0.Position(var_4, var_4, var_4)
    var_111 = module_0.Position(var_4, var_4, var_4)
    var_112 = module_0.Message(text=var_0, code=var_1, index=var_109, start_position=var_110, end_position=var_111)
    var_113 = [var_2]
    var_114 = module_0.Position(var_4, var_4, var_4)
    var_115 = module_0.Position(var_4, var_4, var_4)
    var_116 = module_0.Message(text=var_0, code=var_1, index=var_113, start_position=var_114, end_position=var_115)
    var_117 = [var_2]
    var_118 = module_0.Position(var_4, var_4, var_4)
    var_119 = module_0.Position(var_4, var_4, var_4)
    var_120 = module_0.Message(text=var_0, code=var_1, index=var_117, start_position=var_118, end_position=var_119)
    var_121 = [var_2]
    var_122 = module_0.Position(var_4, var_4, var_4)
    var_123 = module_0.Position(var_4, var_4, var_4)
    var_124 = module_0.Message(text=var_0, code=var_1, index=var_121, start_position=var_122, end_position=var_123)
    var_125 = [var_2]
    var_126 = module_0.Position(var_4, var_4, var_4)
    var_127 = module_0.Position(var_4, var_4, var_4)
    var_128 = module_0.Message(text=var_0, code=var_1, index=var_125, start_position=var_126, end_position=var_127)
    var_129 = [var_2]
    var_130 = module_0.Position(var_4, var_4, var_4)
    var_131 = module_0.Position(var_4, var_4, var_4)
    var_132 = module_0.Message(text=var_0, code=var_1, index=var_129, start_position=var_130, end_position=var_131)
    var_133 = [var_2]
    var_134 = module_0.Position(var_4, var_4, var_4)
    var_135 = module_0.Position(var_4, var_4, var_4)
    var_136 = module_0.Message(text=var_0, code=var_1, index=var_133, start_position=var_134, end_position=var_135)
    var_137 = [var_2]
    var_138 = module_0.Position(var_4, var_4, var_4)
    var_139 = module_0.Position(var_4, var_4, var_4)
    var_140 = module_0.Message(text=var_0, code=var_1, index=var_137, start_position=var_138, end_position=var_139)
    var_141 = [var_2]
    var_142 = module_0.Position(var_4, var_4, var_4)
    var_143 = module_0.Position(var_4, var_4, var_4)
    var_144 = module_0.Message(text=var_0, code=var_1, index=var_141, start_position=var_142, end_position=var_143)
    var_145 = [var_2]
    var_146 = module_0.Position(var_4, var_4, var_4)
    var_147 = module_0.Position(var_4, var_4, var_4)
    var_148 = module_0.Message(text=var_0, code=var_1, index=var_145, start_position=var_146, end_position=var_147)
    var_149 = [var_2]
    var_150 = module_0.Position(var_4, var_4, var_4)
    var_151 = module_0.Position(var_4, var_4, var_4)
    var_152 = module_0.Message(text=var_0, code=var_1, index=var_149, start_position=var_150, end_position=var_151)
    var_153 = [var_2]
    var_154 = module_0.Position(var_4, var_4, var_4)
    var_155 = module_0.Position(var_4, var_4, var_4)
    var_156 = module_0.Message(text=var_0, code=var_1, index=var_153, start_position=var_154, end_position=var_155)
    var_157 = [var_2]
    var_158 = module_0.Position(var_4, var_4, var_4)
    var_159 = module_0.Position(var_4, var_4, var_4)
    var_160 = module_0.Message(text=var_0, code=var_1, index=var_157, start_position=var_158, end_position=var_159)
    var_161 = [var_2]
    var_162 = module_0.Position(var_4, var_4, var_4)
    var_163 = module_0.Position(var_4, var_4, var_4)
    var_164 = module_0.Message(text=var_0, code=var_1, index=var_161, start_position=var_162, end_position=var_163)
    var_165 = [var_2]
    var_166 = module_0.Position(var_4, var_4, var_4)
    var_167 = module_0.Position(var_4, var_4, var_4)
    var_168 = module_0.Message(text=var_0, code=var_1, index=var_165, start_position=var_166, end_position=var_167)
    var_169 = [var_2]
    var_170 = module_0.Position(var_4, var_4, var_4)
    var_171 = module_0.Position(var_4, var_4, var_4)
    var_172 = module_0.Message(text=var_0, code=var_1, index=var_169, start_position=var_170, end_position=var_171)
    var_173 = [var_2]
    var_174 = module_0.Position(var_4, var_4, var_4)
    var_175 = module_0.Position(var_4, var_4, var_4)
    var_176 = module_0.Message(text=var_0, code=var_1, index=var_173, start_position=var_174, end_position=var_175)
    var_177 = [var_2]
    var_178 = module_0.Position(var_4, var_4, var_4)
    var_179 = module_0.Position(var_4, var_4, var_4)
    var_180 = module_0.Message(text=var_0, code=var_1, index=var_177, start_position=var_178, end_position=var_179)



# Parsed testcases at query #39
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = module_0.Position(var_4, var_5, var_6)
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_7)
    var_9 = [var_2]
    var_10 = module_0.Position(var_4, var_5, var_6)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_9, position=var_10)
    var_12 = 'Different message'
    var_13 = [var_2]
    var_14 = module_0.Position(var_4, var_5, var_6)
    var_15 = module_0.Message(text=var_12, code=var_1, index=var_13, position=var_14)
    var_16 = 'different_code'
    var_17 = [var_2]
    var_18 = module_0.Position(var_4, var_5, var_6)
    var_19 = module_0.Message(text=var_0, code=var_16, index=var_17, position=var_18)
    var_20 = 'different_field'
    var_21 = [var_20]
    var_22 = module_0.Position(var_4, var_5, var_6)
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_21, position=var_22)
    var_24 = [var_2]
    var_25 = 4
    var_26 = module_0.Position(var_5, var_6, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, index=var_24, position=var_26)
    var_28 = module_0.Position(var_4, var_5, var_6)
    var_29 = var_8 == var_28



# Parsed testcases at query #40
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = 'different_index'
    var_20 = [var_19]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_21)
    var_23 = [var_3]
    var_24 = 2
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_25)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
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
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = module_0.Position(var_5, var_6, var_7)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = module_0.ValidationResult()



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



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.BaseError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.BaseError(text=var_0, code=var_6)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = 1
    var_4 = module_0.Position(var_3, var_3, var_3)
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_4)
    var_6 = module_0.Position(var_3, var_3, var_3)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = 'Different message'
    var_9 = module_0.Position(var_3, var_3, var_3)
    var_10 = module_0.Message(text=var_8, code=var_1, key=var_2, position=var_9)
    var_11 = 2
    var_12 = module_0.Position(var_11, var_11, var_11)
    var_13 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_12)
    var_14 = 'password'
    var_15 = module_0.Position(var_3, var_3, var_3)
    var_16 = module_0.Message(text=var_0, code=var_1, key=var_14, position=var_15)
    var_17 = 'different'
    var_18 = module_0.Position(var_3, var_3, var_3)
    var_19 = module_0.Message(text=var_0, code=var_17, key=var_2, position=var_18)
    var_20 = module_0.Position(var_3, var_3, var_3)
    var_21 = 'users'
    var_22 = 0
    var_23 = [var_21, var_22, var_2]
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_20)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    var_4 = var_3.__iter__()
    var_5 = list(var_4)
    var_6 = 'Error message'
    var_7 = 'error_code'
    var_8 = module_0.Message(text=var_6, code=var_7)
    var_9 = [var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = module_0.ValidationResult(error=var_10)
    var_12 = var_11.__iter__()
    var_13 = list(var_12)
    var_14 = module_0.ValidationResult(value=var_2, error=var_10)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = module_0.Position(var_0, var_1, var_2)
    var_6 = 4
    var_7 = module_0.Position(var_0, var_1, var_6)
    var_8 = module_0.Position(var_0, var_1, var_2)
    var_9 = module_0.Position(var_0, var_2, var_2)
    var_10 = module_0.Position(var_0, var_1, var_2)
    var_11 = module_0.Position(var_1, var_1, var_2)
    var_12 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'valid data'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'error message'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 4
    var_6 = module_0.Position(var_0, var_1, var_5)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = module_0.Position(var_0, var_1, var_2)
    var_6 = 4
    var_7 = module_0.Position(var_0, var_1, var_6)
    var_8 = var_5 == var_7
    var_9 = module_0.Position(var_0, var_1, var_2)
    var_10 = module_0.Position(var_0, var_2, var_2)
    var_11 = var_9 == var_10
    var_12 = module_0.Position(var_0, var_1, var_2)
    var_13 = module_0.Position(var_1, var_1, var_2)
    var_14 = var_12 == var_13
    var_15 = module_0.Position(var_0, var_1, var_2)
    var_16 = 'not a Position'
    var_17 = var_15 == var_16



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_4]
    var_6 = 'error1'
    var_7 = 'code1'
    var_8 = 'key1'
    var_9 = module_0.Message(text=var_6, code=var_7, key=var_8)
    var_10 = 'error2'
    var_11 = 'code2'
    var_12 = 'key2'
    var_13 = module_0.Message(text=var_10, code=var_11, key=var_12)
    var_14 = [var_9, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    var_16 = 'subkey1'
    var_17 = [var_8, var_16]
    var_18 = module_0.Message(text=var_6, code=var_7, index=var_17)
    var_19 = 'subkey2'
    var_20 = [var_12, var_19]
    var_21 = module_0.Message(text=var_10, code=var_11, index=var_20)
    var_22 = [var_18, var_21]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = 1
    var_25 = 0
    var_26 = module_0.Position(var_24, var_24, var_25)
    var_27 = module_0.BaseError(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_29 = [var_28]



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



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_5, start_position=var_5, end_position=var_5)
    var_7 = [var_3]
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7, position=var_5, start_position=var_5, end_position=var_5)
    var_9 = [var_3]
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_9, position=var_5, start_position=var_5, end_position=var_5)
    var_11 = [var_3]
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_11, position=var_5, start_position=var_5, end_position=var_5)
    var_13 = [var_3]
    var_14 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_13, position=var_5, start_position=var_5, end_position=var_5)
    var_15 = [var_3]
    var_16 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_15, position=var_5, start_position=var_5, end_position=var_5)
    var_17 = [var_3]
    var_18 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_17, position=var_5, start_position=var_5, end_position=var_5)
    var_19 = [var_3]
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_19, position=var_5, start_position=var_5, end_position=var_5)
    var_21 = [var_3]
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_21, position=var_5, start_position=var_5, end_position=var_5)
    var_23 = [var_3]
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_5, start_position=var_5, end_position=var_5)
    var_25 = [var_3]
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_25, position=var_5, start_position=var_5, end_position=var_5)
    var_27 = [var_3]
    var_28 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_27, position=var_5, start_position=var_5, end_position=var_5)
    var_29 = [var_3]
    var_30 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_29, position=var_5, start_position=var_5, end_position=var_5)
    var_31 = [var_3]
    var_32 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_31, position=var_5, start_position=var_5, end_position=var_5)
    var_33 = [var_3]
    var_34 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_33, position=var_5, start_position=var_5, end_position=var_5)
    var_35 = [var_3]
    var_36 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_35, position=var_5, start_position=var_5, end_position=var_5)
    var_37 = [var_3]
    var_38 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_37, position=var_5, start_position=var_5, end_position=var_5)
    var_39 = [var_3]
    var_40 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_39, position=var_5, start_position=var_5, end_position=var_5)
    var_41 = [var_3]
    var_42 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_41, position=var_5, start_position=var_5, end_position=var_5)
    var_43 = [var_3]
    var_44 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_43, position=var_5, start_position=var_5, end_position=var_5)
    var_45 = [var_3]
    var_46 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_45, position=var_5, start_position=var_5, end_position=var_5)
    var_47 = [var_3]
    var_48 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_47, position=var_5, start_position=var_5, end_position=var_5)
    var_49 = [var_3]
    var_50 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_49, position=var_5, start_position=var_5, end_position=var_5)
    var_51 = [var_3]
    var_52 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_51, position=var_5, start_position=var_5, end_position=var_5)
    var_53 = [var_3]
    var_54 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_53, position=var_5, start_position=var_5, end_position=var_5)
    var_55 = [var_3]
    var_56 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_55, position=var_5, start_position=var_5, end_position=var_5)
    var_57 = [var_3]
    var_58 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_57, position=var_5, start_position=var_5, end_position=var_5)
    var_59 = [var_3]
    var_60 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_59, position=var_5, start_position=var_5, end_position=var_5)
    var_61 = [var_3]
    var_62 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_61, position=var_5, start_position=var_5, end_position=var_5)
    var_63 = [var_3]
    var_64 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_63, position=var_5, start_position=var_5, end_position=var_5)



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    var_2 = 2
    var_3 = module_0.Position(var_2, var_2, var_2)
    var_4 = 'Error message'
    var_5 = 'custom'
    var_6 = 'key'
    var_7 = 'index'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, key=var_6, index=var_8, start_position=var_1, end_position=var_3)
    var_10 = [var_7]
    var_11 = module_0.Message(text=var_4, code=var_5, key=var_6, index=var_10, start_position=var_1, end_position=var_3)
    var_12 = 'Different message'
    var_13 = [var_7]
    var_14 = module_0.Message(text=var_12, code=var_5, key=var_6, index=var_13, start_position=var_1, end_position=var_3)
    var_15 = 'different'
    var_16 = [var_7]
    var_17 = module_0.Message(text=var_4, code=var_15, key=var_6, index=var_16, start_position=var_1, end_position=var_3)
    var_18 = [var_7]
    var_19 = module_0.Message(text=var_4, code=var_5, key=var_15, index=var_18, start_position=var_1, end_position=var_3)
    var_20 = [var_15]
    var_21 = module_0.Message(text=var_4, code=var_5, key=var_6, index=var_20, start_position=var_1, end_position=var_3)
    var_22 = [var_7]
    var_23 = module_0.Message(text=var_4, code=var_5, key=var_6, index=var_22, start_position=var_3, end_position=var_1)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_4)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_7, position=var_8)
    var_10 = 'Different'
    var_11 = [var_2]
    var_12 = module_0.Position(var_4, var_4, var_4)
    var_13 = module_0.Message(text=var_10, code=var_1, key=var_2, index=var_11, position=var_12)
    var_14 = 'different'
    var_15 = [var_2]
    var_16 = module_0.Position(var_4, var_4, var_4)
    var_17 = module_0.Message(text=var_0, code=var_14, key=var_2, index=var_15, position=var_16)
    var_18 = [var_14]
    var_19 = module_0.Position(var_4, var_4, var_4)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_18, position=var_19)
    var_21 = [var_2]
    var_22 = 2
    var_23 = module_0.Position(var_22, var_22, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_21, position=var_23)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Different Error'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)
    var_10 = 1
    var_11 = module_0.Position(var_10, var_10, var_10)
    var_12 = 2
    var_13 = module_0.Position(var_12, var_12, var_12)
    var_14 = [var_2]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14, start_position=var_11, end_position=var_11)
    var_16 = [var_2]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16, start_position=var_13, end_position=var_13)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 'different error'
    var_10 = [var_2]
    var_11 = module_0.Message(text=var_9, code=var_1, index=var_10)
    var_12 = [var_2]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different Error'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'name'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different Error'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    var_7 = 'other_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    var_9 = 'other_name'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = module_0.Position(var_13, var_13, var_13)
    var_15 = 2
    var_16 = module_0.Position(var_15, var_15, var_15)
    var_17 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_14)
    var_18 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_16)
    var_19 = module_0.Message(text=var_0, code=var_1, key=var_2, end_position=var_14)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, end_position=var_16)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = 'different_index'
    var_20 = [var_19]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_21)
    var_23 = [var_3]
    var_24 = 2
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_25)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'index'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Position(var_4, var_4, var_4)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_4)
    var_10 = module_0.Position(var_4, var_4, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_8, start_position=var_9, end_position=var_10)
    var_12 = 'error1'
    var_13 = [var_2]
    var_14 = module_0.Position(var_4, var_4, var_4)
    var_15 = module_0.Position(var_4, var_4, var_4)
    var_16 = module_0.Message(text=var_12, code=var_1, index=var_13, start_position=var_14, end_position=var_15)
    var_17 = 'error2'
    var_18 = [var_2]
    var_19 = module_0.Position(var_4, var_4, var_4)
    var_20 = module_0.Position(var_4, var_4, var_4)
    var_21 = module_0.Message(text=var_17, code=var_1, index=var_18, start_position=var_19, end_position=var_20)
    var_22 = 'code1'
    var_23 = [var_2]
    var_24 = module_0.Position(var_4, var_4, var_4)
    var_25 = module_0.Position(var_4, var_4, var_4)
    var_26 = module_0.Message(text=var_0, code=var_22, index=var_23, start_position=var_24, end_position=var_25)
    var_27 = 'code2'
    var_28 = [var_2]
    var_29 = module_0.Position(var_4, var_4, var_4)
    var_30 = module_0.Position(var_4, var_4, var_4)
    var_31 = module_0.Message(text=var_0, code=var_27, index=var_28, start_position=var_29, end_position=var_30)
    var_32 = 'index1'
    var_33 = [var_32]
    var_34 = module_0.Position(var_4, var_4, var_4)
    var_35 = module_0.Position(var_4, var_4, var_4)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_33, start_position=var_34, end_position=var_35)
    var_37 = 'index2'
    var_38 = [var_37]
    var_39 = module_0.Position(var_4, var_4, var_4)
    var_40 = module_0.Position(var_4, var_4, var_4)
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_38, start_position=var_39, end_position=var_40)
    var_42 = [var_2]
    var_43 = module_0.Position(var_4, var_4, var_4)
    var_44 = module_0.Position(var_4, var_4, var_4)
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_42, start_position=var_43, end_position=var_44)
    var_46 = [var_2]
    var_47 = 2
    var_48 = module_0.Position(var_47, var_47, var_47)
    var_49 = module_0.Position(var_4, var_4, var_4)
    var_50 = module_0.Message(text=var_0, code=var_1, index=var_46, start_position=var_48, end_position=var_49)
    var_51 = [var_2]
    var_52 = module_0.Position(var_4, var_4, var_4)
    var_53 = module_0.Position(var_4, var_4, var_4)
    var_54 = module_0.Message(text=var_0, code=var_1, index=var_51, start_position=var_52, end_position=var_53)
    var_55 = [var_2]
    var_56 = module_0.Position(var_4, var_4, var_4)
    var_57 = module_0.Position(var_47, var_47, var_47)
    var_58 = module_0.Message(text=var_0, code=var_1, index=var_55, start_position=var_56, end_position=var_57)
    var_59 = [var_2]
    var_60 = module_0.Position(var_4, var_4, var_4)
    var_61 = module_0.Position(var_4, var_4, var_4)
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_59, start_position=var_60, end_position=var_61)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Another error message'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)
    var_10 = 'another_code'
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_10, index=var_11)
    var_13 = 'another_key'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = module_0.Position(var_5, var_6, var_7)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_8)
    var_10 = [var_3]
    var_11 = module_0.Position(var_5, var_6, var_7)
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_10, position=var_11)
    var_13 = 'Different message'
    var_14 = [var_3]
    var_15 = module_0.Position(var_5, var_6, var_7)
    var_16 = module_0.Message(text=var_13, code=var_1, key=var_2, index=var_14, position=var_15)
    var_17 = [var_3]
    var_18 = 4
    var_19 = module_0.Position(var_6, var_7, var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_17, position=var_19)
    var_21 = 'different_index'
    var_22 = [var_21]
    var_23 = module_0.Position(var_5, var_6, var_7)
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_22, position=var_23)
    var_25 = 'different_code'
    var_26 = [var_3]
    var_27 = module_0.Position(var_5, var_6, var_7)
    var_28 = module_0.Message(text=var_0, code=var_25, key=var_2, index=var_26, position=var_27)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Different Error'
    var_8 = [var_2]
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8)
    var_10 = 'different_code'
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_0, code=var_10, index=var_11)
    var_13 = 'different_key'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'custom'
    var_2 = 'username'
    var_3 = 'users'
    var_4 = 3
    var_5 = [var_3, var_4, var_2]
    var_6 = 1
    var_7 = module_0.Position(var_6, var_6, var_6)
    var_8 = module_0.Position(var_6, var_6, var_6)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_5, start_position=var_7, end_position=var_8)
    var_10 = [var_3, var_4, var_2]
    var_11 = module_0.Position(var_6, var_6, var_6)
    var_12 = module_0.Position(var_6, var_6, var_6)
    var_13 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_10, start_position=var_11, end_position=var_12)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = 'custom'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Position(var_2, var_2, var_2)
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_4)
    var_6 = [var_2]
    var_7 = module_0.Position(var_2, var_2, var_2)
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_6, position=var_7)
    var_9 = 'different'
    var_10 = [var_2]
    var_11 = module_0.Position(var_2, var_2, var_2)
    var_12 = module_0.Message(text=var_9, code=var_1, index=var_10, position=var_11)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Position(var_4, var_4, var_4)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_4)
    var_10 = module_0.Position(var_4, var_4, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_8, start_position=var_9, end_position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_4, var_4, var_4)
    var_14 = module_0.Position(var_4, var_4, var_4)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_12, start_position=var_13, end_position=var_14)
    var_16 = [var_4]
    var_17 = module_0.Position(var_4, var_4, var_4)
    var_18 = module_0.Position(var_4, var_4, var_4)
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_16, start_position=var_17, end_position=var_18)
    var_20 = [var_2]
    var_21 = module_0.Position(var_4, var_4, var_4)
    var_22 = module_0.Position(var_4, var_4, var_4)
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_20, start_position=var_21, end_position=var_22)
    var_24 = [var_2]
    var_25 = 2
    var_26 = module_0.Position(var_4, var_4, var_25)
    var_27 = module_0.Position(var_4, var_4, var_25)
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_24, start_position=var_26, end_position=var_27)
    var_29 = [var_2]
    var_30 = module_0.Position(var_4, var_4, var_4)
    var_31 = module_0.Position(var_4, var_4, var_4)
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_29, start_position=var_30, end_position=var_31)
    var_33 = [var_2]
    var_34 = module_0.Position(var_4, var_4, var_4)
    var_35 = module_0.Position(var_4, var_4, var_4)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_33, start_position=var_34, end_position=var_35)
    var_37 = [var_2]
    var_38 = module_0.Position(var_4, var_4, var_4)
    var_39 = module_0.Position(var_4, var_4, var_4)
    var_40 = module_0.Message(text=var_0, code=var_1, index=var_37, start_position=var_38, end_position=var_39)
    var_41 = [var_2]
    var_42 = module_0.Position(var_4, var_4, var_4)
    var_43 = module_0.Position(var_4, var_4, var_4)
    var_44 = module_0.Message(text=var_0, code=var_1, index=var_41, start_position=var_42, end_position=var_43)
    var_45 = [var_2]
    var_46 = module_0.Position(var_4, var_4, var_4)
    var_47 = module_0.Position(var_4, var_4, var_4)
    var_48 = module_0.Message(text=var_0, code=var_1, index=var_45, start_position=var_46, end_position=var_47)
    var_49 = [var_2]
    var_50 = module_0.Position(var_4, var_4, var_4)
    var_51 = module_0.Position(var_4, var_4, var_4)
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_49, start_position=var_50, end_position=var_51)
    var_53 = [var_2]
    var_54 = module_0.Position(var_4, var_4, var_4)
    var_55 = module_0.Position(var_4, var_4, var_4)
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_53, start_position=var_54, end_position=var_55)
    var_57 = [var_2]
    var_58 = module_0.Position(var_4, var_4, var_4)
    var_59 = module_0.Position(var_4, var_4, var_4)
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_57, start_position=var_58, end_position=var_59)
    var_61 = [var_2]
    var_62 = module_0.Position(var_4, var_4, var_4)
    var_63 = module_0.Position(var_4, var_4, var_4)
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_61, start_position=var_62, end_position=var_63)
    var_65 = [var_2]
    var_66 = module_0.Position(var_4, var_4, var_4)
    var_67 = module_0.Position(var_4, var_4, var_4)
    var_68 = module_0.Message(text=var_0, code=var_1, index=var_65, start_position=var_66, end_position=var_67)
    var_69 = [var_2]
    var_70 = module_0.Position(var_4, var_4, var_4)
    var_71 = module_0.Position(var_4, var_4, var_4)
    var_72 = module_0.Message(text=var_0, code=var_1, index=var_69, start_position=var_70, end_position=var_71)
    var_73 = [var_2]
    var_74 = module_0.Position(var_4, var_4, var_4)
    var_75 = module_0.Position(var_4, var_4, var_4)
    var_76 = module_0.Message(text=var_0, code=var_1, index=var_73, start_position=var_74, end_position=var_75)
    var_77 = [var_2]
    var_78 = module_0.Position(var_4, var_4, var_4)
    var_79 = module_0.Position(var_4, var_4, var_4)
    var_80 = module_0.Message(text=var_0, code=var_1, index=var_77, start_position=var_78, end_position=var_79)
    var_81 = [var_2]
    var_82 = module_0.Position(var_4, var_4, var_4)
    var_83 = module_0.Position(var_4, var_4, var_4)
    var_84 = module_0.Message(text=var_0, code=var_1, index=var_81, start_position=var_82, end_position=var_83)
    var_85 = [var_2]
    var_86 = module_0.Position(var_4, var_4, var_4)
    var_87 = module_0.Position(var_4, var_4, var_4)
    var_88 = module_0.Message(text=var_0, code=var_1, index=var_85, start_position=var_86, end_position=var_87)
    var_89 = [var_2]
    var_90 = module_0.Position(var_4, var_4, var_4)
    var_91 = module_0.Position(var_4, var_4, var_4)
    var_92 = module_0.Message(text=var_0, code=var_1, index=var_89, start_position=var_90, end_position=var_91)
    var_93 = [var_2]
    var_94 = module_0.Position(var_4, var_4, var_4)
    var_95 = module_0.Position(var_4, var_4, var_4)
    var_96 = module_0.Message(text=var_0, code=var_1, index=var_93, start_position=var_94, end_position=var_95)
    var_97 = [var_2]
    var_98 = module_0.Position(var_4, var_4, var_4)
    var_99 = module_0.Position(var_4, var_4, var_4)
    var_100 = module_0.Message(text=var_0, code=var_1, index=var_97, start_position=var_98, end_position=var_99)
    var_101 = [var_2]
    var_102 = module_0.Position(var_4, var_4, var_4)
    var_103 = module_0.Position(var_4, var_4, var_4)
    var_104 = module_0.Message(text=var_0, code=var_1, index=var_101, start_position=var_102, end_position=var_103)
    var_105 = [var_2]
    var_106 = module_0.Position(var_4, var_4, var_4)
    var_107 = module_0.Position(var_4, var_4, var_4)
    var_108 = module_0.Message(text=var_0, code=var_1, index=var_105, start_position=var_106, end_position=var_107)
    var_109 = [var_2]
    var_110 = module_0.Position(var_4, var_4, var_4)
    var_111 = module_0.Position(var_4, var_4, var_4)
    var_112 = module_0.Message(text=var_0, code=var_1, index=var_109, start_position=var_110, end_position=var_111)
    var_113 = [var_2]
    var_114 = module_0.Position(var_4, var_4, var_4)
    var_115 = module_0.Position(var_4, var_4, var_4)
    var_116 = module_0.Message(text=var_0, code=var_1, index=var_113, start_position=var_114, end_position=var_115)
    var_117 = [var_2]
    var_118 = module_0.Position(var_4, var_4, var_4)
    var_119 = module_0.Position(var_4, var_4, var_4)
    var_120 = module_0.Message(text=var_0, code=var_1, index=var_117, start_position=var_118, end_position=var_119)
    var_121 = [var_2]
    var_122 = module_0.Position(var_4, var_4, var_4)
    var_123 = module_0.Position(var_4, var_4, var_4)
    var_124 = module_0.Message(text=var_0, code=var_1, index=var_121, start_position=var_122, end_position=var_123)
    var_125 = [var_2]
    var_126 = module_0.Position(var_4, var_4, var_4)
    var_127 = module_0.Position(var_4, var_4, var_4)
    var_128 = module_0.Message(text=var_0, code=var_1, index=var_125, start_position=var_126, end_position=var_127)
    var_129 = [var_2]
    var_130 = module_0.Position(var_4, var_4, var_4)
    var_131 = module_0.Position(var_4, var_4, var_4)
    var_132 = module_0.Message(text=var_0, code=var_1, index=var_129, start_position=var_130, end_position=var_131)
    var_133 = [var_2]
    var_134 = module_0.Position(var_4, var_4, var_4)
    var_135 = module_0.Position(var_4, var_4, var_4)
    var_136 = module_0.Message(text=var_0, code=var_1, index=var_133, start_position=var_134, end_position=var_135)
    var_137 = [var_2]
    var_138 = module_0.Position(var_4, var_4, var_4)
    var_139 = module_0.Position(var_4, var_4, var_4)
    var_140 = module_0.Message(text=var_0, code=var_1, index=var_137, start_position=var_138, end_position=var_139)
    var_141 = [var_2]
    var_142 = module_0.Position(var_4, var_4, var_4)
    var_143 = module_0.Position(var_4, var_4, var_4)
    var_144 = module_0.Message(text=var_0, code=var_1, index=var_141, start_position=var_142, end_position=var_143)
    var_145 = [var_2]
    var_146 = module_0.Position(var_4, var_4, var_4)
    var_147 = module_0.Position(var_4, var_4, var_4)
    var_148 = module_0.Message(text=var_0, code=var_1, index=var_145, start_position=var_146, end_position=var_147)
    var_149 = [var_2]
    var_150 = module_0.Position(var_4, var_4, var_4)
    var_151 = module_0.Position(var_4, var_4, var_4)
    var_152 = module_0.Message(text=var_0, code=var_1, index=var_149, start_position=var_150, end_position=var_151)
    var_153 = [var_2]
    var_154 = module_0.Position(var_4, var_4, var_4)
    var_155 = module_0.Position(var_4, var_4, var_4)
    var_156 = module_0.Message(text=var_0, code=var_1, index=var_153, start_position=var_154, end_position=var_155)
    var_157 = [var_2]
    var_158 = module_0.Position(var_4, var_4, var_4)
    var_159 = module_0.Position(var_4, var_4, var_4)
    var_160 = module_0.Message(text=var_0, code=var_1, index=var_157, start_position=var_158, end_position=var_159)
    var_161 = [var_2]
    var_162 = module_0.Position(var_4, var_4, var_4)
    var_163 = module_0.Position(var_4, var_4, var_4)
    var_164 = module_0.Message(text=var_0, code=var_1, index=var_161, start_position=var_162, end_position=var_163)
    var_165 = [var_2]
    var_166 = module_0.Position(var_4, var_4, var_4)
    var_167 = module_0.Position(var_4, var_4, var_4)
    var_168 = module_0.Message(text=var_0, code=var_1, index=var_165, start_position=var_166, end_position=var_167)
    var_169 = [var_2]
    var_170 = module_0.Position(var_4, var_4, var_4)
    var_171 = module_0.Position(var_4, var_4, var_4)
    var_172 = module_0.Message(text=var_0, code=var_1, index=var_169, start_position=var_170, end_position=var_171)
    var_173 = [var_2]
    var_174 = module_0.Position(var_4, var_4, var_4)
    var_175 = module_0.Position(var_4, var_4, var_4)
    var_176 = module_0.Message(text=var_0, code=var_1, index=var_173, start_position=var_174, end_position=var_175)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different_code'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = 'different_index'
    var_20 = [var_19]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_20, position=var_21)
    var_23 = [var_3]
    var_24 = 2
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_23, position=var_25)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Different message'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'different'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_0, code=var_15, key=var_2, index=var_16, position=var_17)
    var_19 = [var_15]
    var_20 = module_0.Position(var_5, var_5, var_5)
    var_21 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_19, position=var_20)
    var_22 = [var_3]
    var_23 = 2
    var_24 = module_0.Position(var_23, var_23, var_23)
    var_25 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_22, position=var_24)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'custom'
    var_2 = []
    var_3 = None
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_2, start_position=var_3, end_position=var_3)
    var_5 = []
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5, start_position=var_3, end_position=var_3)
    var_7 = 'different'
    var_8 = []
    var_9 = module_0.Message(text=var_7, code=var_1, index=var_8, start_position=var_3, end_position=var_3)
    var_10 = []
    var_11 = module_0.Message(text=var_0, code=var_7, index=var_10, start_position=var_3, end_position=var_3)
    var_12 = 'key'
    var_13 = [var_12]
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_13, start_position=var_3, end_position=var_3)
    var_15 = []
    var_16 = 1
    var_17 = module_0.Position(var_16, var_16, var_16)
    var_18 = module_0.Position(var_16, var_16, var_16)
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_15, start_position=var_17, end_position=var_18)
    var_20 = []
    var_21 = module_0.Position(var_16, var_16, var_16)
    var_22 = module_0.Position(var_16, var_16, var_16)
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_20, start_position=var_21, end_position=var_22)
    var_24 = []
    var_25 = module_0.Position(var_16, var_16, var_16)
    var_26 = 2
    var_27 = module_0.Position(var_16, var_26, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_24, start_position=var_25, end_position=var_27)
    var_29 = module_1.object()
    var_30 = var_4 == var_29



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_4)
    var_6 = module_0.Position(var_4, var_4, var_4)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_4)
    var_10 = module_0.Position(var_4, var_4, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_8, start_position=var_9, end_position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_4, var_4, var_4)
    var_14 = module_0.Position(var_4, var_4, var_4)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_12, start_position=var_13, end_position=var_14)
    var_16 = 'Different Error'
    var_17 = [var_2]
    var_18 = module_0.Position(var_4, var_4, var_4)
    var_19 = module_0.Position(var_4, var_4, var_4)
    var_20 = module_0.Message(text=var_16, code=var_1, index=var_17, start_position=var_18, end_position=var_19)
    var_21 = [var_2]
    var_22 = module_0.Position(var_4, var_4, var_4)
    var_23 = module_0.Position(var_4, var_4, var_4)
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_21, start_position=var_22, end_position=var_23)
    var_25 = 'different_error'
    var_26 = [var_2]
    var_27 = module_0.Position(var_4, var_4, var_4)
    var_28 = module_0.Position(var_4, var_4, var_4)
    var_29 = module_0.Message(text=var_0, code=var_25, index=var_26, start_position=var_27, end_position=var_28)
    var_30 = [var_2]
    var_31 = module_0.Position(var_4, var_4, var_4)
    var_32 = module_0.Position(var_4, var_4, var_4)
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_30, start_position=var_31, end_position=var_32)
    var_34 = 'different_key'
    var_35 = [var_34]
    var_36 = module_0.Position(var_4, var_4, var_4)
    var_37 = module_0.Position(var_4, var_4, var_4)
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_35, start_position=var_36, end_position=var_37)
    var_39 = [var_2]
    var_40 = module_0.Position(var_4, var_4, var_4)
    var_41 = module_0.Position(var_4, var_4, var_4)
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_39, start_position=var_40, end_position=var_41)
    var_43 = [var_2]
    var_44 = 2
    var_45 = module_0.Position(var_44, var_4, var_4)
    var_46 = module_0.Position(var_4, var_4, var_4)
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_43, start_position=var_45, end_position=var_46)
    var_48 = [var_2]
    var_49 = module_0.Position(var_4, var_4, var_4)
    var_50 = module_0.Position(var_4, var_4, var_4)
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_48, start_position=var_49, end_position=var_50)
    var_52 = [var_2]
    var_53 = module_0.Position(var_4, var_4, var_4)
    var_54 = module_0.Position(var_44, var_4, var_4)
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_52, start_position=var_53, end_position=var_54)
    var_56 = [var_2]
    var_57 = module_0.Position(var_4, var_4, var_4)
    var_58 = module_0.Position(var_4, var_4, var_4)
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_56, start_position=var_57, end_position=var_58)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Error'
    var_5 = 'custom'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, position=var_3)
    var_8 = module_0.Message(text=var_4, code=var_5, key=var_6, position=var_3)
    var_9 = 'Different Error'
    var_10 = module_0.Message(text=var_9, code=var_5, key=var_6, position=var_3)



