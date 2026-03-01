####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Error 1'
    var_6 = module_0.Message(text=var_5, code=var_1)
    var_7 = 'Error 2'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'code1'
    var_10 = module_0.Message(text=var_0, code=var_9)
    var_11 = 'code2'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = [var_2]
    var_14 = module_0.Message(text=var_0, index=var_13)
    var_15 = 'other'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, index=var_16)
    var_18 = 1
    var_19 = 0
    var_20 = module_0.Position(var_18, var_18, var_19)
    var_21 = 2
    var_22 = 10
    var_23 = module_0.Position(var_21, var_18, var_22)
    var_24 = module_0.Message(text=var_0, start_position=var_20, end_position=var_20)
    var_25 = module_0.Message(text=var_0, start_position=var_23, end_position=var_20)
    var_26 = module_0.Message(text=var_0, start_position=var_20, end_position=var_20)
    var_27 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_28 = module_0.Message(text=var_0, position=var_20)
    var_29 = module_0.Message(text=var_0, position=var_20)
    var_30 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_31 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_32 = module_0.Message(text=var_0)
    var_33 = None
    var_34 = module_0.Message(text=var_0, index=var_33)
    var_35 = []
    var_36 = module_0.Message(text=var_0, index=var_35)
    var_37 = module_0.Message(text=var_0, key=var_2)
    var_38 = [var_2]
    var_39 = module_0.Message(text=var_0, index=var_38)
    var_40 = module_0.Message(text=var_0, code=var_33)
    var_41 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='test_value')"
    var_3 = 'Error message'
    var_4 = 'custom'
    var_5 = module_0.ValidationError(text=var_3, code=var_4)
    var_6 = module_0.ValidationResult(error=var_5)
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(error=ValidationError(text='Error message', code='custom'))"
    var_8 = 'First error'
    var_9 = 'max_length'
    var_10 = module_0.Message(text=var_8, code=var_9)
    var_11 = 'Second error'
    var_12 = 'min_length'
    var_13 = module_0.Message(text=var_11, code=var_12)
    var_14 = [var_10, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = module_0.ValidationResult(error=var_15)
    var_17 = repr(var_16)
    var_18 = 'ValidationResult(error=ValidationError(['
    var_19 = repr(var_16)
    var_20 = repr(var_16)
    var_21 = 'Field error'
    var_22 = 'username'
    var_23 = module_0.ValidationError(text=var_21, key=var_22)
    var_24 = module_0.ValidationResult(error=var_23)
    var_25 = repr(var_24)
    assert var_25 == "ValidationResult(error=ValidationError([Message(text='Field error', code='custom', index=['username'])]))"
    var_26 = None
    var_27 = module_0.ValidationResult(value=var_26)
    var_28 = repr(var_27)
    assert var_28 == 'ValidationResult(value=None)'



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = hash(var_2)
    var_5 = hash(var_3)
    var_6 = 'Error 1'
    var_7 = 'code1'
    var_8 = 'field1'
    var_9 = module_0.Message(text=var_6, code=var_7, key=var_8)
    var_10 = 'Error 2'
    var_11 = 'code2'
    var_12 = 'field2'
    var_13 = module_0.Message(text=var_10, code=var_11, key=var_12)
    var_14 = [var_9, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = module_0.ValidationError(messages=var_14)
    var_17 = hash(var_15)
    var_18 = hash(var_16)
    var_19 = module_0.ValidationError(text=var_0, code=var_1)
    var_20 = 'Different message'
    var_21 = module_0.ValidationError(text=var_20, code=var_1)
    var_22 = hash(var_19)
    var_23 = hash(var_21)
    var_24 = module_0.Message(text=var_6, code=var_7)
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_6, code=var_7)
    var_27 = module_0.Message(text=var_10, code=var_11)
    var_28 = [var_26, var_27]
    var_29 = module_0.ValidationError(messages=var_25)
    var_30 = module_0.ValidationError(messages=var_28)
    var_31 = hash(var_29)
    var_32 = hash(var_30)
    var_33 = module_0.ValidationError(text=var_0)
    var_34 = module_0.Message(text=var_6, code=var_7, key=var_8)
    var_35 = module_0.Message(text=var_10, code=var_11, key=var_12)
    var_36 = [var_34, var_35]
    var_37 = module_0.Message(text=var_10, code=var_11, key=var_12)
    var_38 = module_0.Message(text=var_6, code=var_7, key=var_8)
    var_39 = [var_37, var_38]
    var_40 = module_0.ValidationError(messages=var_36)
    var_41 = module_0.ValidationError(messages=var_39)
    var_42 = hash(var_40)
    var_43 = hash(var_41)
    var_44 = 1
    var_45 = 0
    var_46 = module_0.Position(var_44, var_44, var_45)
    var_47 = 'Error'
    var_48 = 'code'
    var_49 = module_0.ValidationError(text=var_47, code=var_48, position=var_46)
    var_50 = module_0.ValidationError(text=var_47, code=var_48, position=var_46)
    var_51 = hash(var_49)
    var_52 = hash(var_50)
    var_53 = module_0.ValidationError(text=var_47, code=var_48, key=var_8)
    var_54 = module_0.ValidationError(text=var_47, code=var_48, key=var_12)
    var_55 = hash(var_53)
    var_56 = hash(var_54)
    var_57 = 'Parse error'
    var_58 = 'parse'
    var_59 = module_0.ParseError(text=var_57, code=var_58)
    var_60 = module_0.ValidationError(text=var_57, code=var_58)
    var_61 = module_0.ValidationError(text=var_0)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='test_value')"
    var_3 = 'Invalid input'
    var_4 = 'invalid'
    var_5 = module_0.ValidationError(text=var_3, code=var_4)
    var_6 = module_0.ValidationResult(error=var_5)
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(error=ValidationError(text='Invalid input', code='invalid'))"
    var_8 = 'Field required'
    var_9 = 'required'
    var_10 = 'username'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Too short'
    var_13 = 'min_length'
    var_14 = 'password'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationResult(error=var_17)
    var_19 = 'ValidationResult(error=ValidationError(['
    var_20 = repr(var_18)
    var_21 = ''
    var_22 = module_0.ValidationResult(value=var_21)
    var_23 = repr(var_22)
    assert var_23 == "ValidationResult(value='')"
    var_24 = 42
    var_25 = module_0.ValidationResult(value=var_24)
    var_26 = repr(var_25)
    assert var_26 == 'ValidationResult(value=42)'
    var_27 = 'Invalid email'
    var_28 = 'invalid_email'
    var_29 = 'users'
    var_30 = 0
    var_31 = 'email'
    var_32 = [var_29, var_30, var_31]
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = module_0.Message(text=var_27, code=var_28, index=var_32, position=var_36)
    var_38 = [var_37]
    var_39 = module_0.ValidationError(messages=var_38)
    var_40 = module_0.ValidationResult(error=var_39)
    var_41 = repr(var_40)
    var_42 = repr(var_40)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='test_value')"
    var_3 = 'Invalid input'
    var_4 = 'invalid'
    var_5 = module_0.ValidationError(text=var_3, code=var_4)
    var_6 = module_0.ValidationResult(error=var_5)
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(error=ValidationError(text='Invalid input', code='invalid'))"
    var_8 = 'Field required'
    var_9 = 'required'
    var_10 = 'username'
    var_11 = module_0.ValidationError(text=var_8, code=var_9, key=var_10)
    var_12 = module_0.ValidationResult(error=var_11)
    var_13 = repr(var_12)
    assert var_13 == "ValidationResult(error=ValidationError([Message(text='Field required', code='required', index=['username'])]))"
    var_14 = 'Too short'
    var_15 = 'min_length'
    var_16 = 'password'
    var_17 = module_0.Message(text=var_14, code=var_15, key=var_16)
    var_18 = 'Invalid format'
    var_19 = 'format'
    var_20 = 'email'
    var_21 = module_0.Message(text=var_18, code=var_19, key=var_20)
    var_22 = [var_17, var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = module_0.ValidationResult(error=var_23)
    var_25 = repr(var_24)
    var_26 = 'ValidationResult(error=ValidationError(['
    var_27 = repr(var_24)
    var_28 = repr(var_24)
    var_29 = None
    var_30 = module_0.ValidationResult(value=var_29)
    var_31 = repr(var_30)
    assert var_31 == 'ValidationResult(value=None)'
    var_32 = 'key'
    var_33 = 'number'
    var_34 = 'value'
    var_35 = 42
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.ValidationResult(value=var_36)
    var_38 = repr(var_37)
    assert var_38 == "ValidationResult(value={'key': 'value', 'number': 42})"



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'other'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = module_0.Position(var_21, var_21, var_2)
    var_23 = module_0.Message(text=var_0, position=var_22)
    var_24 = module_0.Message(text=var_0, position=var_22)
    var_25 = module_0.Position(var_21, var_21, var_2)
    var_26 = 2
    var_27 = 10
    var_28 = module_0.Position(var_26, var_21, var_27)
    var_29 = module_0.Message(text=var_0, position=var_25)
    var_30 = module_0.Message(text=var_0, position=var_28)
    var_31 = module_0.Position(var_21, var_21, var_2)
    var_32 = 5
    var_33 = 4
    var_34 = module_0.Position(var_21, var_32, var_33)
    var_35 = module_0.Message(text=var_0, start_position=var_31, end_position=var_34)
    var_36 = module_0.Message(text=var_0, start_position=var_31, end_position=var_34)
    var_37 = module_0.Position(var_21, var_21, var_2)
    var_38 = module_0.Position(var_21, var_32, var_33)
    var_39 = module_0.Position(var_26, var_21, var_27)
    var_40 = 14
    var_41 = module_0.Position(var_26, var_32, var_40)
    var_42 = module_0.Message(text=var_0, start_position=var_37, end_position=var_38)
    var_43 = module_0.Message(text=var_0, start_position=var_39, end_position=var_41)
    var_44 = module_0.Message(text=var_0, key=var_15)
    var_45 = [var_15]
    var_46 = module_0.Message(text=var_0, index=var_45)
    var_47 = module_0.Message(text=var_0)
    var_48 = module_0.Message(text=var_0)
    var_49 = module_0.Message(text=var_0)
    var_50 = module_0.Position(var_21, var_21, var_2)
    var_51 = module_0.Message(text=var_0, position=var_50)
    var_52 = module_0.Message(text=var_0)
    var_53 = 'users'
    var_54 = 'name'
    var_55 = [var_53, var_2, var_54]
    var_56 = module_0.Message(text=var_0, index=var_55)
    var_57 = [var_53, var_2, var_54]
    var_58 = module_0.Message(text=var_0, index=var_57)
    var_59 = [var_53, var_2, var_54]
    var_60 = module_0.Message(text=var_0, index=var_59)
    var_61 = [var_53, var_21, var_54]
    var_62 = module_0.Message(text=var_0, index=var_61)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Invalid input'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 42
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = []
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.ValidationResult(value=var_10)
    var_12 = None
    var_13 = module_0.ValidationResult(value=var_12)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Error message'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 42
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = 'Error 1'
    var_8 = 'field1'
    var_9 = module_0.Message(text=var_7, key=var_8)
    var_10 = 'Error 2'
    var_11 = 'field2'
    var_12 = module_0.Message(text=var_10, key=var_11)
    var_13 = [var_9, var_12]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = module_0.ValidationResult(error=var_14)
    var_16 = 'key'
    var_17 = 'list'
    var_18 = 'value'
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = {var_16: var_18, var_17: var_22}
    var_24 = module_0.ValidationResult(value=var_23)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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
    var_9 = 10
    var_10 = 20
    var_11 = 30
    var_12 = module_0.Position(var_9, var_10, var_11)
    var_13 = module_0.Position(var_9, var_10, var_11)
    var_14 = 100
    var_15 = 200
    var_16 = 300
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = -1
    var_19 = -2
    var_20 = -3
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = -1
    var_23 = -2
    var_24 = -3
    var_25 = module_0.Position(var_22, var_23, var_24)
    var_26 = 0
    var_27 = module_0.Position(var_26, var_26, var_26)
    var_28 = module_0.Position(var_26, var_26, var_26)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'
    var_4 = 'Field required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.ValidationError(text=var_4, code=var_5, key=var_6)
    var_8 = str(var_7)
    assert var_8 == "{'username': 'Field required'}"
    var_9 = 'Invalid email'
    var_10 = 'email'
    var_11 = module_0.Message(text=var_9, code=var_1, key=var_10)
    var_12 = 'Too short'
    var_13 = 'min_length'
    var_14 = 'password'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = str(var_17)
    assert var_18 == "{'email': 'Invalid email', 'password': 'Too short'}"
    var_19 = 'users'
    var_20 = 0
    var_21 = [var_19, var_20, var_10]
    var_22 = module_0.Message(text=var_0, code=var_1, index=var_21)
    var_23 = 'Required field'
    var_24 = 1
    var_25 = 'name'
    var_26 = [var_19, var_24, var_25]
    var_27 = module_0.Message(text=var_23, code=var_5, index=var_26)
    var_28 = [var_22, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = "{'users': {0: {'email': 'Invalid value'}, 1: {'name': 'Required field'}}}"
    var_31 = str(var_29)
    var_32 = 'Invalid format'
    var_33 = 'format'
    var_34 = module_0.Message(text=var_32, code=var_33, key=var_10)
    var_35 = 'Too long'
    var_36 = 'max_length'
    var_37 = 'profile'
    var_38 = 'bio'
    var_39 = [var_37, var_38]
    var_40 = module_0.Message(text=var_35, code=var_36, index=var_39)
    var_41 = [var_34, var_40]
    var_42 = module_0.ValidationError(messages=var_41)
    var_43 = "{'email': 'Invalid format', 'profile': {'bio': 'Too long'}}"
    var_44 = str(var_42)
    var_45 = 'Root error'
    var_46 = 'root_error'
    var_47 = []
    var_48 = module_0.Message(text=var_45, code=var_46, index=var_47)
    var_49 = 'Field error'
    var_50 = 'field_error'
    var_51 = 'field'
    var_52 = module_0.Message(text=var_49, code=var_50, key=var_51)
    var_53 = [var_48, var_52]
    var_54 = module_0.ValidationError(messages=var_53)
    var_55 = str(var_54)
    assert var_55 == "{'': 'Root error', 'field': 'Field error'}"
    var_56 = 'Parse failed'
    var_57 = 'parse_error'
    var_58 = module_0.ParseError(text=var_56, code=var_57)
    var_59 = str(var_58)
    assert var_59 == 'Parse failed'
    var_60 = 5
    var_61 = 4
    var_62 = module_0.Position(var_24, var_60, var_61)
    var_63 = 'Error with position'
    var_64 = 'error'
    var_65 = module_0.ValidationError(text=var_63, code=var_64, position=var_62)
    var_66 = str(var_65)
    assert var_66 == 'Error with position'



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Error message'
    var_3 = 'custom'
    var_4 = module_0.ValidationError(text=var_2, code=var_3)
    var_5 = module_0.ValidationResult(error=var_4)
    var_6 = 42
    var_7 = module_0.ValidationResult(value=var_6)
    var_8 = 'Error 1'
    var_9 = 'error1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'error2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationResult(error=var_17)
    var_19 = module_0.ValidationResult()
    var_20 = iter(var_19)
    var_21 = next(var_20)
    var_22 = next(var_20)
    var_23 = next(var_20)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Error message'
    var_3 = 'custom'
    var_4 = module_0.ValidationError(text=var_2, code=var_3)
    var_5 = module_0.ValidationResult(error=var_4)
    var_6 = 42
    var_7 = module_0.ValidationResult(value=var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.ValidationResult(value=var_10)
    var_12 = None
    var_13 = module_0.ValidationResult(value=var_12)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'
    var_4 = 'Field is required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.BaseError(text=var_4, code=var_5, key=var_6)
    var_8 = str(var_7)
    assert var_8 == "{'username': 'Field is required'}"
    var_9 = 'Invalid email'
    var_10 = 'email'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_9, code=var_1, index=var_11)
    var_13 = 'Too short'
    var_14 = 'min_length'
    var_15 = 'password'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = str(var_19)
    assert var_20 == "{'email': 'Invalid email', 'password': 'Too short'}"
    var_21 = 'users'
    var_22 = 0
    var_23 = [var_21, var_22, var_10]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    var_25 = 'Required field'
    var_26 = 1
    var_27 = 'name'
    var_28 = [var_21, var_26, var_27]
    var_29 = module_0.Message(text=var_25, code=var_5, index=var_28)
    var_30 = [var_24, var_29]
    var_31 = module_0.BaseError(messages=var_30)
    var_32 = "{'users': {0: {'email': 'Invalid value'}, 1: {'name': 'Required field'}}}"
    var_33 = str(var_31)
    var_34 = 'Root error'
    var_35 = 'root_error'
    var_36 = []
    var_37 = module_0.Message(text=var_34, code=var_35, index=var_36)
    var_38 = 'Another error'
    var_39 = 'another'
    var_40 = 'field'
    var_41 = [var_40]
    var_42 = module_0.Message(text=var_38, code=var_39, index=var_41)
    var_43 = [var_37, var_42]
    var_44 = module_0.BaseError(messages=var_43)
    var_45 = str(var_44)
    assert var_45 == "{'': 'Root error', 'field': 'Another error'}"
    var_46 = 5
    var_47 = 4
    var_48 = module_0.Position(var_26, var_46, var_47)
    var_49 = 'Error with position'
    var_50 = 'error'
    var_51 = module_0.BaseError(text=var_49, code=var_50, position=var_48)
    var_52 = str(var_51)
    assert var_52 == 'Error with position'
    var_53 = 'Error 1'
    var_54 = 'err1'
    var_55 = 'a'
    var_56 = [var_55, var_22]
    var_57 = module_0.Message(text=var_53, code=var_54, index=var_56)
    var_58 = 'Error 2'
    var_59 = 'err2'
    var_60 = 'b'
    var_61 = 'key'
    var_62 = [var_60, var_61]
    var_63 = module_0.Message(text=var_58, code=var_59, index=var_62)
    var_64 = [var_57, var_63]
    var_65 = module_0.BaseError(messages=var_64)
    var_66 = "{'a': {0: 'Error 1'}, 'b': {'key': 'Error 2'}}"
    var_67 = str(var_65)



# Parsed testcases at query #15
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11, start_position=var_3, end_position=var_7)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = [var_12, var_17]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = [var_17, var_12]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = 'Error 3'
    var_25 = 'code3'
    var_26 = 'field3'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_24, code=var_25, index=var_27)
    var_29 = [var_12, var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = [var_12]
    var_32 = module_0.BaseError(messages=var_31)
    var_33 = 'Single error'
    var_34 = 'single'
    var_35 = 'field'
    var_36 = module_0.BaseError(text=var_33, code=var_34, key=var_35)
    var_37 = module_0.BaseError(text=var_33, code=var_34, key=var_35)
    var_38 = 'Different error'
    var_39 = module_0.BaseError(text=var_38, code=var_34, key=var_35)
    var_40 = 'different'
    var_41 = module_0.BaseError(text=var_33, code=var_40, key=var_35)
    var_42 = module_0.BaseError(text=var_33, code=var_34, key=var_40)
    var_43 = module_0.Position(var_0, var_1, var_2)
    var_44 = module_0.Position(var_4, var_5, var_6)
    var_45 = [var_10]
    var_46 = module_0.Message(text=var_8, code=var_9, index=var_45, start_position=var_43, end_position=var_44)
    var_47 = [var_46, var_17]
    var_48 = module_0.BaseError(messages=var_47)
    var_49 = 7
    var_50 = 8
    var_51 = 9
    var_52 = module_0.Position(var_49, var_50, var_51)
    var_53 = [var_10]
    var_54 = module_0.Message(text=var_8, code=var_9, index=var_53, start_position=var_52, end_position=var_7)
    var_55 = [var_54, var_17]
    var_56 = module_0.BaseError(messages=var_55)
    var_57 = hash(var_19)
    var_58 = hash(var_21)
    var_59 = [var_12, var_17]
    var_60 = module_0.ValidationError(messages=var_59)
    var_61 = [var_12, var_17]
    var_62 = module_0.ValidationError(messages=var_61)
    var_63 = [var_12, var_17]
    var_64 = module_0.ParseError(messages=var_63)
    var_65 = [var_12, var_17]
    var_66 = module_0.ParseError(messages=var_65)



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = module_0.Position(var_3, var_4, var_5)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_8)
    var_10 = [var_2]
    var_11 = module_0.Position(var_3, var_4, var_5)
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_10, position=var_11)
    var_13 = [var_2]
    var_14 = module_0.Position(var_3, var_4, var_5)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_13, position=var_14)
    var_16 = [var_2]
    var_17 = module_0.Position(var_3, var_4, var_5)
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_3, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_16, start_position=var_17, end_position=var_20)
    var_22 = [var_2]
    var_23 = module_0.Position(var_3, var_4, var_5)
    var_24 = module_0.Position(var_3, var_18, var_19)
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_22, start_position=var_23, end_position=var_24)
    var_26 = 'Error 1'
    var_27 = module_0.Message(text=var_26, code=var_1, key=var_2)
    var_28 = 'Error 2'
    var_29 = module_0.Message(text=var_28, code=var_1, key=var_2)
    var_30 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_31 = 'required'
    var_32 = module_0.Message(text=var_0, code=var_31, key=var_2)
    var_33 = 'field1'
    var_34 = [var_33]
    var_35 = module_0.Message(text=var_0, code=var_1, index=var_34)
    var_36 = 'field2'
    var_37 = [var_36]
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_37)
    var_39 = module_0.Position(var_3, var_4, var_5)
    var_40 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_39)
    var_41 = 4
    var_42 = module_0.Position(var_4, var_5, var_41)
    var_43 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_42)
    var_44 = [var_2]
    var_45 = module_0.Position(var_3, var_4, var_5)
    var_46 = module_0.Position(var_3, var_18, var_19)
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_44, start_position=var_45, end_position=var_46)
    var_48 = [var_2]
    var_49 = module_0.Position(var_3, var_4, var_5)
    var_50 = 7
    var_51 = module_0.Position(var_3, var_19, var_50)
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_48, start_position=var_49, end_position=var_51)
    var_53 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_54 = module_0.Position(var_3, var_4, var_5)
    var_55 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_54)
    var_56 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_57 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_58 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_59 = module_0.Message(text=var_0, key=var_2)
    var_60 = 'custom'
    var_61 = module_0.Message(text=var_0, code=var_60, key=var_2)
    var_62 = module_0.Message(text=var_0, code=var_1)
    var_63 = []
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)
    var_65 = module_0.Position(var_3, var_4, var_5)
    var_66 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_65)
    var_67 = module_0.Position(var_3, var_4, var_5)
    var_68 = module_0.Position(var_3, var_4, var_5)
    var_69 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_67, end_position=var_68)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Error 1'
    var_5 = 'code1'
    var_6 = 'field1'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_8 = 'Error 2'
    var_9 = 'code2'
    var_10 = 'field2'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = [var_7, var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = [var_7, var_11]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = 'Error message 1'
    var_17 = module_0.ValidationError(text=var_16, code=var_1)
    var_18 = 'Error message 2'
    var_19 = module_0.ValidationError(text=var_18, code=var_1)
    var_20 = module_0.ValidationError(text=var_0, code=var_5)
    var_21 = module_0.ValidationError(text=var_0, code=var_9)
    var_22 = module_0.ValidationError(text=var_0, code=var_1)
    var_23 = module_0.ParseError(text=var_0, code=var_1)
    var_24 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = [var_24, var_25]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = [var_25, var_24]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Error'
    var_31 = 'code'
    var_32 = 'users'
    var_33 = 0
    var_34 = 'name'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.Message(text=var_30, code=var_31, index=var_35)
    var_37 = [var_32, var_33, var_34]
    var_38 = module_0.Message(text=var_30, code=var_31, index=var_37)
    var_39 = [var_36]
    var_40 = module_0.ValidationError(messages=var_39)
    var_41 = [var_38]
    var_42 = module_0.ValidationError(messages=var_41)
    var_43 = [var_32, var_33, var_34]
    var_44 = module_0.Message(text=var_30, code=var_31, index=var_43)
    var_45 = 1
    var_46 = [var_32, var_45, var_34]
    var_47 = module_0.Message(text=var_30, code=var_31, index=var_46)
    var_48 = [var_44]
    var_49 = module_0.ValidationError(messages=var_48)
    var_50 = [var_47]
    var_51 = module_0.ValidationError(messages=var_50)
    var_52 = 5
    var_53 = 10
    var_54 = module_0.Position(var_45, var_52, var_53)
    var_55 = module_0.Message(text=var_30, code=var_31, position=var_54)
    var_56 = module_0.Message(text=var_30, code=var_31, position=var_54)
    var_57 = [var_55]
    var_58 = module_0.ValidationError(messages=var_57)
    var_59 = [var_56]
    var_60 = module_0.ValidationError(messages=var_59)
    var_61 = module_0.Position(var_45, var_52, var_53)
    var_62 = 2
    var_63 = 20
    var_64 = module_0.Position(var_62, var_52, var_63)
    var_65 = module_0.Message(text=var_30, code=var_31, position=var_61)
    var_66 = module_0.Message(text=var_30, code=var_31, position=var_64)
    var_67 = [var_65]
    var_68 = module_0.ValidationError(messages=var_67)
    var_69 = [var_66]
    var_70 = module_0.ValidationError(messages=var_69)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 1
    var_10 = 0
    var_11 = module_0.Position(var_9, var_9, var_10)
    var_12 = module_0.Message(text=var_0, code=var_1, position=var_11)
    var_13 = module_0.Message(text=var_0, code=var_1, position=var_11)
    var_14 = module_0.Position(var_9, var_9, var_10)
    var_15 = 5
    var_16 = 4
    var_17 = module_0.Position(var_9, var_15, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_17)
    var_20 = 'Error 1'
    var_21 = module_0.Message(text=var_20, code=var_1)
    var_22 = 'Error 2'
    var_23 = module_0.Message(text=var_22, code=var_1)
    var_24 = 'code1'
    var_25 = module_0.Message(text=var_0, code=var_24)
    var_26 = 'code2'
    var_27 = module_0.Message(text=var_0, code=var_26)
    var_28 = 'field1'
    var_29 = [var_28]
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_29)
    var_31 = 'field2'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)
    var_34 = module_0.Position(var_9, var_9, var_10)
    var_35 = 2
    var_36 = 10
    var_37 = module_0.Position(var_35, var_9, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, position=var_34)
    var_39 = module_0.Message(text=var_0, code=var_1, position=var_37)
    var_40 = module_0.Position(var_9, var_9, var_10)
    var_41 = module_0.Position(var_35, var_9, var_36)
    var_42 = module_0.Position(var_9, var_15, var_16)
    var_43 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_42)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_41, end_position=var_42)
    var_45 = module_0.Position(var_9, var_9, var_10)
    var_46 = module_0.Position(var_9, var_15, var_16)
    var_47 = 9
    var_48 = module_0.Position(var_9, var_36, var_47)
    var_49 = module_0.Message(text=var_0, code=var_1, start_position=var_45, end_position=var_46)
    var_50 = module_0.Message(text=var_0, code=var_1, start_position=var_45, end_position=var_48)
    var_51 = module_0.Message(text=var_0, code=var_1)
    var_52 = module_0.Message(text=var_0)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = module_0.Message(text=var_0, code=var_1)
    var_55 = []
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_55)
    var_57 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_58 = [var_2]
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58)
    var_60 = module_0.Position(var_9, var_9, var_10)
    var_61 = module_0.Message(text=var_0, code=var_1, position=var_60)
    var_62 = module_0.Message(text=var_0, code=var_1, start_position=var_60, end_position=var_60)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = 'Error 1'
    var_6 = module_0.Message(text=var_5, code=var_1)
    var_7 = 'Error 2'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'max_length'
    var_10 = module_0.Message(text=var_0, code=var_9)
    var_11 = 'min_length'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'users'
    var_14 = 0
    var_15 = [var_13, var_14]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = [var_13, var_17]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = module_0.Position(var_17, var_17, var_14)
    var_21 = 2
    var_22 = 10
    var_23 = module_0.Position(var_21, var_17, var_22)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_26 = module_0.Position(var_17, var_17, var_14)
    var_27 = 5
    var_28 = 4
    var_29 = module_0.Position(var_17, var_27, var_28)
    var_30 = module_0.Position(var_17, var_17, var_14)
    var_31 = 6
    var_32 = module_0.Position(var_17, var_31, var_27)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_26, end_position=var_29)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_32)
    var_35 = module_0.Message(text=var_0, code=var_1)
    var_36 = module_0.Message(text=var_0, code=var_1)
    var_37 = module_0.Position(var_17, var_17, var_14)
    var_38 = module_0.Position(var_17, var_27, var_28)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_37, end_position=var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, start_position=var_37, end_position=var_38)
    var_41 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_42 = [var_2]
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_42)
    var_44 = module_0.Message(text=var_0, code=var_1)
    var_45 = module_0.Position(var_17, var_17, var_14)
    var_46 = module_0.Message(text=var_0, code=var_1, position=var_45)
    var_47 = module_0.Message(text=var_0, code=var_1, position=var_45)
    var_48 = []
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_48)
    var_50 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #20
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11, start_position=var_3, end_position=var_7)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = [var_12, var_17]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = [var_17, var_12]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = 'Error 3'
    var_25 = 'code3'
    var_26 = 'field3'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_24, code=var_25, index=var_27)
    var_29 = [var_12, var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = 'Single error'
    var_32 = 'single'
    var_33 = module_0.BaseError(text=var_31, code=var_32)
    var_34 = module_0.BaseError(text=var_31, code=var_32)
    var_35 = 'Different error'
    var_36 = module_0.BaseError(text=var_35, code=var_32)
    var_37 = [var_12, var_17]
    var_38 = module_0.ValidationError(messages=var_37)
    var_39 = [var_12, var_17]
    var_40 = module_0.ValidationError(messages=var_39)
    var_41 = [var_12, var_17]
    var_42 = module_0.ParseError(messages=var_41)
    var_43 = [var_12, var_17]
    var_44 = module_0.ParseError(messages=var_43)
    var_45 = hash(var_19)
    var_46 = hash(var_21)
    var_47 = hash(var_33)
    var_48 = hash(var_34)
    var_49 = hash(var_19)
    var_50 = hash(var_30)
    var_51 = hash(var_33)
    var_52 = hash(var_36)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'other'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = module_0.Position(var_21, var_21, var_2)
    var_23 = module_0.Message(text=var_0, position=var_22)
    var_24 = module_0.Message(text=var_0, position=var_22)
    var_25 = module_0.Position(var_21, var_21, var_2)
    var_26 = 2
    var_27 = 10
    var_28 = module_0.Position(var_26, var_21, var_27)
    var_29 = module_0.Message(text=var_0, position=var_25)
    var_30 = module_0.Message(text=var_0, position=var_28)
    var_31 = module_0.Position(var_21, var_21, var_2)
    var_32 = 5
    var_33 = 4
    var_34 = module_0.Position(var_21, var_32, var_33)
    var_35 = module_0.Message(text=var_0, start_position=var_31, end_position=var_34)
    var_36 = module_0.Message(text=var_0, start_position=var_31, end_position=var_34)
    var_37 = module_0.Position(var_21, var_21, var_2)
    var_38 = module_0.Position(var_21, var_32, var_33)
    var_39 = module_0.Position(var_26, var_21, var_27)
    var_40 = 14
    var_41 = module_0.Position(var_26, var_32, var_40)
    var_42 = module_0.Message(text=var_0, start_position=var_37, end_position=var_38)
    var_43 = module_0.Message(text=var_0, start_position=var_39, end_position=var_41)
    var_44 = module_0.Message(text=var_0)
    var_45 = module_0.Message(text=var_0)
    var_46 = module_0.Message(text=var_0)
    var_47 = module_0.Message(text=var_0, key=var_15)
    var_48 = [var_15]
    var_49 = module_0.Message(text=var_0, index=var_48)
    var_50 = module_0.Position(var_21, var_21, var_2)
    var_51 = module_0.Message(text=var_0, position=var_50)
    var_52 = module_0.Message(text=var_0)
    var_53 = 'users'
    var_54 = 'name'
    var_55 = [var_53, var_2, var_54]
    var_56 = module_0.Message(text=var_0, index=var_55)
    var_57 = [var_53, var_2, var_54]
    var_58 = module_0.Message(text=var_0, index=var_57)



# Parsed testcases at query #22
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11, start_position=var_3, end_position=var_7)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = [var_12, var_17]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = 'Single error'
    var_23 = 'single'
    var_24 = 'field'
    var_25 = module_0.BaseError(text=var_22, code=var_23, key=var_24)
    var_26 = module_0.BaseError(text=var_22, code=var_23, key=var_24)
    var_27 = 'Error 3'
    var_28 = 'code3'
    var_29 = module_0.Message(text=var_27, code=var_28)
    var_30 = [var_12, var_29]
    var_31 = module_0.BaseError(messages=var_30)
    var_32 = [var_17, var_12]
    var_33 = module_0.BaseError(messages=var_32)
    var_34 = [var_12, var_17]
    var_35 = module_0.ValidationError(messages=var_34)
    var_36 = [var_12, var_17]
    var_37 = module_0.ParseError(messages=var_36)
    var_38 = module_0.Position(var_0, var_1, var_2)
    var_39 = module_0.Position(var_4, var_5, var_6)
    var_40 = [var_10]
    var_41 = module_0.Message(text=var_8, code=var_9, index=var_40, start_position=var_38, end_position=var_39)
    var_42 = [var_41, var_17]
    var_43 = module_0.BaseError(messages=var_42)
    var_44 = 'different_field'
    var_45 = [var_44]
    var_46 = module_0.Message(text=var_8, code=var_9, index=var_45, start_position=var_3, end_position=var_7)
    var_47 = [var_46, var_17]
    var_48 = module_0.BaseError(messages=var_47)
    var_49 = 'different_code'
    var_50 = [var_10]
    var_51 = module_0.Message(text=var_8, code=var_49, index=var_50, start_position=var_3, end_position=var_7)
    var_52 = [var_51, var_17]
    var_53 = module_0.BaseError(messages=var_52)
    var_54 = 'Different text'
    var_55 = [var_10]
    var_56 = module_0.Message(text=var_54, code=var_9, index=var_55, start_position=var_3, end_position=var_7)
    var_57 = [var_56, var_17]
    var_58 = module_0.BaseError(messages=var_57)
    var_59 = 'Test'
    var_60 = 'test'
    var_61 = module_0.Message(text=var_59, code=var_60)
    var_62 = [var_61]
    var_63 = module_0.BaseError(messages=var_62)
    var_64 = [var_61]
    var_65 = module_0.BaseError(messages=var_64)



# Parsed testcases at query #23
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = module_0.Message(text=var_8, code=var_9, position=var_3)
    var_11 = 'Error 2'
    var_12 = 'code2'
    var_13 = module_0.Message(text=var_11, code=var_12, position=var_7)
    var_14 = [var_10, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = [var_10, var_13]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = hash(var_15)
    var_19 = hash(var_17)
    var_20 = [var_13, var_10]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = 'Single error'
    var_23 = 'single'
    var_24 = module_0.ValidationError(text=var_22, code=var_23)
    var_25 = module_0.ValidationError(text=var_22, code=var_23)
    var_26 = 'Different error'
    var_27 = module_0.ValidationError(text=var_26, code=var_23)
    var_28 = 'different'
    var_29 = module_0.ValidationError(text=var_22, code=var_28)
    var_30 = 'Error 3'
    var_31 = 'code3'
    var_32 = module_0.Message(text=var_30, code=var_31)
    var_33 = [var_10, var_32]
    var_34 = module_0.ValidationError(messages=var_33)
    var_35 = module_0.ValidationError(text=var_8, code=var_9, position=var_3)
    var_36 = [var_10]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = [var_10, var_13]
    var_39 = module_0.ParseError(messages=var_38)
    var_40 = [var_10, var_13]
    var_41 = module_0.ParseError(messages=var_40)
    var_42 = 'Indexed error'
    var_43 = 'indexed'
    var_44 = 'key1'
    var_45 = 0
    var_46 = [var_44, var_45]
    var_47 = module_0.Message(text=var_42, code=var_43, index=var_46)
    var_48 = [var_47]
    var_49 = module_0.ValidationError(messages=var_48)
    var_50 = [var_47]
    var_51 = module_0.ValidationError(messages=var_50)
    var_52 = [var_44, var_0]
    var_53 = module_0.Message(text=var_42, code=var_43, index=var_52)
    var_54 = [var_53]
    var_55 = module_0.ValidationError(messages=var_54)



# Parsed testcases at query #24
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11, start_position=var_3, end_position=var_7)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = [var_12, var_17]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = [var_17, var_12]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = 'Error 3'
    var_25 = 'code3'
    var_26 = 'field3'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_24, code=var_25, index=var_27)
    var_29 = [var_12, var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = [var_12]
    var_32 = module_0.BaseError(messages=var_31)
    var_33 = 'Single error'
    var_34 = 'single'
    var_35 = 'field'
    var_36 = module_0.BaseError(text=var_33, code=var_34, key=var_35)
    var_37 = module_0.Message(text=var_33, code=var_34, key=var_35)
    var_38 = [var_37]
    var_39 = module_0.BaseError(messages=var_38)
    var_40 = [var_12, var_17]
    var_41 = module_0.ValidationError(messages=var_40)
    var_42 = [var_12, var_17]
    var_43 = module_0.ParseError(messages=var_42)
    var_44 = hash(var_19)
    var_45 = hash(var_21)
    var_46 = 'Error'
    var_47 = 'code'
    var_48 = [var_35]
    var_49 = module_0.Message(text=var_46, code=var_47, index=var_48)
    var_50 = 'Different'
    var_51 = [var_35]
    var_52 = module_0.Message(text=var_50, code=var_47, index=var_51)
    var_53 = [var_49]
    var_54 = module_0.BaseError(messages=var_53)
    var_55 = [var_52]
    var_56 = module_0.BaseError(messages=var_55)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = hash(var_2)
    var_5 = hash(var_3)
    var_6 = 'Error 1'
    var_7 = 'code1'
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_6, code=var_7, index=var_9)
    var_11 = 'Error 2'
    var_12 = 'code2'
    var_13 = 'field2'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_11, code=var_12, index=var_14)
    var_16 = [var_10, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationError(messages=var_16)
    var_19 = hash(var_17)
    var_20 = hash(var_18)
    var_21 = module_0.ValidationError(text=var_0, code=var_1)
    var_22 = 'Different message'
    var_23 = module_0.ValidationError(text=var_22, code=var_1)
    var_24 = hash(var_21)
    var_25 = hash(var_23)
    var_26 = module_0.Message(text=var_6, code=var_7)
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_6, code=var_7)
    var_29 = module_0.Message(text=var_11, code=var_12)
    var_30 = [var_28, var_29]
    var_31 = module_0.ValidationError(messages=var_27)
    var_32 = module_0.ValidationError(messages=var_30)
    var_33 = hash(var_31)
    var_34 = hash(var_32)
    var_35 = module_0.ValidationError(text=var_0, code=var_7)
    var_36 = module_0.ValidationError(text=var_0, code=var_12)
    var_37 = hash(var_35)
    var_38 = hash(var_36)
    var_39 = module_0.ValidationError(text=var_0)
    var_40 = [var_8]
    var_41 = module_0.Message(text=var_6, code=var_7, index=var_40)
    var_42 = [var_13]
    var_43 = module_0.Message(text=var_11, code=var_12, index=var_42)
    var_44 = [var_41, var_43]
    var_45 = [var_13]
    var_46 = module_0.Message(text=var_11, code=var_12, index=var_45)
    var_47 = [var_8]
    var_48 = module_0.Message(text=var_6, code=var_7, index=var_47)
    var_49 = [var_46, var_48]
    var_50 = module_0.ValidationError(messages=var_44)
    var_51 = module_0.ValidationError(messages=var_49)
    var_52 = hash(var_50)
    var_53 = hash(var_51)
    var_54 = 1
    var_55 = 0
    var_56 = module_0.Position(var_54, var_54, var_55)
    var_57 = 'Error'
    var_58 = 'code'
    var_59 = module_0.Message(text=var_57, code=var_58, position=var_56)
    var_60 = [var_59]
    var_61 = module_0.Message(text=var_57, code=var_58, position=var_56)
    var_62 = [var_61]
    var_63 = module_0.ValidationError(messages=var_60)
    var_64 = module_0.ValidationError(messages=var_62)
    var_65 = hash(var_63)
    var_66 = hash(var_64)
    var_67 = module_0.Position(var_54, var_54, var_55)
    var_68 = 2
    var_69 = 10
    var_70 = module_0.Position(var_68, var_54, var_69)
    var_71 = module_0.Message(text=var_57, code=var_58, position=var_67)
    var_72 = [var_71]
    var_73 = module_0.Message(text=var_57, code=var_58, position=var_70)
    var_74 = [var_73]
    var_75 = module_0.ValidationError(messages=var_72)
    var_76 = module_0.ValidationError(messages=var_74)
    var_77 = hash(var_75)
    var_78 = hash(var_76)
    var_79 = 'Parse error'
    var_80 = 'parse_error'
    var_81 = module_0.ParseError(text=var_79, code=var_80)
    var_82 = module_0.ParseError(text=var_79, code=var_80)
    var_83 = hash(var_81)
    var_84 = hash(var_82)
    var_85 = 'Validation error'
    var_86 = 'validation'
    var_87 = module_0.ValidationError(text=var_85, code=var_86)
    var_88 = module_0.ValidationError(text=var_85, code=var_86)
    var_89 = hash(var_87)
    var_90 = hash(var_88)
    var_91 = module_0.ParseError(text=var_57, code=var_58)
    var_92 = module_0.ValidationError(text=var_57, code=var_58)
    var_93 = hash(var_91)
    var_94 = hash(var_92)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = module_0.Position(var_7, var_7, var_8)
    var_10 = module_0.Message(text=var_0, position=var_9)
    var_11 = module_0.Message(text=var_0, position=var_9)
    var_12 = module_0.Position(var_7, var_7, var_8)
    var_13 = 5
    var_14 = 4
    var_15 = module_0.Position(var_7, var_13, var_14)
    var_16 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_17 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_18 = 'Error 1'
    var_19 = module_0.Message(text=var_18)
    var_20 = 'Error 2'
    var_21 = module_0.Message(text=var_20)
    var_22 = 'code1'
    var_23 = module_0.Message(text=var_0, code=var_22)
    var_24 = 'code2'
    var_25 = module_0.Message(text=var_0, code=var_24)
    var_26 = 'field1'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, index=var_27)
    var_29 = 'field2'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_0, index=var_30)
    var_32 = module_0.Position(var_7, var_7, var_8)
    var_33 = 2
    var_34 = 10
    var_35 = module_0.Position(var_33, var_7, var_34)
    var_36 = module_0.Message(text=var_0, start_position=var_32)
    var_37 = module_0.Message(text=var_0, start_position=var_35)
    var_38 = module_0.Message(text=var_0, end_position=var_32)
    var_39 = module_0.Message(text=var_0, end_position=var_35)
    var_40 = module_0.Message(text=var_0)
    var_41 = module_0.Message(text=var_0)
    var_42 = module_0.Message(text=var_0)
    var_43 = module_0.Message(text=var_0, key=var_2)
    var_44 = [var_2]
    var_45 = module_0.Message(text=var_0, index=var_44)
    var_46 = module_0.Position(var_7, var_7, var_8)
    var_47 = module_0.Message(text=var_0, position=var_46)
    var_48 = module_0.Message(text=var_0, start_position=var_46, end_position=var_46)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = 'Error 1'
    var_10 = module_0.Message(text=var_9, code=var_1, key=var_2)
    var_11 = 'Error 2'
    var_12 = module_0.Message(text=var_11, code=var_1, key=var_2)
    var_13 = 'code1'
    var_14 = module_0.Message(text=var_0, code=var_13, key=var_2)
    var_15 = 'code2'
    var_16 = module_0.Message(text=var_0, code=var_15, key=var_2)
    var_17 = [var_2, var_4]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = [var_2, var_3]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = module_0.Position(var_3, var_3, var_4)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_21)
    var_23 = 2
    var_24 = module_0.Position(var_23, var_3, var_4)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_24)
    var_26 = module_0.Position(var_3, var_3, var_4)
    var_27 = module_0.Message(text=var_0, code=var_1, end_position=var_26)
    var_28 = module_0.Position(var_3, var_23, var_4)
    var_29 = module_0.Message(text=var_0, code=var_1, end_position=var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_31 = [var_2]
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_31)
    var_33 = module_0.Position(var_3, var_3, var_4)
    var_34 = module_0.Message(text=var_0, code=var_1, position=var_33)
    var_35 = module_0.Position(var_3, var_3, var_4)
    var_36 = module_0.Position(var_3, var_3, var_4)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_36)
    var_38 = module_0.Message(text=var_0, code=var_1)
    var_39 = module_0.Message(text=var_0, code=var_1)
    var_40 = module_0.Message(text=var_0, code=var_1)
    var_41 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_42 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_43 = hash(var_41)
    var_44 = hash(var_42)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, index=var_16)
    var_18 = 'other'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, index=var_19)
    var_21 = 1
    var_22 = module_0.Position(var_21, var_21, var_2)
    var_23 = 2
    var_24 = 10
    var_25 = module_0.Position(var_23, var_21, var_24)
    var_26 = module_0.Message(text=var_0, start_position=var_22)
    var_27 = module_0.Message(text=var_0, start_position=var_25)
    var_28 = module_0.Message(text=var_0, end_position=var_22)
    var_29 = module_0.Message(text=var_0, end_position=var_25)
    var_30 = module_0.Message(text=var_0, start_position=var_22, end_position=var_25)
    var_31 = module_0.Message(text=var_0, start_position=var_22, end_position=var_25)
    var_32 = module_0.Message(text=var_0, position=var_22)
    var_33 = module_0.Message(text=var_0, start_position=var_22, end_position=var_22)
    var_34 = module_0.Message(text=var_0)
    var_35 = module_0.Message(text=var_0, key=var_15)
    var_36 = [var_15]
    var_37 = module_0.Message(text=var_0, index=var_36)
    var_38 = module_0.Message(text=var_0)
    var_39 = []
    var_40 = module_0.Message(text=var_0, index=var_39)
    var_41 = module_0.Message(text=var_0)
    var_42 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, start_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_26)
    var_29 = module_0.Message(text=var_0, end_position=var_23)
    var_30 = module_0.Message(text=var_0, end_position=var_26)
    var_31 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_32 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_33 = module_0.Message(text=var_0, position=var_23)
    var_34 = module_0.Message(text=var_0, start_position=var_23, end_position=var_23)
    var_35 = module_0.Message(text=var_0)
    var_36 = module_0.Message(text=var_0, key=var_2)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0)
    var_40 = []
    var_41 = module_0.Message(text=var_0, index=var_40)
    var_42 = 'max_length'
    var_43 = 'users'
    var_44 = 'name'
    var_45 = [var_43, var_22, var_44]
    var_46 = module_0.Message(text=var_0, code=var_42, index=var_45)
    var_47 = module_0.Message(text=var_0, position=var_23)
    var_48 = module_0.Message(text=var_0, position=var_23)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = module_0.Message(text=var_0, code=var_1)
    var_8 = module_0.Message(text=var_0, code=var_1)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = module_0.Position(var_9, var_10, var_11)
    var_13 = module_0.Message(text=var_0, position=var_12)
    var_14 = module_0.Message(text=var_0, position=var_12)
    var_15 = module_0.Position(var_9, var_10, var_11)
    var_16 = 5
    var_17 = 6
    var_18 = module_0.Position(var_9, var_16, var_17)
    var_19 = module_0.Message(text=var_0, start_position=var_15, end_position=var_18)
    var_20 = module_0.Message(text=var_0, start_position=var_15, end_position=var_18)
    var_21 = 'Error 1'
    var_22 = module_0.Message(text=var_21)
    var_23 = 'Error 2'
    var_24 = module_0.Message(text=var_23)
    var_25 = 'max_length'
    var_26 = module_0.Message(text=var_0, code=var_25)
    var_27 = 'min_length'
    var_28 = module_0.Message(text=var_0, code=var_27)
    var_29 = 'field1'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_0, index=var_30)
    var_32 = 'field2'
    var_33 = [var_32]
    var_34 = module_0.Message(text=var_0, index=var_33)
    var_35 = module_0.Position(var_9, var_10, var_11)
    var_36 = 4
    var_37 = module_0.Position(var_10, var_11, var_36)
    var_38 = module_0.Message(text=var_0, start_position=var_35)
    var_39 = module_0.Message(text=var_0, start_position=var_37)
    var_40 = module_0.Message(text=var_0, end_position=var_35)
    var_41 = module_0.Message(text=var_0, end_position=var_37)
    var_42 = module_0.Message(text=var_0)
    var_43 = module_0.Message(text=var_0, key=var_2)
    var_44 = [var_2]
    var_45 = module_0.Message(text=var_0, index=var_44)
    var_46 = module_0.Message(text=var_0, position=var_35)
    var_47 = module_0.Message(text=var_0, start_position=var_35, end_position=var_35)
    var_48 = [var_2]
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_48)
    var_50 = [var_2]
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_50)
    var_52 = hash(var_49)
    var_53 = hash(var_51)
    var_54 = [var_29]
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = [var_32]
    var_57 = module_0.Message(text=var_0, code=var_1, index=var_56)
    var_58 = hash(var_55)
    var_59 = hash(var_57)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = module_0.Message(text=var_0, code=var_1)
    var_8 = module_0.Message(text=var_0, code=var_1)
    var_9 = 1
    var_10 = 0
    var_11 = module_0.Position(var_9, var_9, var_10)
    var_12 = module_0.Message(text=var_0, position=var_11)
    var_13 = module_0.Message(text=var_0, position=var_11)
    var_14 = module_0.Position(var_9, var_9, var_10)
    var_15 = 5
    var_16 = 4
    var_17 = module_0.Position(var_9, var_15, var_16)
    var_18 = module_0.Message(text=var_0, start_position=var_14, end_position=var_17)
    var_19 = module_0.Message(text=var_0, start_position=var_14, end_position=var_17)
    var_20 = 'Error 1'
    var_21 = module_0.Message(text=var_20)
    var_22 = 'Error 2'
    var_23 = module_0.Message(text=var_22)
    var_24 = 'code1'
    var_25 = module_0.Message(text=var_0, code=var_24)
    var_26 = 'code2'
    var_27 = module_0.Message(text=var_0, code=var_26)
    var_28 = 'field1'
    var_29 = [var_28]
    var_30 = module_0.Message(text=var_0, index=var_29)
    var_31 = 'field2'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, index=var_32)
    var_34 = module_0.Position(var_9, var_9, var_10)
    var_35 = 2
    var_36 = 10
    var_37 = module_0.Position(var_35, var_9, var_36)
    var_38 = module_0.Message(text=var_0, start_position=var_34)
    var_39 = module_0.Message(text=var_0, start_position=var_37)
    var_40 = module_0.Message(text=var_0, end_position=var_34)
    var_41 = module_0.Message(text=var_0, end_position=var_37)
    var_42 = module_0.Message(text=var_0)
    var_43 = module_0.Message(text=var_0, key=var_2)
    var_44 = [var_2]
    var_45 = module_0.Message(text=var_0, index=var_44)
    var_46 = module_0.Message(text=var_0, position=var_34)
    var_47 = module_0.Message(text=var_0, start_position=var_34, end_position=var_34)
    var_48 = module_0.Message(text=var_0, position=var_34)
    var_49 = module_0.Message(text=var_0, start_position=var_34, end_position=var_37)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_2)
    var_6 = 5
    var_7 = 4
    var_8 = module_0.Position(var_4, var_6, var_7)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_8)
    var_10 = [var_2]
    var_11 = module_0.Position(var_4, var_4, var_2)
    var_12 = module_0.Position(var_4, var_6, var_7)
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_10, start_position=var_11, end_position=var_12)
    var_14 = 'Different'
    var_15 = [var_2]
    var_16 = module_0.Position(var_4, var_4, var_2)
    var_17 = module_0.Position(var_4, var_6, var_7)
    var_18 = module_0.Message(text=var_14, code=var_1, index=var_15, start_position=var_16, end_position=var_17)
    var_19 = 'different'
    var_20 = [var_2]
    var_21 = module_0.Position(var_4, var_4, var_2)
    var_22 = module_0.Position(var_4, var_6, var_7)
    var_23 = module_0.Message(text=var_0, code=var_19, index=var_20, start_position=var_21, end_position=var_22)
    var_24 = [var_4]
    var_25 = module_0.Position(var_4, var_4, var_2)
    var_26 = module_0.Position(var_4, var_6, var_7)
    var_27 = module_0.Message(text=var_0, code=var_1, index=var_24, start_position=var_25, end_position=var_26)
    var_28 = [var_2]
    var_29 = 2
    var_30 = module_0.Position(var_29, var_4, var_2)
    var_31 = module_0.Position(var_4, var_6, var_7)
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_28, start_position=var_30, end_position=var_31)
    var_33 = [var_2]
    var_34 = module_0.Position(var_4, var_4, var_2)
    var_35 = module_0.Position(var_29, var_6, var_7)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_33, start_position=var_34, end_position=var_35)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_37)
    var_39 = [var_2]
    var_40 = module_0.Message(text=var_0, code=var_1, index=var_39)
    var_41 = [var_2]
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_41)
    var_43 = [var_2]
    var_44 = module_0.Position(var_4, var_4, var_2)
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_43, position=var_44)
    var_46 = [var_2]
    var_47 = module_0.Position(var_4, var_4, var_2)
    var_48 = module_0.Message(text=var_0, code=var_1, index=var_46, position=var_47)
    var_49 = [var_2]
    var_50 = module_0.Position(var_29, var_4, var_2)
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_49, position=var_50)
    var_52 = 'field'
    var_53 = module_0.Message(text=var_0, code=var_1, key=var_52)
    var_54 = module_0.Message(text=var_0, code=var_1, key=var_52)
    var_55 = module_0.Message(text=var_0, code=var_1, key=var_19)
    var_56 = module_0.Message(text=var_0, code=var_1)
    var_57 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_2)
    var_6 = 5
    var_7 = 4
    var_8 = module_0.Position(var_4, var_6, var_7)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_3, start_position=var_5, end_position=var_8)
    var_10 = [var_2]
    var_11 = module_0.Position(var_4, var_4, var_2)
    var_12 = module_0.Position(var_4, var_6, var_7)
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_10, start_position=var_11, end_position=var_12)
    var_14 = 'Different'
    var_15 = [var_2]
    var_16 = module_0.Position(var_4, var_4, var_2)
    var_17 = module_0.Position(var_4, var_6, var_7)
    var_18 = module_0.Message(text=var_14, code=var_1, index=var_15, start_position=var_16, end_position=var_17)
    var_19 = 'max_length'
    var_20 = [var_2]
    var_21 = module_0.Position(var_4, var_4, var_2)
    var_22 = module_0.Position(var_4, var_6, var_7)
    var_23 = module_0.Message(text=var_0, code=var_19, index=var_20, start_position=var_21, end_position=var_22)
    var_24 = [var_4]
    var_25 = module_0.Position(var_4, var_4, var_2)
    var_26 = module_0.Position(var_4, var_6, var_7)
    var_27 = module_0.Message(text=var_0, code=var_1, index=var_24, start_position=var_25, end_position=var_26)
    var_28 = [var_2]
    var_29 = 2
    var_30 = module_0.Position(var_29, var_4, var_2)
    var_31 = module_0.Position(var_4, var_6, var_7)
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_28, start_position=var_30, end_position=var_31)
    var_33 = [var_2]
    var_34 = module_0.Position(var_4, var_4, var_2)
    var_35 = module_0.Position(var_29, var_6, var_7)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_33, start_position=var_34, end_position=var_35)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_37)
    var_39 = [var_2]
    var_40 = module_0.Message(text=var_0, code=var_1, index=var_39)
    var_41 = [var_2]
    var_42 = module_0.Position(var_4, var_4, var_2)
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_41, position=var_42)
    var_44 = [var_2]
    var_45 = module_0.Position(var_4, var_4, var_2)
    var_46 = module_0.Position(var_4, var_4, var_2)
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_44, start_position=var_45, end_position=var_46)
    var_48 = 'field'
    var_49 = module_0.Message(text=var_0, code=var_1, key=var_48)
    var_50 = [var_48]
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_50)
    var_52 = module_0.Message(text=var_0, code=var_1)
    var_53 = []
    var_54 = module_0.Message(text=var_0, code=var_1, index=var_53)
    var_55 = 'users'
    var_56 = 3
    var_57 = 'username'
    var_58 = [var_55, var_56, var_57]
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58)
    var_60 = [var_55, var_56, var_57]
    var_61 = module_0.Message(text=var_0, code=var_1, index=var_60)



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = [var_2]
    var_16 = module_0.Message(text=var_0, index=var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = module_0.Message(text=var_0, index=var_18)
    var_20 = module_0.Position(var_17, var_17, var_2)
    var_21 = 2
    var_22 = 10
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = module_0.Message(text=var_0, start_position=var_20)
    var_25 = module_0.Message(text=var_0, start_position=var_23)
    var_26 = module_0.Message(text=var_0, end_position=var_20)
    var_27 = module_0.Message(text=var_0, end_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_29 = module_0.Message(text=var_0, start_position=var_20, end_position=var_23)
    var_30 = module_0.Message(text=var_0, position=var_20)
    var_31 = module_0.Message(text=var_0, start_position=var_20, end_position=var_20)
    var_32 = module_0.Message(text=var_0)
    var_33 = 'field'
    var_34 = module_0.Message(text=var_0, key=var_33)
    var_35 = [var_33]
    var_36 = module_0.Message(text=var_0, index=var_35)
    var_37 = []
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'max_length'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'min_length'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_29 = module_0.Position(var_21, var_21, var_22)
    var_30 = module_0.Message(text=var_0, code=var_1, position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, position=var_29)
    var_32 = module_0.Position(var_21, var_21, var_22)
    var_33 = 5
    var_34 = 4
    var_35 = module_0.Position(var_21, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_35)
    var_38 = module_0.Position(var_21, var_21, var_22)
    var_39 = module_0.Position(var_21, var_33, var_34)
    var_40 = module_0.Position(var_24, var_21, var_25)
    var_41 = 14
    var_42 = module_0.Position(var_24, var_33, var_41)
    var_43 = module_0.Message(text=var_0, code=var_1, start_position=var_38, end_position=var_39)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_42)
    var_45 = module_0.Message(text=var_0, code=var_1)
    var_46 = module_0.Message(text=var_0, code=var_1)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_49 = [var_2]
    var_50 = module_0.Message(text=var_0, code=var_1, index=var_49)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_15)
    var_52 = [var_18]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = []
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = module_0.Message(text=var_0, code=var_1)
    var_57 = 'users'
    var_58 = 'name'
    var_59 = [var_57, var_22, var_58]
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_59)
    var_61 = [var_57, var_22, var_58]
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_61)
    var_63 = [var_57, var_22, var_58]
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)
    var_65 = [var_57, var_21, var_58]
    var_66 = module_0.Message(text=var_0, code=var_1, index=var_65)



# Parsed testcases at query #36
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
    var_8 = 'Error message'
    var_9 = 'max_length'
    var_10 = 'username'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10, start_position=var_3, end_position=var_7)
    var_12 = module_0.Message(text=var_8, code=var_9, key=var_10, start_position=var_3, end_position=var_7)
    var_13 = 'users'
    var_14 = 0
    var_15 = [var_13, var_14, var_10]
    var_16 = module_0.Message(text=var_8, code=var_9, index=var_15, start_position=var_3, end_position=var_7)
    var_17 = [var_13, var_14, var_10]
    var_18 = module_0.Message(text=var_8, code=var_9, index=var_17, start_position=var_3, end_position=var_7)
    var_19 = module_0.Message(text=var_8, code=var_9, key=var_10, position=var_3)
    var_20 = module_0.Message(text=var_8, code=var_9, key=var_10, position=var_3)
    var_21 = 'Error message 1'
    var_22 = module_0.Message(text=var_21, code=var_9, key=var_10)
    var_23 = 'Error message 2'
    var_24 = module_0.Message(text=var_23, code=var_9, key=var_10)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'min_length'
    var_27 = module_0.Message(text=var_8, code=var_26, key=var_10)
    var_28 = [var_13, var_14, var_10]
    var_29 = module_0.Message(text=var_8, code=var_9, index=var_28)
    var_30 = [var_13, var_0, var_10]
    var_31 = module_0.Message(text=var_8, code=var_9, index=var_30)
    var_32 = module_0.Message(text=var_8, code=var_9, key=var_10, start_position=var_3, end_position=var_7)
    var_33 = module_0.Message(text=var_8, code=var_9, key=var_10, start_position=var_7, end_position=var_3)
    var_34 = module_0.Message(text=var_8)
    var_35 = module_0.Message(text=var_8)
    var_36 = 'custom'
    var_37 = module_0.Message(text=var_8, code=var_36)
    var_38 = module_0.Message(text=var_8, key=var_10)
    var_39 = [var_10]
    var_40 = module_0.Message(text=var_8, index=var_39)
    var_41 = module_0.Message(text=var_8, code=var_9)
    var_42 = module_0.Message(text=var_8, code=var_9)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11, start_position=var_3, end_position=var_7)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = [var_12, var_17]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = hash(var_19)
    var_23 = hash(var_21)
    var_24 = [var_17, var_12]
    var_25 = module_0.BaseError(messages=var_24)
    var_26 = hash(var_19)
    var_27 = hash(var_25)
    var_28 = 'Error 3'
    var_29 = 'code3'
    var_30 = 'field3'
    var_31 = [var_30]
    var_32 = module_0.Message(text=var_28, code=var_29, index=var_31)
    var_33 = [var_12, var_32]
    var_34 = module_0.BaseError(messages=var_33)
    var_35 = hash(var_19)
    var_36 = hash(var_34)
    var_37 = [var_12]
    var_38 = module_0.BaseError(messages=var_37)
    var_39 = 'Single error'
    var_40 = 'single'
    var_41 = 'field'
    var_42 = module_0.BaseError(text=var_39, code=var_40, key=var_41)
    var_43 = module_0.BaseError(text=var_39, code=var_40, key=var_41)
    var_44 = hash(var_42)
    var_45 = hash(var_43)
    var_46 = 'Different error'
    var_47 = module_0.BaseError(text=var_46, code=var_40, key=var_41)
    var_48 = [var_12, var_17]
    var_49 = module_0.ValidationError(messages=var_48)
    var_50 = [var_12, var_17]
    var_51 = module_0.ValidationError(messages=var_50)
    var_52 = hash(var_49)
    var_53 = hash(var_51)
    var_54 = [var_12, var_17]
    var_55 = module_0.BaseError(messages=var_54)
    var_56 = [var_12, var_17]
    var_57 = module_0.ParseError(messages=var_56)
    var_58 = [var_12, var_17]
    var_59 = module_0.ParseError(messages=var_58)
    var_60 = hash(var_57)
    var_61 = hash(var_59)
    var_62 = 'A'
    var_63 = 'code'
    var_64 = [var_41]
    var_65 = module_0.Message(text=var_62, code=var_63, index=var_64)
    var_66 = 'B'
    var_67 = [var_41]
    var_68 = module_0.Message(text=var_66, code=var_63, index=var_67)
    var_69 = hash(var_65)
    var_70 = hash(var_68)
    var_71 = [var_65]
    var_72 = module_0.BaseError(messages=var_71)
    var_73 = [var_68]
    var_74 = module_0.BaseError(messages=var_73)
    var_75 = 'Global error'
    var_76 = 'global'
    var_77 = module_0.Message(text=var_75, code=var_76)
    var_78 = module_0.Message(text=var_75, code=var_76)
    var_79 = [var_77]
    var_80 = module_0.BaseError(messages=var_79)
    var_81 = [var_78]
    var_82 = module_0.BaseError(messages=var_81)
    var_83 = 'Error'
    var_84 = module_0.Message(text=var_83, code=var_63, position=var_3)
    var_85 = module_0.Message(text=var_83, code=var_63, position=var_3)
    var_86 = module_0.Message(text=var_83, code=var_63, position=var_7)
    var_87 = [var_84]
    var_88 = module_0.BaseError(messages=var_87)
    var_89 = [var_85]
    var_90 = module_0.BaseError(messages=var_89)
    var_91 = [var_86]
    var_92 = module_0.BaseError(messages=var_91)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Field is required'
    var_1 = 'required'
    var_2 = 'username'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = var_3._messages
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 1
    var_7 = 5
    var_8 = 4
    var_9 = module_0.Position(var_6, var_7, var_8)
    var_10 = 'Invalid format'
    var_11 = 'format'
    var_12 = module_0.BaseError(text=var_10, code=var_11, position=var_9)
    var_13 = 'Too short'
    var_14 = 'min_length'
    var_15 = 'password'
    var_16 = module_0.Message(text=var_13, code=var_14, key=var_15)
    var_17 = 'Missing uppercase'
    var_18 = 'pattern'
    var_19 = module_0.Message(text=var_17, code=var_18, key=var_15)
    var_20 = [var_16, var_19]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = var_21._messages
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'Test'
    var_25 = module_0.BaseError(text=var_24, messages=var_20)
    var_26 = []
    var_27 = module_0.BaseError(messages=var_26)
    var_28 = 'Error'
    var_29 = 'field'
    var_30 = module_0.BaseError(text=var_28, key=var_29)
    var_31 = 'Error 1'
    var_32 = 'users'
    var_33 = 0
    var_34 = 'name'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.Message(text=var_31, index=var_35)
    var_37 = 'Error 2'
    var_38 = 'email'
    var_39 = [var_32, var_33, var_38]
    var_40 = module_0.Message(text=var_37, index=var_39)
    var_41 = [var_36, var_40]
    var_42 = module_0.BaseError(messages=var_41)
    var_43 = 'Global error'
    var_44 = module_0.BaseError(text=var_43)
    var_45 = 'test'
    var_46 = module_0.BaseError(text=var_28, key=var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = list(var_46)
    var_49 = 'Same'
    var_50 = 'key'
    var_51 = module_0.BaseError(text=var_49, key=var_50)
    var_52 = module_0.BaseError(text=var_49, key=var_50)
    var_53 = 'Different'
    var_54 = module_0.BaseError(text=var_53, key=var_50)
    var_55 = 'Test'
    var_56 = module_0.BaseError(text=var_55, key=var_29)
    var_57 = module_0.BaseError(text=var_55, key=var_29)
    var_58 = hash(var_56)
    var_59 = hash(var_57)
    var_60 = 'Message'
    var_61 = module_0.BaseError(text=var_60, key=var_29)
    var_62 = var_61.messages()
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'parent'
    var_65 = var_61.messages(add_prefix=var_64)
    var_66 = len(var_65)
    assert var_66 == 1
    var_67 = 'Simple error'
    var_68 = module_0.BaseError(text=var_67)
    var_69 = str(var_68)
    assert var_69 == 'Simple error'
    var_70 = 'field1'
    var_71 = module_0.Message(text=var_31, key=var_70)
    var_72 = 'field2'
    var_73 = module_0.Message(text=var_37, key=var_72)
    var_74 = [var_71, var_73]
    var_75 = module_0.BaseError(messages=var_74)
    var_76 = str(var_75)
    assert var_76 == "{'field1': 'Error 1', 'field2': 'Error 2'}"
    var_77 = 'custom'
    var_78 = module_0.BaseError(text=var_28, code=var_77)
    var_79 = repr(var_78)
    assert var_79 == "BaseError(text='Error', code='custom')"
    var_80 = module_0.Message(text=var_31, key=var_70)
    var_81 = module_0.Message(text=var_37, key=var_72)
    var_82 = [var_80, var_81]
    var_83 = module_0.BaseError(messages=var_82)
    var_84 = repr(var_83)
    var_85 = 'A'
    var_86 = 'a'
    var_87 = module_0.Message(text=var_85, key=var_86)
    var_88 = 'B'
    var_89 = 'b'
    var_90 = module_0.Message(text=var_88, key=var_89)
    var_91 = [var_87, var_90]
    var_92 = module_0.BaseError(messages=var_91)
    var_93 = list(var_92)
    var_94 = set(var_93)



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_29 = module_0.Position(var_21, var_21, var_22)
    var_30 = module_0.Message(text=var_0, code=var_1, position=var_29)
    var_31 = module_0.Message(text=var_0, code=var_1, position=var_29)
    var_32 = module_0.Position(var_21, var_21, var_22)
    var_33 = 5
    var_34 = 4
    var_35 = module_0.Position(var_21, var_33, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_32, end_position=var_35)
    var_38 = module_0.Position(var_21, var_21, var_22)
    var_39 = module_0.Position(var_21, var_33, var_34)
    var_40 = module_0.Position(var_24, var_21, var_25)
    var_41 = 14
    var_42 = module_0.Position(var_24, var_33, var_41)
    var_43 = module_0.Message(text=var_0, code=var_1, start_position=var_38, end_position=var_39)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_42)
    var_45 = module_0.Message(text=var_0, code=var_1)
    var_46 = module_0.Message(text=var_0, code=var_1)
    var_47 = module_0.Position(var_21, var_21, var_22)
    var_48 = module_0.Message(text=var_0, code=var_1, position=var_47)
    var_49 = module_0.Message(text=var_0, code=var_1, start_position=var_47, end_position=var_47)
    var_50 = module_0.Message(text=var_0, code=var_1)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_52 = [var_2]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = module_0.Message(text=var_0, code=var_1)
    var_55 = []
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_55)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'
    var_4 = 'Field required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.ValidationError(text=var_4, code=var_5, key=var_6)
    var_8 = str(var_7)
    assert var_8 == "{'username': 'Field required'}"
    var_9 = [var_6]
    var_10 = module_0.Message(text=var_4, code=var_5, index=var_9)
    var_11 = 'Too short'
    var_12 = 'min_length'
    var_13 = 'password'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_11, code=var_12, index=var_14)
    var_16 = [var_10, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = str(var_17)
    assert var_18 == "{'username': 'Field required', 'password': 'Too short'}"
    var_19 = 'Invalid email'
    var_20 = 'users'
    var_21 = 0
    var_22 = 'email'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.Message(text=var_19, code=var_1, index=var_23)
    var_25 = 'Too long'
    var_26 = 'max_length'
    var_27 = 1
    var_28 = 'name'
    var_29 = [var_20, var_27, var_28]
    var_30 = module_0.Message(text=var_25, code=var_26, index=var_29)
    var_31 = [var_24, var_30]
    var_32 = module_0.ValidationError(messages=var_31)
    var_33 = "{'users': {0: {'email': 'Invalid email'}, 1: {'name': 'Too long'}}}"
    var_34 = str(var_32)
    var_35 = 'Invalid format'
    var_36 = []
    var_37 = module_0.Message(text=var_35, code=var_1, index=var_36)
    var_38 = 'Missing field'
    var_39 = 'profile'
    var_40 = 'age'
    var_41 = [var_39, var_40]
    var_42 = module_0.Message(text=var_38, code=var_5, index=var_41)
    var_43 = [var_37, var_42]
    var_44 = module_0.ValidationError(messages=var_43)
    var_45 = str(var_44)
    assert var_45 == "{'': 'Invalid format', 'profile': {'age': 'Missing field'}}"
    var_46 = 'Invalid JSON'
    var_47 = 'parse_error'
    var_48 = module_0.ParseError(text=var_46, code=var_47)
    var_49 = str(var_48)
    assert var_49 == 'Invalid JSON'
    var_50 = 5
    var_51 = 4
    var_52 = module_0.Position(var_27, var_50, var_51)
    var_53 = 'Syntax error'
    var_54 = 'syntax'
    var_55 = 'query'
    var_56 = module_0.ValidationError(text=var_53, code=var_54, key=var_55, position=var_52)
    var_57 = str(var_56)
    assert var_57 == "{'query': 'Syntax error'}"



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "ValidationError(text='Invalid value', code='invalid')"
    var_4 = 'Field required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.ValidationError(text=var_4, code=var_5, key=var_6)
    var_8 = repr(var_7)
    assert var_8 == "ValidationError([Message(text='Field required', code='required', index=['username'])])"
    var_9 = 'Too short'
    var_10 = 'min_length'
    var_11 = [var_6]
    var_12 = module_0.Message(text=var_9, code=var_10, index=var_11)
    var_13 = 'Invalid format'
    var_14 = 'format'
    var_15 = 'email'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = repr(var_19)
    var_21 = 1
    var_22 = 5
    var_23 = 4
    var_24 = module_0.Position(var_21, var_22, var_23)
    var_25 = 'Syntax error'
    var_26 = 'syntax'
    var_27 = module_0.ValidationError(text=var_25, code=var_26, position=var_24)
    var_28 = repr(var_27)
    var_29 = 'Invalid JSON'
    var_30 = 'invalid_json'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = repr(var_31)
    assert var_32 == "ParseError(text='Invalid JSON', code='invalid_json')"
    var_33 = 'Invalid'
    var_34 = 'users'
    var_35 = 0
    var_36 = 'name'
    var_37 = [var_34, var_35, var_36]
    var_38 = module_0.Message(text=var_33, code=var_1, index=var_37)
    var_39 = [var_38]
    var_40 = module_0.ValidationError(messages=var_39)
    var_41 = repr(var_40)
    var_42 = []
    var_43 = module_0.ValidationError(messages=var_42)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, start_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_26)
    var_29 = module_0.Message(text=var_0, end_position=var_23)
    var_30 = module_0.Message(text=var_0, end_position=var_26)
    var_31 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_32 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_33 = module_0.Message(text=var_0, position=var_23)
    var_34 = module_0.Message(text=var_0, position=var_23)
    var_35 = module_0.Message(text=var_0)
    var_36 = module_0.Message(text=var_0, key=var_2)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0)
    var_40 = []
    var_41 = module_0.Message(text=var_0, index=var_40)
    var_42 = module_0.Message(text=var_0)
    var_43 = None
    var_44 = module_0.Message(text=var_0, start_position=var_43, end_position=var_43)



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_2)
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_7, position=var_8)
    var_10 = 'Error 1'
    var_11 = module_0.Message(text=var_10, code=var_1)
    var_12 = 'Error 2'
    var_13 = module_0.Message(text=var_12, code=var_1)
    var_14 = 'code1'
    var_15 = module_0.Message(text=var_0, code=var_14)
    var_16 = 'code2'
    var_17 = module_0.Message(text=var_0, code=var_16)
    var_18 = 'field'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 'other'
    var_22 = [var_21]
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_22)
    var_24 = module_0.Position(var_4, var_4, var_2)
    var_25 = module_0.Message(text=var_0, start_position=var_24)
    var_26 = 2
    var_27 = 10
    var_28 = module_0.Position(var_26, var_4, var_27)
    var_29 = module_0.Message(text=var_0, start_position=var_28)
    var_30 = 5
    var_31 = 4
    var_32 = module_0.Position(var_4, var_30, var_31)
    var_33 = module_0.Message(text=var_0, end_position=var_32)
    var_34 = 9
    var_35 = module_0.Position(var_4, var_27, var_34)
    var_36 = module_0.Message(text=var_0, end_position=var_35)
    var_37 = module_0.Position(var_4, var_4, var_2)
    var_38 = module_0.Message(text=var_0, position=var_37)
    var_39 = module_0.Position(var_4, var_4, var_2)
    var_40 = module_0.Position(var_4, var_4, var_2)
    var_41 = module_0.Message(text=var_0, start_position=var_39, end_position=var_40)
    var_42 = module_0.Message(text=var_0)
    var_43 = module_0.Message(text=var_0)
    var_44 = module_0.Message(text=var_0)
    var_45 = module_0.Message(text=var_0, key=var_18)
    var_46 = [var_18]
    var_47 = module_0.Message(text=var_0, index=var_46)
    var_48 = module_0.Position(var_4, var_4, var_2)
    var_49 = module_0.Message(text=var_0, position=var_48)
    var_50 = module_0.Message(text=var_0)
    var_51 = 'users'
    var_52 = 'name'
    var_53 = [var_51, var_2, var_52]
    var_54 = module_0.Message(text=var_0, index=var_53)
    var_55 = [var_51, var_2, var_52]
    var_56 = module_0.Message(text=var_0, index=var_55)



# Parsed testcases at query #8
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
    var_8 = 'Error message'
    var_9 = 'custom'
    var_10 = 'field'
    var_11 = 0
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_8, code=var_9, index=var_12, start_position=var_3, end_position=var_7)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_8, code=var_9, index=var_14, start_position=var_3, end_position=var_7)
    var_16 = 'Different message'
    var_17 = [var_10, var_11]
    var_18 = module_0.Message(text=var_16, code=var_9, index=var_17, start_position=var_3, end_position=var_7)
    var_19 = 'different_code'
    var_20 = [var_10, var_11]
    var_21 = module_0.Message(text=var_8, code=var_19, index=var_20, start_position=var_3, end_position=var_7)
    var_22 = 'different_field'
    var_23 = [var_22, var_11]
    var_24 = module_0.Message(text=var_8, code=var_9, index=var_23, start_position=var_3, end_position=var_7)
    var_25 = 7
    var_26 = 8
    var_27 = 9
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = [var_10, var_11]
    var_30 = module_0.Message(text=var_8, code=var_9, index=var_29, start_position=var_28, end_position=var_7)
    var_31 = [var_10, var_11]
    var_32 = module_0.Message(text=var_8, code=var_9, index=var_31, start_position=var_3, end_position=var_28)
    var_33 = module_0.Message(text=var_8, code=var_9)
    var_34 = module_0.Message(text=var_8, code=var_9)
    var_35 = module_0.Message(text=var_8, code=var_9, start_position=var_3, end_position=var_7)
    var_36 = module_0.Message(text=var_8, key=var_10)
    var_37 = module_0.Message(text=var_8, key=var_10)
    var_38 = module_0.Message(text=var_8, key=var_22)
    var_39 = module_0.Message(text=var_8, position=var_3)
    var_40 = module_0.Message(text=var_8, position=var_3)
    var_41 = module_0.Message(text=var_8, position=var_7)
    var_42 = module_0.Message(text=var_8, start_position=var_3, end_position=var_3)
    var_43 = module_0.Message(text=var_8, position=var_3)
    var_44 = [var_10, var_11]



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = module_0.Position(var_7, var_7, var_8)
    var_10 = module_0.Message(text=var_0, position=var_9)
    var_11 = module_0.Message(text=var_0, position=var_9)
    var_12 = module_0.Position(var_7, var_7, var_8)
    var_13 = 5
    var_14 = 4
    var_15 = module_0.Position(var_7, var_13, var_14)
    var_16 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_17 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_18 = 'Error 1'
    var_19 = module_0.Message(text=var_18)
    var_20 = 'Error 2'
    var_21 = module_0.Message(text=var_20)
    var_22 = 'max_length'
    var_23 = module_0.Message(text=var_0, code=var_22)
    var_24 = 'min_length'
    var_25 = module_0.Message(text=var_0, code=var_24)
    var_26 = 'field1'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, index=var_27)
    var_29 = 'field2'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_0, index=var_30)
    var_32 = module_0.Position(var_7, var_7, var_8)
    var_33 = 2
    var_34 = 10
    var_35 = module_0.Position(var_33, var_7, var_34)
    var_36 = module_0.Message(text=var_0, start_position=var_32)
    var_37 = module_0.Message(text=var_0, start_position=var_35)
    var_38 = module_0.Message(text=var_0, end_position=var_32)
    var_39 = module_0.Message(text=var_0, end_position=var_35)
    var_40 = module_0.Message(text=var_0)
    var_41 = module_0.Message(text=var_0, key=var_2)
    var_42 = [var_2]
    var_43 = module_0.Message(text=var_0, index=var_42)
    var_44 = module_0.Position(var_7, var_7, var_8)
    var_45 = module_0.Message(text=var_0, position=var_44)
    var_46 = module_0.Message(text=var_0, start_position=var_44, end_position=var_44)
    var_47 = [var_2]
    var_48 = module_0.Message(text=var_0, code=var_1, index=var_47)
    var_49 = [var_2]
    var_50 = module_0.Message(text=var_0, code=var_1, index=var_49)
    var_51 = hash(var_48)
    var_52 = hash(var_50)
    var_53 = [var_2]
    var_54 = module_0.Message(text=var_18, code=var_22, index=var_53)
    var_55 = [var_2]
    var_56 = module_0.Message(text=var_20, code=var_22, index=var_55)
    var_57 = hash(var_54)
    var_58 = hash(var_56)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = 'Error 1'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'Error 2'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'max_length'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'min_length'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, index=var_17)
    var_19 = 'other'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, index=var_20)
    var_22 = module_0.Position(var_3, var_3, var_4)
    var_23 = module_0.Message(text=var_0, start_position=var_22)
    var_24 = 2
    var_25 = module_0.Position(var_24, var_3, var_4)
    var_26 = module_0.Message(text=var_0, start_position=var_25)
    var_27 = module_0.Position(var_3, var_3, var_4)
    var_28 = module_0.Message(text=var_0, end_position=var_27)
    var_29 = module_0.Position(var_3, var_24, var_4)
    var_30 = module_0.Message(text=var_0, end_position=var_29)
    var_31 = module_0.Position(var_3, var_3, var_4)
    var_32 = module_0.Message(text=var_0, position=var_31)
    var_33 = module_0.Position(var_3, var_3, var_4)
    var_34 = module_0.Position(var_3, var_3, var_4)
    var_35 = module_0.Message(text=var_0, start_position=var_33, end_position=var_34)
    var_36 = module_0.Message(text=var_0)
    var_37 = module_0.Message(text=var_0)
    var_38 = module_0.Message(text=var_0)
    var_39 = module_0.Position(var_3, var_3, var_4)
    var_40 = module_0.Message(text=var_0, position=var_39)
    var_41 = module_0.Message(text=var_0)
    var_42 = module_0.Message(text=var_0, key=var_2)
    var_43 = [var_2]
    var_44 = module_0.Message(text=var_0, index=var_43)
    var_45 = 'users'
    var_46 = 3
    var_47 = 'username'
    var_48 = [var_45, var_46, var_47]
    var_49 = module_0.Message(text=var_0, index=var_48)
    var_50 = [var_45, var_46, var_47]
    var_51 = module_0.Message(text=var_0, index=var_50)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 0
    var_6 = module_0.Position(var_4, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_8, position=var_9)
    var_11 = 'Error 1'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'Error 2'
    var_14 = module_0.Message(text=var_13, code=var_1)
    var_15 = 'code1'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = 'code2'
    var_18 = module_0.Message(text=var_0, code=var_17)
    var_19 = 'field1'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 'field2'
    var_23 = [var_22]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    var_25 = module_0.Position(var_4, var_4, var_5)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25)
    var_27 = 2
    var_28 = module_0.Position(var_27, var_4, var_5)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_28)
    var_30 = module_0.Position(var_4, var_4, var_5)
    var_31 = module_0.Message(text=var_0, code=var_1, end_position=var_30)
    var_32 = module_0.Position(var_4, var_27, var_5)
    var_33 = module_0.Message(text=var_0, code=var_1, end_position=var_32)
    var_34 = module_0.Position(var_4, var_4, var_5)
    var_35 = module_0.Message(text=var_0, code=var_1, position=var_34)
    var_36 = module_0.Position(var_4, var_4, var_5)
    var_37 = module_0.Position(var_4, var_4, var_5)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_36, end_position=var_37)
    var_39 = module_0.Message(text=var_0, code=var_1)
    var_40 = module_0.Message(text=var_0, code=var_1)
    var_41 = []
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_41)
    var_43 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_44 = [var_2]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)
    var_46 = 'users'
    var_47 = 'name'
    var_48 = [var_46, var_5, var_47]
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_48)
    var_50 = [var_46, var_5, var_47]
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_50)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = module_0.Position(var_3, var_4, var_5)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_8)
    var_10 = [var_2]
    var_11 = module_0.Position(var_3, var_4, var_5)
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_10, position=var_11)
    var_13 = [var_2]
    var_14 = module_0.Position(var_3, var_4, var_5)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_13, position=var_14)
    var_16 = module_0.Position(var_3, var_4, var_5)
    var_17 = 5
    var_18 = 6
    var_19 = module_0.Position(var_3, var_17, var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_16, end_position=var_19)
    var_21 = module_0.Position(var_3, var_4, var_5)
    var_22 = module_0.Position(var_3, var_17, var_18)
    var_23 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_21, end_position=var_22)
    var_24 = 'Error 1'
    var_25 = module_0.Message(text=var_24, code=var_1, key=var_2)
    var_26 = 'Error 2'
    var_27 = module_0.Message(text=var_26, code=var_1, key=var_2)
    var_28 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_29 = 'required'
    var_30 = module_0.Message(text=var_0, code=var_29, key=var_2)
    var_31 = 'field1'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)
    var_34 = 'field2'
    var_35 = [var_34]
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_35)
    var_37 = module_0.Position(var_3, var_4, var_5)
    var_38 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_37)
    var_39 = 4
    var_40 = module_0.Position(var_4, var_5, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_40)
    var_42 = module_0.Position(var_3, var_4, var_5)
    var_43 = module_0.Position(var_3, var_17, var_18)
    var_44 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_42, end_position=var_43)
    var_45 = module_0.Position(var_4, var_5, var_39)
    var_46 = module_0.Position(var_3, var_17, var_18)
    var_47 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_45, end_position=var_46)
    var_48 = module_0.Position(var_3, var_4, var_5)
    var_49 = module_0.Position(var_3, var_17, var_18)
    var_50 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_48, end_position=var_49)
    var_51 = module_0.Position(var_3, var_4, var_5)
    var_52 = module_0.Position(var_4, var_5, var_39)
    var_53 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_51, end_position=var_52)
    var_54 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_55 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_56 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_57 = module_0.Message(text=var_0, key=var_2)
    var_58 = 'custom'
    var_59 = module_0.Message(text=var_0, code=var_58, key=var_2)
    var_60 = 'users'
    var_61 = 'username'
    var_62 = [var_60, var_5, var_61]
    var_63 = module_0.Message(text=var_0, code=var_1, index=var_62)
    var_64 = [var_60, var_5, var_61]
    var_65 = module_0.Message(text=var_0, code=var_1, index=var_64)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'other'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = module_0.Position(var_21, var_21, var_2)
    var_23 = module_0.Message(text=var_0, position=var_22)
    var_24 = module_0.Message(text=var_0, position=var_22)
    var_25 = module_0.Position(var_21, var_21, var_2)
    var_26 = 5
    var_27 = 4
    var_28 = module_0.Position(var_21, var_26, var_27)
    var_29 = module_0.Message(text=var_0, start_position=var_25, end_position=var_28)
    var_30 = module_0.Message(text=var_0, start_position=var_25, end_position=var_28)
    var_31 = module_0.Position(var_21, var_21, var_2)
    var_32 = 2
    var_33 = 10
    var_34 = module_0.Position(var_32, var_21, var_33)
    var_35 = module_0.Message(text=var_0, start_position=var_31)
    var_36 = module_0.Message(text=var_0, start_position=var_34)
    var_37 = module_0.Position(var_21, var_26, var_27)
    var_38 = 9
    var_39 = module_0.Position(var_21, var_33, var_38)
    var_40 = module_0.Message(text=var_0, end_position=var_37)
    var_41 = module_0.Message(text=var_0, end_position=var_39)
    var_42 = module_0.Message(text=var_0, key=var_15)
    var_43 = [var_15]
    var_44 = module_0.Message(text=var_0, index=var_43)
    var_45 = module_0.Message(text=var_0)
    var_46 = module_0.Message(text=var_0)
    var_47 = []
    var_48 = module_0.Message(text=var_0, index=var_47)
    var_49 = 'users'
    var_50 = 'name'
    var_51 = [var_49, var_2, var_50]
    var_52 = module_0.Message(text=var_0, index=var_51)
    var_53 = [var_49, var_2, var_50]
    var_54 = module_0.Message(text=var_0, index=var_53)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = 'Different'
    var_10 = module_0.Position(var_3, var_3, var_4)
    var_11 = module_0.Message(text=var_9, code=var_1, key=var_2, position=var_10)
    var_12 = 'max_length'
    var_13 = module_0.Position(var_3, var_3, var_4)
    var_14 = module_0.Message(text=var_0, code=var_12, key=var_2, position=var_13)
    var_15 = 'other_field'
    var_16 = module_0.Position(var_3, var_3, var_4)
    var_17 = module_0.Message(text=var_0, code=var_1, key=var_15, position=var_16)
    var_18 = 2
    var_19 = 10
    var_20 = module_0.Position(var_18, var_3, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_20)
    var_22 = module_0.Position(var_3, var_3, var_4)
    var_23 = module_0.Position(var_3, var_3, var_4)
    var_24 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_22, end_position=var_23)
    var_25 = module_0.Position(var_3, var_3, var_4)
    var_26 = module_0.Position(var_3, var_18, var_3)
    var_27 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_25, end_position=var_26)
    var_28 = [var_2]
    var_29 = module_0.Position(var_3, var_3, var_4)
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_28, position=var_29)
    var_31 = 'nested'
    var_32 = [var_2, var_31]
    var_33 = module_0.Position(var_3, var_3, var_4)
    var_34 = module_0.Message(text=var_0, code=var_1, index=var_32, position=var_33)
    var_35 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_36 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_37 = 'users'
    var_38 = 3
    var_39 = 'username'
    var_40 = [var_37, var_38, var_39]
    var_41 = module_0.Position(var_3, var_3, var_4)
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_40, position=var_41)
    var_43 = [var_37, var_38, var_39]
    var_44 = module_0.Position(var_3, var_3, var_4)
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_43, position=var_44)
    var_46 = 4
    var_47 = [var_37, var_46, var_39]
    var_48 = module_0.Position(var_3, var_3, var_4)
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_47, position=var_48)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, start_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_26)
    var_29 = module_0.Message(text=var_0, end_position=var_23)
    var_30 = module_0.Message(text=var_0, end_position=var_26)
    var_31 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_32 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_33 = module_0.Message(text=var_0, position=var_23)
    var_34 = module_0.Message(text=var_0, start_position=var_23, end_position=var_23)
    var_35 = module_0.Message(text=var_0)
    var_36 = module_0.Message(text=var_0, key=var_2)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0)
    var_40 = []
    var_41 = module_0.Message(text=var_0, index=var_40)
    var_42 = module_0.Message(text=var_0)
    var_43 = None
    var_44 = module_0.Message(text=var_0, start_position=var_43, end_position=var_43)



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = module_0.Position(var_7, var_7, var_8)
    var_10 = [var_2]
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_10, position=var_9)
    var_12 = [var_2]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12, position=var_9)
    var_14 = 'Error 1'
    var_15 = [var_2]
    var_16 = module_0.Message(text=var_14, code=var_1, index=var_15)
    var_17 = 'Error 2'
    var_18 = [var_2]
    var_19 = module_0.Message(text=var_17, code=var_1, index=var_18)
    var_20 = 'code1'
    var_21 = [var_2]
    var_22 = module_0.Message(text=var_0, code=var_20, index=var_21)
    var_23 = 'code2'
    var_24 = [var_2]
    var_25 = module_0.Message(text=var_0, code=var_23, index=var_24)
    var_26 = 'field1'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = 'field2'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_0, code=var_1, index=var_30)
    var_32 = module_0.Position(var_7, var_7, var_8)
    var_33 = 2
    var_34 = 10
    var_35 = module_0.Position(var_33, var_7, var_34)
    var_36 = [var_2]
    var_37 = module_0.Message(text=var_0, code=var_1, index=var_36, start_position=var_32)
    var_38 = [var_2]
    var_39 = module_0.Message(text=var_0, code=var_1, index=var_38, start_position=var_35)
    var_40 = [var_2]
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_40, end_position=var_32)
    var_42 = [var_2]
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_42, end_position=var_35)
    var_44 = [var_2]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)
    var_46 = [var_2]
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)
    var_48 = [var_2]
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_48)
    var_50 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_51 = [var_2]
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_51)
    var_53 = [var_2]
    var_54 = module_0.Message(text=var_0, code=var_1, index=var_53, position=var_32)
    var_55 = [var_2]
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_55, start_position=var_32, end_position=var_32)
    var_57 = [var_2]
    var_58 = module_0.Message(text=var_0, code=var_1, index=var_57)
    var_59 = [var_2]
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_59)
    var_61 = hash(var_58)
    var_62 = hash(var_60)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = 'Error 1'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'Error 2'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'code1'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'code2'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 'other'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = module_0.Position(var_3, var_3, var_4)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_22)
    var_24 = 2
    var_25 = module_0.Position(var_24, var_3, var_4)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25)
    var_27 = module_0.Position(var_3, var_3, var_4)
    var_28 = module_0.Message(text=var_0, code=var_1, end_position=var_27)
    var_29 = module_0.Position(var_3, var_24, var_4)
    var_30 = module_0.Message(text=var_0, code=var_1, end_position=var_29)
    var_31 = module_0.Position(var_3, var_3, var_4)
    var_32 = module_0.Message(text=var_0, code=var_1, position=var_31)
    var_33 = module_0.Position(var_3, var_3, var_4)
    var_34 = module_0.Position(var_3, var_3, var_4)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_34)
    var_36 = module_0.Message(text=var_0, code=var_1)
    var_37 = module_0.Message(text=var_0, code=var_1)
    var_38 = module_0.Message(text=var_0, code=var_1)
    var_39 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_40 = [var_2]
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_40)
    var_42 = 'users'
    var_43 = 'name'
    var_44 = [var_42, var_4, var_43]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)
    var_46 = [var_42, var_4, var_43]
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 0
    var_6 = module_0.Position(var_4, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_8, position=var_9)
    var_11 = 'Error 1'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'Error 2'
    var_14 = module_0.Message(text=var_13, code=var_1)
    var_15 = 'max_length'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = 'min_length'
    var_18 = module_0.Message(text=var_0, code=var_17)
    var_19 = 'field1'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 'field2'
    var_23 = [var_22]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    var_25 = module_0.Position(var_4, var_4, var_5)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25)
    var_27 = 2
    var_28 = 10
    var_29 = module_0.Position(var_27, var_4, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29)
    var_31 = module_0.Position(var_4, var_4, var_5)
    var_32 = module_0.Message(text=var_0, code=var_1, end_position=var_31)
    var_33 = module_0.Position(var_27, var_4, var_28)
    var_34 = module_0.Message(text=var_0, code=var_1, end_position=var_33)
    var_35 = module_0.Position(var_4, var_4, var_5)
    var_36 = module_0.Message(text=var_0, code=var_1, position=var_35)
    var_37 = module_0.Position(var_4, var_4, var_5)
    var_38 = module_0.Position(var_4, var_4, var_5)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_37, end_position=var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_41 = [var_2]
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_41)
    var_43 = module_0.Message(text=var_0, code=var_1)
    var_44 = module_0.Message(text=var_0, code=var_1)
    var_45 = module_0.Message(text=var_0, code=var_1)
    var_46 = []
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)
    var_48 = module_0.Message(text=var_0, code=var_1)
    var_49 = 'users'
    var_50 = 3
    var_51 = 'username'
    var_52 = [var_49, var_50, var_51]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = [var_49, var_50, var_51]
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = module_0.Position(var_4, var_4, var_5)
    var_57 = module_0.Message(text=var_0, code=var_1, start_position=var_56)
    var_58 = module_0.Message(text=var_0, code=var_1)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, start_position=var_23, end_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_26, end_position=var_23)
    var_29 = module_0.Message(text=var_0, start_position=var_23, end_position=var_23)
    var_30 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_31 = module_0.Message(text=var_0, position=var_23)
    var_32 = module_0.Message(text=var_0, position=var_23)
    var_33 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_34 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_35 = module_0.Message(text=var_0)
    var_36 = module_0.Message(text=var_0, key=var_2)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0, key=var_2)
    var_40 = module_0.Message(text=var_0, key=var_2)
    var_41 = module_0.Message(text=var_0, position=var_23)
    var_42 = module_0.Message(text=var_0)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = module_0.Position(var_3, var_4, var_5)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_8)
    var_10 = [var_2]
    var_11 = module_0.Position(var_3, var_4, var_5)
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_10, position=var_11)
    var_13 = [var_2]
    var_14 = module_0.Position(var_3, var_4, var_5)
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_13, position=var_14)
    var_16 = [var_2]
    var_17 = module_0.Position(var_3, var_4, var_5)
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_3, var_18, var_19)
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_16, start_position=var_17, end_position=var_20)
    var_22 = [var_2]
    var_23 = module_0.Position(var_3, var_4, var_5)
    var_24 = module_0.Position(var_3, var_18, var_19)
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_22, start_position=var_23, end_position=var_24)
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_27 = 'Different'
    var_28 = module_0.Message(text=var_27, code=var_1, key=var_2)
    var_29 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_30 = 'required'
    var_31 = module_0.Message(text=var_0, code=var_30, key=var_2)
    var_32 = 'field1'
    var_33 = [var_32]
    var_34 = module_0.Message(text=var_0, code=var_1, index=var_33)
    var_35 = 'field2'
    var_36 = [var_35]
    var_37 = module_0.Message(text=var_0, code=var_1, index=var_36)
    var_38 = module_0.Position(var_3, var_4, var_5)
    var_39 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_38)
    var_40 = 4
    var_41 = module_0.Position(var_4, var_5, var_40)
    var_42 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_41)
    var_43 = module_0.Position(var_3, var_4, var_5)
    var_44 = module_0.Position(var_3, var_18, var_19)
    var_45 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_43, end_position=var_44)
    var_46 = module_0.Position(var_3, var_4, var_5)
    var_47 = 7
    var_48 = module_0.Position(var_3, var_19, var_47)
    var_49 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_46, end_position=var_48)
    var_50 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_52 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_53 = module_0.Position(var_3, var_4, var_5)
    var_54 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_53)
    var_55 = module_0.Position(var_3, var_4, var_5)
    var_56 = module_0.Position(var_3, var_4, var_5)
    var_57 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_55, end_position=var_56)
    var_58 = module_0.Position(var_3, var_4, var_5)
    var_59 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_58)
    var_60 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_58)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = module_0.Position(var_3, var_4, var_5)
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_8)
    var_10 = 'Error 1'
    var_11 = module_0.Message(text=var_10, code=var_1)
    var_12 = 'Error 2'
    var_13 = module_0.Message(text=var_12, code=var_1)
    var_14 = 'max_length'
    var_15 = module_0.Message(text=var_0, code=var_14)
    var_16 = 'min_length'
    var_17 = module_0.Message(text=var_0, code=var_16)
    var_18 = 'users'
    var_19 = 0
    var_20 = 'name'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.Message(text=var_0, index=var_21)
    var_23 = [var_18, var_3, var_20]
    var_24 = module_0.Message(text=var_0, index=var_23)
    var_25 = module_0.Position(var_3, var_3, var_3)
    var_26 = 5
    var_27 = module_0.Position(var_3, var_26, var_26)
    var_28 = module_0.Message(text=var_0, start_position=var_25, end_position=var_27)
    var_29 = 6
    var_30 = module_0.Position(var_4, var_3, var_29)
    var_31 = 10
    var_32 = module_0.Position(var_4, var_26, var_31)
    var_33 = module_0.Message(text=var_0, start_position=var_30, end_position=var_32)
    var_34 = module_0.Position(var_3, var_3, var_3)
    var_35 = module_0.Position(var_3, var_26, var_26)
    var_36 = module_0.Message(text=var_0, start_position=var_34, end_position=var_35)
    var_37 = module_0.Position(var_3, var_3, var_3)
    var_38 = module_0.Position(var_3, var_31, var_31)
    var_39 = module_0.Message(text=var_0, start_position=var_37, end_position=var_38)
    var_40 = module_0.Position(var_3, var_4, var_5)
    var_41 = module_0.Message(text=var_0, position=var_40)
    var_42 = module_0.Position(var_3, var_4, var_5)
    var_43 = module_0.Position(var_3, var_4, var_5)
    var_44 = module_0.Message(text=var_0, start_position=var_42, end_position=var_43)
    var_45 = module_0.Message(text=var_0)
    var_46 = module_0.Message(text=var_0)
    var_47 = module_0.Message(text=var_0)
    var_48 = module_0.Message(text=var_0, key=var_2)
    var_49 = [var_2]
    var_50 = module_0.Message(text=var_0, index=var_49)
    var_51 = [var_2]
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_51)
    var_53 = [var_2]
    var_54 = module_0.Message(text=var_0, code=var_1, index=var_53)
    var_55 = hash(var_52)
    var_56 = hash(var_54)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = [var_2]
    var_10 = module_0.Position(var_3, var_3, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_9, position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_3, var_3, var_4)
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_12, position=var_13)
    var_15 = module_0.Position(var_3, var_3, var_4)
    var_16 = 5
    var_17 = 4
    var_18 = module_0.Position(var_3, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_15, end_position=var_18)
    var_20 = module_0.Position(var_3, var_3, var_4)
    var_21 = module_0.Position(var_3, var_16, var_17)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_20, end_position=var_21)
    var_23 = 'Error 1'
    var_24 = module_0.Message(text=var_23, code=var_1, key=var_2)
    var_25 = 'Error 2'
    var_26 = module_0.Message(text=var_25, code=var_1, key=var_2)
    var_27 = 'max_length'
    var_28 = module_0.Message(text=var_0, code=var_27, key=var_2)
    var_29 = 'min_length'
    var_30 = module_0.Message(text=var_0, code=var_29, key=var_2)
    var_31 = 'field1'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)
    var_34 = 'field2'
    var_35 = [var_34]
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_35)
    var_37 = module_0.Position(var_3, var_3, var_4)
    var_38 = module_0.Message(text=var_0, code=var_1, position=var_37)
    var_39 = 2
    var_40 = 10
    var_41 = module_0.Position(var_39, var_3, var_40)
    var_42 = module_0.Message(text=var_0, code=var_1, position=var_41)
    var_43 = module_0.Position(var_3, var_3, var_4)
    var_44 = module_0.Position(var_3, var_16, var_17)
    var_45 = module_0.Message(text=var_0, code=var_1, start_position=var_43, end_position=var_44)
    var_46 = module_0.Position(var_3, var_3, var_4)
    var_47 = 6
    var_48 = module_0.Position(var_3, var_47, var_16)
    var_49 = module_0.Message(text=var_0, code=var_1, start_position=var_46, end_position=var_48)
    var_50 = module_0.Message(text=var_0, code=var_1)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_52 = [var_2]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = module_0.Message(text=var_0, code=var_1)
    var_55 = []
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_55)
    var_57 = module_0.Message(text=var_0, code=var_1)
    var_58 = module_0.Message(text=var_0, code=var_1)
    var_59 = module_0.Position(var_3, var_3, var_4)
    var_60 = module_0.Message(text=var_0, code=var_1, position=var_59)
    var_61 = module_0.Position(var_3, var_3, var_4)
    var_62 = module_0.Position(var_3, var_3, var_4)
    var_63 = module_0.Message(text=var_0, code=var_1, start_position=var_61, end_position=var_62)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 0
    var_6 = module_0.Position(var_4, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_8, position=var_9)
    var_11 = 'Error 1'
    var_12 = [var_2]
    var_13 = module_0.Message(text=var_11, code=var_1, index=var_12)
    var_14 = 'Error 2'
    var_15 = [var_2]
    var_16 = module_0.Message(text=var_14, code=var_1, index=var_15)
    var_17 = 'max_length'
    var_18 = [var_2]
    var_19 = module_0.Message(text=var_0, code=var_17, index=var_18)
    var_20 = 'min_length'
    var_21 = [var_2]
    var_22 = module_0.Message(text=var_0, code=var_20, index=var_21)
    var_23 = 'field1'
    var_24 = [var_23]
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_24)
    var_26 = 'field2'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = module_0.Position(var_4, var_4, var_5)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29)
    var_31 = 2
    var_32 = 10
    var_33 = module_0.Position(var_31, var_4, var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_33)
    var_35 = 5
    var_36 = 4
    var_37 = module_0.Position(var_4, var_35, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, end_position=var_37)
    var_39 = 9
    var_40 = module_0.Position(var_4, var_32, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, end_position=var_40)
    var_42 = module_0.Position(var_4, var_4, var_5)
    var_43 = module_0.Message(text=var_0, code=var_1, position=var_42)
    var_44 = module_0.Position(var_4, var_4, var_5)
    var_45 = module_0.Position(var_4, var_4, var_5)
    var_46 = module_0.Message(text=var_0, code=var_1, start_position=var_44, end_position=var_45)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = module_0.Message(text=var_0, code=var_1)
    var_49 = module_0.Message(text=var_0, code=var_1)
    var_50 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_51 = [var_2]
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_51)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = []
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = 'users'
    var_57 = 3
    var_58 = 'username'
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_59)
    var_61 = [var_56, var_57, var_58]
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_61)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = 'code1'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'code2'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, start_position=var_23)
    var_28 = module_0.Message(text=var_0, start_position=var_26)
    var_29 = module_0.Message(text=var_0, end_position=var_23)
    var_30 = module_0.Message(text=var_0, end_position=var_26)
    var_31 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_32 = module_0.Message(text=var_0, start_position=var_23, end_position=var_26)
    var_33 = module_0.Message(text=var_0, position=var_23)
    var_34 = module_0.Message(text=var_0, start_position=var_23, end_position=var_23)
    var_35 = module_0.Message(text=var_0)
    var_36 = module_0.Message(text=var_0, key=var_2)
    var_37 = [var_2]
    var_38 = module_0.Message(text=var_0, index=var_37)
    var_39 = module_0.Message(text=var_0)
    var_40 = []
    var_41 = module_0.Message(text=var_0, index=var_40)
    var_42 = module_0.Message(text=var_0)
    var_43 = None
    var_44 = module_0.Message(text=var_0, start_position=var_43, end_position=var_43)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 1
    var_10 = 0
    var_11 = module_0.Position(var_9, var_9, var_10)
    var_12 = module_0.Message(text=var_0, code=var_1, position=var_11)
    var_13 = module_0.Message(text=var_0, code=var_1, position=var_11)
    var_14 = module_0.Position(var_9, var_9, var_10)
    var_15 = 5
    var_16 = 4
    var_17 = module_0.Position(var_9, var_15, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, start_position=var_14, end_position=var_17)
    var_20 = 'Error 1'
    var_21 = module_0.Message(text=var_20, code=var_1)
    var_22 = 'Error 2'
    var_23 = module_0.Message(text=var_22, code=var_1)
    var_24 = 'code1'
    var_25 = module_0.Message(text=var_0, code=var_24)
    var_26 = 'code2'
    var_27 = module_0.Message(text=var_0, code=var_26)
    var_28 = 'field1'
    var_29 = [var_28]
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_29)
    var_31 = 'field2'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)
    var_34 = module_0.Position(var_9, var_9, var_10)
    var_35 = 2
    var_36 = 10
    var_37 = module_0.Position(var_35, var_9, var_36)
    var_38 = module_0.Message(text=var_0, code=var_1, position=var_34)
    var_39 = module_0.Message(text=var_0, code=var_1, position=var_37)
    var_40 = module_0.Position(var_9, var_9, var_10)
    var_41 = module_0.Position(var_35, var_9, var_36)
    var_42 = module_0.Position(var_9, var_15, var_16)
    var_43 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_42)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_41, end_position=var_42)
    var_45 = module_0.Position(var_9, var_9, var_10)
    var_46 = module_0.Position(var_9, var_15, var_16)
    var_47 = 9
    var_48 = module_0.Position(var_9, var_36, var_47)
    var_49 = module_0.Message(text=var_0, code=var_1, start_position=var_45, end_position=var_46)
    var_50 = module_0.Message(text=var_0, code=var_1, start_position=var_45, end_position=var_48)
    var_51 = module_0.Message(text=var_0, code=var_1)
    var_52 = module_0.Message(text=var_0, code=var_1)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = 'users'
    var_55 = 3
    var_56 = 'username'
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.Message(text=var_0, code=var_1, index=var_57)
    var_59 = [var_54, var_55, var_56]
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_59)
    var_61 = [var_2]
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_61)
    var_63 = 'subfield'
    var_64 = [var_2, var_63]
    var_65 = module_0.Message(text=var_0, code=var_1, index=var_64)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = [var_2]
    var_10 = module_0.Position(var_3, var_3, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_9, position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_3, var_3, var_4)
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_12, position=var_13)
    var_15 = 'Error 1'
    var_16 = module_0.Message(text=var_15, code=var_1, key=var_2)
    var_17 = 'Error 2'
    var_18 = module_0.Message(text=var_17, code=var_1, key=var_2)
    var_19 = 'max_length'
    var_20 = module_0.Message(text=var_0, code=var_19, key=var_2)
    var_21 = 'min_length'
    var_22 = module_0.Message(text=var_0, code=var_21, key=var_2)
    var_23 = 'field1'
    var_24 = [var_23]
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_24)
    var_26 = 'field2'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = module_0.Position(var_3, var_3, var_4)
    var_30 = 5
    var_31 = 4
    var_32 = module_0.Position(var_3, var_30, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_32)
    var_34 = 2
    var_35 = 10
    var_36 = module_0.Position(var_34, var_3, var_35)
    var_37 = 14
    var_38 = module_0.Position(var_34, var_30, var_37)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_36, end_position=var_38)
    var_40 = module_0.Position(var_3, var_3, var_4)
    var_41 = module_0.Position(var_3, var_30, var_31)
    var_42 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_41)
    var_43 = module_0.Position(var_3, var_3, var_4)
    var_44 = 9
    var_45 = module_0.Position(var_3, var_35, var_44)
    var_46 = module_0.Message(text=var_0, code=var_1, start_position=var_43, end_position=var_45)
    var_47 = module_0.Position(var_3, var_3, var_4)
    var_48 = module_0.Message(text=var_0, code=var_1, position=var_47)
    var_49 = module_0.Position(var_3, var_3, var_4)
    var_50 = module_0.Position(var_3, var_3, var_4)
    var_51 = module_0.Message(text=var_0, code=var_1, start_position=var_49, end_position=var_50)
    var_52 = module_0.Message(text=var_0, code=var_1)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = []
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = module_0.Message(text=var_0, code=var_1)
    var_57 = module_0.Message(text=var_0, code=var_1)
    var_58 = 'users'
    var_59 = 3
    var_60 = 'username'
    var_61 = [var_58, var_59, var_60]
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_61)
    var_63 = [var_58, var_59, var_60]
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7, code=var_1)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'max_length'
    var_12 = module_0.Message(text=var_0, code=var_11)
    var_13 = 'min_length'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'field1'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 'field2'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = 0
    var_23 = module_0.Position(var_21, var_21, var_22)
    var_24 = 2
    var_25 = 10
    var_26 = module_0.Position(var_24, var_21, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_23)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_29 = module_0.Position(var_21, var_21, var_22)
    var_30 = 5
    var_31 = 4
    var_32 = module_0.Position(var_21, var_30, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_32)
    var_35 = module_0.Position(var_21, var_21, var_22)
    var_36 = module_0.Position(var_21, var_30, var_31)
    var_37 = module_0.Position(var_24, var_21, var_25)
    var_38 = 14
    var_39 = module_0.Position(var_24, var_30, var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, start_position=var_35, end_position=var_36)
    var_41 = module_0.Message(text=var_0, code=var_1, start_position=var_37, end_position=var_39)
    var_42 = module_0.Message(text=var_0, code=var_1)
    var_43 = module_0.Message(text=var_0, code=var_1)
    var_44 = module_0.Position(var_21, var_21, var_22)
    var_45 = module_0.Message(text=var_0, code=var_1, position=var_44)
    var_46 = module_0.Message(text=var_0, code=var_1, start_position=var_44, end_position=var_44)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_49 = [var_2]
    var_50 = module_0.Message(text=var_0, code=var_1, index=var_49)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_15)
    var_52 = [var_18]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = []
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = module_0.Message(text=var_0, code=var_1)
    var_57 = 'users'
    var_58 = 'name'
    var_59 = [var_57, var_22, var_58]
    var_60 = module_0.Message(text=var_0, code=var_1, index=var_59)
    var_61 = [var_57, var_22, var_58]
    var_62 = module_0.Message(text=var_0, code=var_1, index=var_61)
    var_63 = [var_57, var_22, var_58]
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)
    var_65 = [var_57, var_21, var_58]
    var_66 = module_0.Message(text=var_0, code=var_1, index=var_65)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = 'Error 1'
    var_10 = module_0.Message(text=var_9, code=var_1)
    var_11 = 'Error 2'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'code1'
    var_14 = module_0.Message(text=var_0, code=var_13)
    var_15 = 'code2'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = 'other'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = module_0.Position(var_3, var_3, var_4)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_22)
    var_24 = 2
    var_25 = module_0.Position(var_24, var_3, var_4)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25)
    var_27 = module_0.Position(var_3, var_3, var_4)
    var_28 = module_0.Message(text=var_0, code=var_1, end_position=var_27)
    var_29 = module_0.Position(var_3, var_24, var_4)
    var_30 = module_0.Message(text=var_0, code=var_1, end_position=var_29)
    var_31 = module_0.Position(var_3, var_3, var_4)
    var_32 = module_0.Message(text=var_0, code=var_1, position=var_31)
    var_33 = module_0.Position(var_3, var_3, var_4)
    var_34 = module_0.Position(var_3, var_3, var_4)
    var_35 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_34)
    var_36 = module_0.Message(text=var_0, code=var_1)
    var_37 = module_0.Message(text=var_0, code=var_1)
    var_38 = module_0.Message(text=var_0, code=var_1)
    var_39 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_40 = [var_2]
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_40)
    var_42 = 'users'
    var_43 = 'name'
    var_44 = [var_42, var_4, var_43]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)
    var_46 = [var_42, var_4, var_43]
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 0
    var_6 = module_0.Position(var_4, var_4, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_6)
    var_8 = [var_2]
    var_9 = module_0.Position(var_4, var_4, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_8, position=var_9)
    var_11 = 'Error 1'
    var_12 = module_0.Message(text=var_11, code=var_1)
    var_13 = 'Error 2'
    var_14 = module_0.Message(text=var_13, code=var_1)
    var_15 = 'max_length'
    var_16 = module_0.Message(text=var_0, code=var_15)
    var_17 = 'min_length'
    var_18 = module_0.Message(text=var_0, code=var_17)
    var_19 = 'field1'
    var_20 = [var_19]
    var_21 = module_0.Message(text=var_0, code=var_1, index=var_20)
    var_22 = 'field2'
    var_23 = [var_22]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    var_25 = module_0.Position(var_4, var_4, var_5)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_25)
    var_27 = 2
    var_28 = 10
    var_29 = module_0.Position(var_27, var_4, var_28)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29)
    var_31 = module_0.Position(var_4, var_4, var_5)
    var_32 = module_0.Message(text=var_0, code=var_1, end_position=var_31)
    var_33 = module_0.Position(var_27, var_4, var_28)
    var_34 = module_0.Message(text=var_0, code=var_1, end_position=var_33)
    var_35 = module_0.Position(var_4, var_4, var_5)
    var_36 = module_0.Message(text=var_0, code=var_1, position=var_35)
    var_37 = module_0.Position(var_4, var_4, var_5)
    var_38 = module_0.Position(var_4, var_4, var_5)
    var_39 = module_0.Message(text=var_0, code=var_1, start_position=var_37, end_position=var_38)
    var_40 = module_0.Message(text=var_0, code=var_1)
    var_41 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_42 = [var_2]
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_42)
    var_44 = module_0.Message(text=var_0, code=var_1)
    var_45 = []
    var_46 = module_0.Message(text=var_0, code=var_1, index=var_45)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = None
    var_49 = module_0.Message(text=var_0, code=var_1, start_position=var_48, end_position=var_48)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_2)
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_7, position=var_8)
    var_10 = 'Error 1'
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_10, code=var_1, index=var_11)
    var_13 = 'Error 2'
    var_14 = [var_2]
    var_15 = module_0.Message(text=var_13, code=var_1, index=var_14)
    var_16 = 'max_length'
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, code=var_16, index=var_17)
    var_19 = 'min_length'
    var_20 = [var_2]
    var_21 = module_0.Message(text=var_0, code=var_19, index=var_20)
    var_22 = [var_2]
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_22)
    var_24 = [var_4]
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_24)
    var_26 = [var_2]
    var_27 = module_0.Position(var_4, var_4, var_2)
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_26, start_position=var_27)
    var_29 = [var_2]
    var_30 = 2
    var_31 = module_0.Position(var_4, var_30, var_4)
    var_32 = module_0.Message(text=var_0, code=var_1, index=var_29, start_position=var_31)
    var_33 = [var_2]
    var_34 = module_0.Position(var_4, var_4, var_2)
    var_35 = module_0.Message(text=var_0, code=var_1, index=var_33, end_position=var_34)
    var_36 = [var_2]
    var_37 = module_0.Position(var_4, var_30, var_4)
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_36, end_position=var_37)
    var_39 = [var_2]
    var_40 = module_0.Position(var_4, var_4, var_2)
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_39, position=var_40)
    var_42 = [var_2]
    var_43 = module_0.Position(var_4, var_4, var_2)
    var_44 = module_0.Position(var_4, var_4, var_2)
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_42, start_position=var_43, end_position=var_44)
    var_46 = [var_2]
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)
    var_48 = [var_2]
    var_49 = module_0.Message(text=var_0, code=var_1, index=var_48)
    var_50 = [var_2]
    var_51 = module_0.Message(text=var_0, code=var_1, index=var_50)
    var_52 = 'username'
    var_53 = module_0.Message(text=var_0, code=var_1, key=var_52)
    var_54 = [var_52]
    var_55 = module_0.Message(text=var_0, code=var_1, index=var_54)
    var_56 = 'users'
    var_57 = 3
    var_58 = [var_56, var_57, var_52]
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58)
    var_60 = [var_56, var_57, var_52]
    var_61 = module_0.Message(text=var_0, code=var_1, index=var_60)
    var_62 = [var_2]
    var_63 = module_0.Position(var_4, var_4, var_2)
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_62, position=var_63)
    var_65 = [var_2]
    var_66 = module_0.Position(var_4, var_4, var_2)
    var_67 = module_0.Position(var_4, var_30, var_4)
    var_68 = module_0.Message(text=var_0, code=var_1, index=var_65, start_position=var_66, end_position=var_67)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_5)
    var_7 = module_0.Position(var_3, var_3, var_4)
    var_8 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_7)
    var_9 = [var_2]
    var_10 = module_0.Position(var_3, var_3, var_4)
    var_11 = module_0.Message(text=var_0, code=var_1, index=var_9, position=var_10)
    var_12 = [var_2]
    var_13 = module_0.Position(var_3, var_3, var_4)
    var_14 = module_0.Message(text=var_0, code=var_1, index=var_12, position=var_13)
    var_15 = module_0.Position(var_3, var_3, var_4)
    var_16 = 5
    var_17 = 4
    var_18 = module_0.Position(var_3, var_16, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_15, end_position=var_18)
    var_20 = module_0.Position(var_3, var_3, var_4)
    var_21 = module_0.Position(var_3, var_16, var_17)
    var_22 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_20, end_position=var_21)
    var_23 = 'Error 1'
    var_24 = module_0.Message(text=var_23, code=var_1, key=var_2)
    var_25 = 'Error 2'
    var_26 = module_0.Message(text=var_25, code=var_1, key=var_2)
    var_27 = 'max_length'
    var_28 = module_0.Message(text=var_0, code=var_27, key=var_2)
    var_29 = 'min_length'
    var_30 = module_0.Message(text=var_0, code=var_29, key=var_2)
    var_31 = 'field1'
    var_32 = [var_31]
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_32)
    var_34 = 'field2'
    var_35 = [var_34]
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_35)
    var_37 = module_0.Position(var_3, var_3, var_4)
    var_38 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_37)
    var_39 = 2
    var_40 = 10
    var_41 = module_0.Position(var_39, var_3, var_40)
    var_42 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_41)
    var_43 = module_0.Position(var_3, var_3, var_4)
    var_44 = module_0.Position(var_3, var_16, var_17)
    var_45 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_43, end_position=var_44)
    var_46 = module_0.Position(var_39, var_3, var_40)
    var_47 = module_0.Position(var_3, var_16, var_17)
    var_48 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_46, end_position=var_47)
    var_49 = module_0.Position(var_3, var_3, var_4)
    var_50 = module_0.Position(var_3, var_16, var_17)
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_49, end_position=var_50)
    var_52 = module_0.Position(var_3, var_3, var_4)
    var_53 = 9
    var_54 = module_0.Position(var_3, var_40, var_53)
    var_55 = module_0.Message(text=var_0, code=var_1, key=var_2, start_position=var_52, end_position=var_54)
    var_56 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_57 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_58 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_59 = module_0.Message(text=var_0, code=var_1)
    var_60 = module_0.Message(text=var_0, code=var_1)
    var_61 = 'users'
    var_62 = 3
    var_63 = 'username'
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.Message(text=var_0, code=var_1, index=var_64)
    var_66 = [var_61, var_62, var_63]
    var_67 = module_0.Message(text=var_0, code=var_1, index=var_66)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 0
    var_3 = [var_2]
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_2)
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_5)
    var_7 = [var_2]
    var_8 = module_0.Position(var_4, var_4, var_2)
    var_9 = module_0.Message(text=var_0, code=var_1, index=var_7, position=var_8)
    var_10 = 'Error 1'
    var_11 = [var_2]
    var_12 = module_0.Message(text=var_10, code=var_1, index=var_11)
    var_13 = 'Error 2'
    var_14 = [var_2]
    var_15 = module_0.Message(text=var_13, code=var_1, index=var_14)
    var_16 = 'code1'
    var_17 = [var_2]
    var_18 = module_0.Message(text=var_0, code=var_16, index=var_17)
    var_19 = 'code2'
    var_20 = [var_2]
    var_21 = module_0.Message(text=var_0, code=var_19, index=var_20)
    var_22 = [var_2]
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_22)
    var_24 = [var_4]
    var_25 = module_0.Message(text=var_0, code=var_1, index=var_24)
    var_26 = [var_2]
    var_27 = module_0.Position(var_4, var_4, var_2)
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_26, start_position=var_27)
    var_29 = [var_2]
    var_30 = 2
    var_31 = 10
    var_32 = module_0.Position(var_30, var_4, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, index=var_29, start_position=var_32)
    var_34 = [var_2]
    var_35 = module_0.Position(var_4, var_4, var_2)
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_34, end_position=var_35)
    var_37 = [var_2]
    var_38 = module_0.Position(var_30, var_4, var_31)
    var_39 = module_0.Message(text=var_0, code=var_1, index=var_37, end_position=var_38)
    var_40 = [var_2]
    var_41 = module_0.Position(var_4, var_4, var_2)
    var_42 = module_0.Message(text=var_0, code=var_1, index=var_40, position=var_41)
    var_43 = [var_2]
    var_44 = module_0.Position(var_4, var_4, var_2)
    var_45 = module_0.Position(var_4, var_4, var_2)
    var_46 = module_0.Message(text=var_0, code=var_1, index=var_43, start_position=var_44, end_position=var_45)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = module_0.Message(text=var_0, code=var_1)
    var_49 = module_0.Message(text=var_0, code=var_1)
    var_50 = 'field'
    var_51 = module_0.Message(text=var_0, code=var_1, key=var_50)
    var_52 = [var_50]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = 'users'
    var_55 = 'name'
    var_56 = [var_54, var_2, var_55]
    var_57 = module_0.Message(text=var_0, code=var_1, index=var_56)
    var_58 = [var_54, var_2, var_55]
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58)
    var_60 = [var_2]
    var_61 = module_0.Position(var_4, var_4, var_2)
    var_62 = 5
    var_63 = 4
    var_64 = module_0.Position(var_4, var_62, var_63)
    var_65 = module_0.Message(text=var_0, code=var_1, index=var_60, start_position=var_61, end_position=var_64)
    var_66 = [var_2]
    var_67 = module_0.Position(var_30, var_4, var_31)
    var_68 = 14
    var_69 = module_0.Position(var_30, var_62, var_68)
    var_70 = module_0.Message(text=var_0, code=var_1, index=var_66, start_position=var_67, end_position=var_69)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'max_length'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_2]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 1
    var_10 = 5
    var_11 = 10
    var_12 = module_0.Position(var_9, var_10, var_11)
    var_13 = module_0.Message(text=var_0, code=var_1, position=var_12)
    var_14 = module_0.Message(text=var_0, code=var_1, position=var_12)
    var_15 = module_0.Position(var_9, var_10, var_11)
    var_16 = 15
    var_17 = module_0.Position(var_9, var_11, var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_17)
    var_20 = 'Error 1'
    var_21 = module_0.Message(text=var_20, code=var_1)
    var_22 = 'Error 2'
    var_23 = module_0.Message(text=var_22, code=var_1)
    var_24 = module_0.Message(text=var_0, code=var_1)
    var_25 = 'min_length'
    var_26 = module_0.Message(text=var_0, code=var_25)
    var_27 = [var_2]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = 'email'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_0, code=var_1, index=var_30)
    var_32 = module_0.Position(var_9, var_10, var_11)
    var_33 = 2
    var_34 = 20
    var_35 = module_0.Position(var_33, var_10, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, position=var_32)
    var_37 = module_0.Message(text=var_0, code=var_1, position=var_35)
    var_38 = module_0.Position(var_9, var_10, var_11)
    var_39 = module_0.Position(var_9, var_11, var_16)
    var_40 = module_0.Position(var_33, var_10, var_34)
    var_41 = 25
    var_42 = module_0.Position(var_33, var_11, var_41)
    var_43 = module_0.Message(text=var_0, code=var_1, start_position=var_38, end_position=var_39)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_40, end_position=var_42)
    var_45 = module_0.Message(text=var_0, code=var_1)
    var_46 = module_0.Message(text=var_0, code=var_1, position=var_32)
    var_47 = module_0.Message(text=var_0)
    var_48 = 'custom'
    var_49 = module_0.Message(text=var_0, code=var_48)
    var_50 = module_0.Message(text=var_0, code=var_1)
    var_51 = []
    var_52 = module_0.Message(text=var_0, code=var_1, index=var_51)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = 'users'
    var_55 = 3
    var_56 = [var_54, var_55, var_2]
    var_57 = module_0.Message(text=var_0, code=var_1, index=var_56)
    var_58 = [var_54, var_55, var_2]
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58)



