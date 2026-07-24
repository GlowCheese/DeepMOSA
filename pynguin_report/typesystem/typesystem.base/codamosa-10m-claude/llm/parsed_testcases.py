####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Test error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 42
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'key'
    var_10 = 'nested'
    var_11 = 'value'
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_9: var_11, var_10: var_15}
    var_17 = module_0.ValidationResult(value=var_16)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationError(messages=var_16)
    var_19 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_20 = 'Different Error'
    var_21 = module_0.Message(text=var_20, code=var_13, key=var_14)
    var_22 = [var_19, var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = module_0.ParseError(text=var_0, code=var_1)
    var_25 = 1
    var_26 = 5
    var_27 = 4
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = 'Error with position'
    var_30 = 'pos_code'
    var_31 = module_0.ValidationError(text=var_29, code=var_30, position=var_28)
    var_32 = module_0.ValidationError(text=var_29, code=var_30, position=var_28)
    var_33 = 2
    var_34 = 10
    var_35 = 9
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = module_0.ValidationError(text=var_29, code=var_30, position=var_36)
    var_38 = 'Nested error'
    var_39 = 'nested'
    var_40 = 'users'
    var_41 = 0
    var_42 = 'name'
    var_43 = [var_40, var_41, var_42]
    var_44 = module_0.Message(text=var_38, code=var_39, index=var_43)
    var_45 = [var_44]
    var_46 = module_0.ValidationError(messages=var_45)
    var_47 = module_0.ValidationError(messages=var_45)
    var_48 = [var_40, var_25, var_42]
    var_49 = module_0.Message(text=var_38, code=var_39, index=var_48)
    var_50 = [var_49]
    var_51 = module_0.ValidationError(messages=var_50)



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = 'Different'
    var_24 = module_0.Message(text=var_23, code=var_13, key=var_14)
    var_25 = [var_22, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_28 = [var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = 'parse_code'
    var_32 = module_0.ParseError(text=var_30, code=var_31)
    var_33 = module_0.ParseError(text=var_30, code=var_31)
    var_34 = 1
    var_35 = 5
    var_36 = 4
    var_37 = module_0.Position(var_34, var_35, var_36)
    var_38 = module_0.Position(var_34, var_35, var_36)
    var_39 = 'Error'
    var_40 = 'test'
    var_41 = module_0.ValidationError(text=var_39, code=var_40, position=var_37)
    var_42 = module_0.ValidationError(text=var_39, code=var_40, position=var_38)
    var_43 = 2
    var_44 = 10
    var_45 = 15
    var_46 = module_0.Position(var_43, var_44, var_45)
    var_47 = module_0.ValidationError(text=var_39, code=var_40, position=var_46)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'test error'
    var_3 = 'test_code'
    var_4 = module_0.ValidationError(text=var_2, code=var_3)
    var_5 = module_0.ValidationResult(error=var_4)
    var_6 = 42
    var_7 = module_0.ValidationResult(value=var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'error message'
    var_11 = module_0.ValidationError(text=var_10)
    var_12 = module_0.ValidationResult(error=var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = module_0.ValidationResult(value=var_17)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationResult(error=var_2)
    var_4 = repr(var_3)
    var_5 = 'ValidationResult(error='
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.ValidationResult(value=var_8)
    var_10 = repr(var_9)
    assert var_10 == "ValidationResult(value={'key': 'value'})"
    var_11 = None
    var_12 = module_0.ValidationResult(value=var_11)
    var_13 = repr(var_12)
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = 'test_string'
    var_15 = module_0.ValidationResult(value=var_14)
    var_16 = repr(var_15)
    assert var_16 == "ValidationResult(value='test_string')"
    var_17 = 42
    var_18 = module_0.ValidationResult(value=var_17)
    var_19 = repr(var_18)
    assert var_19 == 'ValidationResult(value=42)'
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.ValidationResult(value=var_23)
    var_25 = repr(var_24)
    assert var_25 == 'ValidationResult(value=[1, 2, 3])'



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 3
    var_12 = 7
    var_13 = 20
    var_14 = module_0.Position(var_11, var_12, var_13)
    var_15 = 0
    var_16 = module_0.Position(var_15, var_15, var_15)
    var_17 = module_0.Position(var_15, var_15, var_15)
    var_18 = -1
    var_19 = -1
    var_20 = -1
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = -1
    var_23 = -1
    var_24 = -1
    var_25 = module_0.Position(var_22, var_23, var_24)



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
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 20
    var_12 = 100
    var_13 = module_0.Position(var_2, var_11, var_12)
    var_14 = 0
    var_15 = module_0.Position(var_14, var_14, var_14)
    var_16 = module_0.Position(var_14, var_14, var_14)
    var_17 = -1
    var_18 = -5
    var_19 = -10
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = -1
    var_22 = -5
    var_23 = -10
    var_24 = module_0.Position(var_21, var_22, var_23)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = module_0.Position(var_5, var_7, var_9)
    var_12 = 0
    var_13 = module_0.Position(var_12, var_12, var_12)
    var_14 = module_0.Position(var_12, var_12, var_12)
    var_15 = -1
    var_16 = -5
    var_17 = -10
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = -1
    var_20 = -5
    var_21 = -10
    var_22 = module_0.Position(var_19, var_20, var_21)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Test error'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = 'Error 3'
    var_21 = 'code3'
    var_22 = 'field3'
    var_23 = module_0.Message(text=var_20, code=var_21, key=var_22)
    var_24 = [var_11, var_15, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = [var_15, var_11]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = 'Parse error'
    var_29 = 'parse_code'
    var_30 = module_0.ParseError(text=var_28, code=var_29)
    var_31 = module_0.ParseError(text=var_28, code=var_29)
    var_32 = 1
    var_33 = 5
    var_34 = 4
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = 'Error at position'
    var_37 = 'pos_code'
    var_38 = module_0.Message(text=var_36, code=var_37, position=var_35)
    var_39 = module_0.Message(text=var_36, code=var_37, position=var_35)
    var_40 = [var_38]
    var_41 = module_0.ValidationError(messages=var_40)
    var_42 = [var_39]
    var_43 = module_0.ValidationError(messages=var_42)
    var_44 = 2
    var_45 = 10
    var_46 = 9
    var_47 = module_0.Position(var_44, var_45, var_46)
    var_48 = module_0.Message(text=var_36, code=var_37, position=var_47)
    var_49 = [var_48]
    var_50 = module_0.ValidationError(messages=var_49)
    var_51 = 'Nested error'
    var_52 = 'nested_code'
    var_53 = 'users'
    var_54 = 0
    var_55 = 'email'
    var_56 = [var_53, var_54, var_55]
    var_57 = module_0.Message(text=var_51, code=var_52, index=var_56)
    var_58 = [var_53, var_54, var_55]
    var_59 = module_0.Message(text=var_51, code=var_52, index=var_58)
    var_60 = [var_57]
    var_61 = module_0.ValidationError(messages=var_60)
    var_62 = [var_59]
    var_63 = module_0.ValidationError(messages=var_62)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field1'
    var_9 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_10 = 'field2'
    var_11 = module_0.Message(text=var_0, code=var_1, key=var_10)
    var_12 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_13 = module_0.Message(text=var_0, code=var_1, key=var_8)
    var_14 = 'users'
    var_15 = 0
    var_16 = 'name'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = [var_14, var_15, var_16]
    var_20 = module_0.Message(text=var_0, code=var_1, index=var_19)
    var_21 = 1
    var_22 = [var_14, var_21, var_16]
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_22)
    var_24 = 5
    var_25 = 4
    var_26 = module_0.Position(var_21, var_24, var_25)
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_26)
    var_29 = 2
    var_30 = 10
    var_31 = module_0.Position(var_29, var_24, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, position=var_31)
    var_33 = module_0.Position(var_21, var_21, var_15)
    var_34 = 9
    var_35 = module_0.Position(var_21, var_30, var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_35)
    var_38 = 15
    var_39 = 14
    var_40 = module_0.Position(var_21, var_38, var_39)
    var_41 = module_0.Message(text=var_0, code=var_1, start_position=var_33, end_position=var_40)
    var_42 = module_0.Message(text=var_0)
    var_43 = 'custom'
    var_44 = module_0.Message(text=var_0, code=var_43)
    var_45 = 'Complex error'
    var_46 = 'complex'
    var_47 = 'field'
    var_48 = [var_47, var_15]
    var_49 = module_0.Message(text=var_45, code=var_46, index=var_48, start_position=var_33, end_position=var_35)
    var_50 = [var_47, var_15]
    var_51 = module_0.Message(text=var_45, code=var_46, index=var_50, start_position=var_33, end_position=var_35)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 50
    var_12 = module_0.Position(var_1, var_2, var_11)
    var_13 = 0
    var_14 = module_0.Position(var_13, var_13, var_13)
    var_15 = module_0.Position(var_13, var_13, var_13)
    var_16 = -1
    var_17 = -1
    var_18 = -1
    var_19 = module_0.Position(var_16, var_17, var_18)
    var_20 = -1
    var_21 = -1
    var_22 = -1
    var_23 = module_0.Position(var_20, var_21, var_22)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = module_0.Position(var_1, var_1, var_2)
    var_6 = 3
    var_7 = module_0.Position(var_0, var_6, var_2)
    var_8 = 6
    var_9 = module_0.Position(var_0, var_1, var_8)
    var_10 = 10
    var_11 = 20
    var_12 = 100
    var_13 = module_0.Position(var_10, var_11, var_12)
    var_14 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = 'field2'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = 1
    var_15 = 0
    var_16 = module_0.Position(var_14, var_15, var_15)
    var_17 = 2
    var_18 = 5
    var_19 = module_0.Position(var_17, var_15, var_18)
    var_20 = module_0.Message(text=var_0, code=var_1, start_position=var_16)
    var_21 = module_0.Message(text=var_0, code=var_1, start_position=var_19)
    var_22 = module_0.Message(text=var_0, code=var_1, end_position=var_16)
    var_23 = module_0.Message(text=var_0, code=var_1, end_position=var_19)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_25 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_26 = 'field'
    var_27 = [var_26]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27, start_position=var_16, end_position=var_19)
    var_29 = [var_26]
    var_30 = module_0.Message(text=var_0, code=var_1, index=var_29, start_position=var_16, end_position=var_19)
    var_31 = module_0.Message(text=var_0)
    var_32 = 'custom'
    var_33 = module_0.Message(text=var_0, code=var_32)
    var_34 = 'username'
    var_35 = module_0.Message(text=var_0, key=var_34)
    var_36 = [var_34]
    var_37 = module_0.Message(text=var_0, index=var_36)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'key1'
    var_9 = module_0.ValidationError(text=var_0, code=var_1, key=var_8)
    var_10 = 'key2'
    var_11 = module_0.ValidationError(text=var_0, code=var_1, key=var_10)
    var_12 = 'Error 1'
    var_13 = 'code1'
    var_14 = 'field1'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = 'Error 2'
    var_17 = 'code2'
    var_18 = 'field2'
    var_19 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_20 = [var_15, var_19]
    var_21 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_22 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_23 = [var_21, var_22]
    var_24 = module_0.ValidationError(messages=var_20)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_27 = [var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_30 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_31 = [var_29, var_30]
    var_32 = module_0.ValidationError(messages=var_31)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = 'Error'
    var_38 = 'code'
    var_39 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_40 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_41 = 2
    var_42 = 10
    var_43 = 9
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.ValidationError(text=var_37, code=var_38, position=var_44)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = [var_10]
    var_20 = module_0.Message(text=var_8, code=var_9, index=var_19)
    var_21 = [var_15]
    var_22 = module_0.Message(text=var_13, code=var_14, index=var_21)
    var_23 = [var_20, var_22]
    var_24 = module_0.ValidationError(messages=var_18)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = [var_10]
    var_27 = module_0.Message(text=var_8, code=var_9, index=var_26)
    var_28 = 'Different Error'
    var_29 = [var_15]
    var_30 = module_0.Message(text=var_28, code=var_14, index=var_29)
    var_31 = [var_27, var_30]
    var_32 = module_0.ValidationError(messages=var_31)
    var_33 = [var_10]
    var_34 = module_0.Message(text=var_8, code=var_9, index=var_33)
    var_35 = [var_34]
    var_36 = module_0.ValidationError(messages=var_35)
    var_37 = 'Parse error'
    var_38 = 'parse_code'
    var_39 = module_0.ParseError(text=var_37, code=var_38)
    var_40 = module_0.ParseError(text=var_37, code=var_38)
    var_41 = 1
    var_42 = 5
    var_43 = 4
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.Position(var_41, var_42, var_43)
    var_46 = 'Error'
    var_47 = 'code'
    var_48 = module_0.ValidationError(text=var_46, code=var_47, position=var_44)
    var_49 = module_0.ValidationError(text=var_46, code=var_47, position=var_45)
    var_50 = 2
    var_51 = 10
    var_52 = 20
    var_53 = module_0.Position(var_50, var_51, var_52)
    var_54 = module_0.ValidationError(text=var_46, code=var_47, position=var_53)



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Different'
    var_27 = module_0.Message(text=var_26, code=var_13, key=var_14)
    var_28 = [var_25, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = 'parse_code'
    var_32 = module_0.ParseError(text=var_30, code=var_31)
    var_33 = module_0.ParseError(text=var_30, code=var_31)
    var_34 = module_0.ValidationError(text=var_30, code=var_31)
    var_35 = 1
    var_36 = 5
    var_37 = 4
    var_38 = module_0.Position(var_35, var_36, var_37)
    var_39 = module_0.Position(var_35, var_36, var_37)
    var_40 = 'Error'
    var_41 = 'code'
    var_42 = module_0.ValidationError(text=var_40, code=var_41, position=var_38)
    var_43 = module_0.ValidationError(text=var_40, code=var_41, position=var_39)
    var_44 = 2
    var_45 = 10
    var_46 = 15
    var_47 = module_0.Position(var_44, var_45, var_46)
    var_48 = module_0.ValidationError(text=var_40, code=var_41, position=var_47)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = 'Different error'
    var_23 = module_0.Message(text=var_22, code=var_9, key=var_10)
    var_24 = [var_23, var_15]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = 'Parse error'
    var_27 = 'parse_code'
    var_28 = module_0.ParseError(text=var_26, code=var_27)
    var_29 = module_0.ParseError(text=var_26, code=var_27)
    var_30 = 1
    var_31 = 5
    var_32 = 4
    var_33 = module_0.Position(var_30, var_31, var_32)
    var_34 = 'Error at position'
    var_35 = 'pos_code'
    var_36 = module_0.ValidationError(text=var_34, code=var_35, position=var_33)
    var_37 = module_0.ValidationError(text=var_34, code=var_35, position=var_33)
    var_38 = 2
    var_39 = 10
    var_40 = 9
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.ValidationError(text=var_34, code=var_35, position=var_41)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = module_0.Message(text=var_4, code=var_13, key=var_14)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = module_0.Position(var_33, var_34, var_35)
    var_38 = 'Error'
    var_39 = 'code'
    var_40 = module_0.ValidationError(text=var_38, code=var_39, position=var_36)
    var_41 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_42 = 2
    var_43 = 10
    var_44 = 15
    var_45 = module_0.Position(var_42, var_43, var_44)
    var_46 = module_0.ValidationError(text=var_38, code=var_39, position=var_45)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = 'field2'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = [var_8]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = [var_8]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 5
    var_20 = 4
    var_21 = module_0.Position(var_18, var_19, var_20)
    var_22 = module_0.Position(var_18, var_19, var_20)
    var_23 = 'Error'
    var_24 = 'code'
    var_25 = module_0.Message(text=var_23, code=var_24, position=var_21)
    var_26 = module_0.Message(text=var_23, code=var_24, position=var_22)
    var_27 = 0
    var_28 = module_0.Position(var_18, var_18, var_27)
    var_29 = 10
    var_30 = 9
    var_31 = module_0.Position(var_18, var_29, var_30)
    var_32 = module_0.Message(text=var_23, code=var_24, start_position=var_28, end_position=var_31)
    var_33 = module_0.Message(text=var_23, code=var_24, start_position=var_28, end_position=var_31)
    var_34 = 2
    var_35 = 20
    var_36 = module_0.Position(var_34, var_18, var_35)
    var_37 = module_0.Message(text=var_23, code=var_24, start_position=var_36, end_position=var_31)
    var_38 = module_0.Message(text=var_23, code=var_24, start_position=var_28, end_position=var_36)
    var_39 = 'username'
    var_40 = module_0.Message(text=var_23, code=var_24, key=var_39)
    var_41 = module_0.Message(text=var_23, code=var_24, key=var_39)
    var_42 = module_0.Message(text=var_0)
    var_43 = 'custom'
    var_44 = module_0.Message(text=var_0, code=var_43)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Different Error'
    var_27 = module_0.Message(text=var_26, code=var_13, key=var_14)
    var_28 = [var_25, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = 'parse_code'
    var_32 = module_0.ParseError(text=var_30, code=var_31)
    var_33 = module_0.ParseError(text=var_30, code=var_31)
    var_34 = 1
    var_35 = 5
    var_36 = 4
    var_37 = module_0.Position(var_34, var_35, var_36)
    var_38 = 'Error with position'
    var_39 = 'pos_code'
    var_40 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_41 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_42 = 2
    var_43 = 10
    var_44 = 9
    var_45 = module_0.Position(var_42, var_43, var_44)
    var_46 = module_0.ValidationError(text=var_38, code=var_39, position=var_45)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'key1'
    var_9 = module_0.ValidationError(text=var_0, code=var_1, key=var_8)
    var_10 = 'key2'
    var_11 = module_0.ValidationError(text=var_0, code=var_1, key=var_10)
    var_12 = 'Error 1'
    var_13 = 'code1'
    var_14 = 'field1'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = 'Error 2'
    var_17 = 'code2'
    var_18 = 'field2'
    var_19 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_20 = [var_15, var_19]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_15, var_19]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Error 3'
    var_25 = 'code3'
    var_26 = 'field3'
    var_27 = module_0.Message(text=var_24, code=var_25, key=var_26)
    var_28 = [var_15, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = module_0.ParseError(text=var_30)
    var_32 = module_0.ValidationError(text=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = 'Error at position'
    var_38 = 'pos_code'
    var_39 = module_0.Message(text=var_37, code=var_38, position=var_36)
    var_40 = [var_39]
    var_41 = module_0.ValidationError(messages=var_40)
    var_42 = [var_39]
    var_43 = module_0.ValidationError(messages=var_42)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = [var_8]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = 0
    var_15 = module_0.Position(var_13, var_13, var_14)
    var_16 = 2
    var_17 = 5
    var_18 = module_0.Position(var_16, var_13, var_17)
    var_19 = module_0.Message(text=var_0, code=var_1, start_position=var_15)
    var_20 = module_0.Message(text=var_0, code=var_1, start_position=var_18)
    var_21 = module_0.Message(text=var_0, code=var_1, start_position=var_15)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_15)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_18)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_15)
    var_25 = 'username'
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_25)
    var_27 = [var_25]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_15)
    var_31 = module_0.Message(text=var_0)
    var_32 = 'custom'
    var_33 = module_0.Message(text=var_0, code=var_32)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Different Error'
    var_27 = module_0.Message(text=var_26, code=var_13, key=var_14)
    var_28 = [var_25, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = 'parse_code'
    var_32 = module_0.ParseError(text=var_30, code=var_31)
    var_33 = module_0.ParseError(text=var_30, code=var_31)
    var_34 = module_0.ValidationError(text=var_30, code=var_31)
    var_35 = 1
    var_36 = 5
    var_37 = module_0.Position(var_35, var_36, var_36)
    var_38 = 'Error'
    var_39 = 'pos_code'
    var_40 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_41 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_42 = 2
    var_43 = 10
    var_44 = module_0.Position(var_42, var_43, var_43)
    var_45 = module_0.ValidationError(text=var_38, code=var_39, position=var_44)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = 'other_field'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_0, code=var_1, index=var_12)
    var_14 = [var_8]
    var_15 = module_0.Message(text=var_0, code=var_1, index=var_14)
    var_16 = [var_8]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = 1
    var_19 = 0
    var_20 = module_0.Position(var_18, var_19, var_19)
    var_21 = 5
    var_22 = module_0.Position(var_18, var_21, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_22)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_26 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_22)
    var_29 = module_0.Message(text=var_0)
    var_30 = 'custom'
    var_31 = module_0.Message(text=var_0, code=var_30)
    var_32 = 'username'
    var_33 = module_0.Message(text=var_0, code=var_1, key=var_32)
    var_34 = [var_32]
    var_35 = module_0.Message(text=var_0, code=var_1, index=var_34)
    var_36 = module_0.Message(text=var_0, code=var_1, position=var_20)
    var_37 = module_0.Message(text=var_0, code=var_1, start_position=var_20, end_position=var_20)
    var_38 = 'users'
    var_39 = 'email'
    var_40 = [var_38, var_19, var_39]
    var_41 = module_0.Message(text=var_0, code=var_1, index=var_40)
    var_42 = [var_38, var_19, var_39]
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_42)
    var_44 = [var_38, var_18, var_39]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_15, var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_11]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Parse error'
    var_25 = 'parse_code'
    var_26 = module_0.ParseError(text=var_24, code=var_25)
    var_27 = module_0.ParseError(text=var_24, code=var_25)
    var_28 = 'Same message'
    var_29 = 'same_code'
    var_30 = module_0.ValidationError(text=var_28, code=var_29)
    var_31 = module_0.ParseError(text=var_28, code=var_29)
    var_32 = 1
    var_33 = 5
    var_34 = 10
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = 'Error'
    var_37 = 'code'
    var_38 = module_0.ValidationError(text=var_36, code=var_37, position=var_35)
    var_39 = module_0.ValidationError(text=var_36, code=var_37, position=var_35)
    var_40 = 2
    var_41 = 20
    var_42 = module_0.Position(var_40, var_34, var_41)
    var_43 = module_0.ValidationError(text=var_36, code=var_37, position=var_42)
    var_44 = 'username'
    var_45 = module_0.Message(text=var_36, code=var_37, key=var_44)
    var_46 = [var_45]
    var_47 = module_0.ValidationError(messages=var_46)
    var_48 = module_0.Message(text=var_36, code=var_37, key=var_44)
    var_49 = [var_48]
    var_50 = module_0.ValidationError(messages=var_49)
    var_51 = 'email'
    var_52 = module_0.Message(text=var_36, code=var_37, key=var_51)
    var_53 = [var_52]
    var_54 = module_0.ValidationError(messages=var_53)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = 'Error'
    var_38 = 'test'
    var_39 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_40 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_41 = 2
    var_42 = 10
    var_43 = 20
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.ValidationError(text=var_37, code=var_38, position=var_44)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = [var_8]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 'username'
    var_14 = module_0.Message(text=var_0, code=var_1, key=var_13)
    var_15 = [var_13]
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_15)
    var_17 = 1
    var_18 = 0
    var_19 = module_0.Position(var_17, var_17, var_18)
    var_20 = 2
    var_21 = module_0.Position(var_17, var_20, var_17)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_19)
    var_23 = module_0.Message(text=var_0, code=var_1, start_position=var_21)
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_19)
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_21)
    var_26 = module_0.Message(text=var_0, code=var_1, position=var_19)
    var_27 = module_0.Message(text=var_0, code=var_1, start_position=var_19, end_position=var_19)
    var_28 = 'Complex Error'
    var_29 = 'complex'
    var_30 = 'field1'
    var_31 = 'field2'
    var_32 = [var_30, var_31]
    var_33 = module_0.Message(text=var_28, code=var_29, index=var_32, start_position=var_19, end_position=var_21)
    var_34 = [var_30, var_31]
    var_35 = module_0.Message(text=var_28, code=var_29, index=var_34, start_position=var_19, end_position=var_21)
    var_36 = module_0.Message(text=var_0)
    var_37 = 'custom'
    var_38 = module_0.Message(text=var_0, code=var_37)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = 'Error 3'
    var_21 = 'code3'
    var_22 = 'field3'
    var_23 = module_0.Message(text=var_20, code=var_21, key=var_22)
    var_24 = [var_11, var_15, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = [var_15, var_11]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = 'Parse error'
    var_29 = 'parse_code'
    var_30 = module_0.ParseError(text=var_28, code=var_29)
    var_31 = module_0.ParseError(text=var_28, code=var_29)
    var_32 = 'Same message'
    var_33 = 'same_code'
    var_34 = module_0.ValidationError(text=var_32, code=var_33)
    var_35 = module_0.ParseError(text=var_32, code=var_33)
    var_36 = 1
    var_37 = 5
    var_38 = 4
    var_39 = module_0.Position(var_36, var_37, var_38)
    var_40 = 'Error with position'
    var_41 = 'pos_code'
    var_42 = module_0.Message(text=var_40, code=var_41, position=var_39)
    var_43 = module_0.Message(text=var_40, code=var_41, position=var_39)
    var_44 = [var_42]
    var_45 = module_0.ValidationError(messages=var_44)
    var_46 = [var_43]
    var_47 = module_0.ValidationError(messages=var_46)
    var_48 = 2
    var_49 = 10
    var_50 = 9
    var_51 = module_0.Position(var_48, var_49, var_50)
    var_52 = module_0.Message(text=var_40, code=var_41, position=var_51)
    var_53 = [var_52]
    var_54 = module_0.ValidationError(messages=var_53)
    var_55 = 'Nested error'
    var_56 = 'nested_code'
    var_57 = 'users'
    var_58 = 0
    var_59 = 'name'
    var_60 = [var_57, var_58, var_59]
    var_61 = module_0.Message(text=var_55, code=var_56, index=var_60)
    var_62 = [var_57, var_58, var_59]
    var_63 = module_0.Message(text=var_55, code=var_56, index=var_62)
    var_64 = [var_61]
    var_65 = module_0.ValidationError(messages=var_64)
    var_66 = [var_63]
    var_67 = module_0.ValidationError(messages=var_66)
    var_68 = [var_57, var_36, var_59]
    var_69 = module_0.Message(text=var_55, code=var_56, index=var_68)
    var_70 = [var_69]
    var_71 = module_0.ValidationError(messages=var_70)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = 'Different Error'
    var_24 = module_0.Message(text=var_23, code=var_13, key=var_14)
    var_25 = [var_22, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_28 = [var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = 'Parse error'
    var_31 = 'parse_code'
    var_32 = module_0.ParseError(text=var_30, code=var_31)
    var_33 = module_0.ParseError(text=var_30, code=var_31)
    var_34 = 'Error'
    var_35 = 'code'
    var_36 = module_0.ValidationError(text=var_34, code=var_35)
    var_37 = module_0.ParseError(text=var_34, code=var_35)
    var_38 = 1
    var_39 = 5
    var_40 = 4
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.Position(var_38, var_39, var_40)
    var_43 = 'Error at position'
    var_44 = 'pos_error'
    var_45 = module_0.Message(text=var_43, code=var_44, position=var_41)
    var_46 = module_0.Message(text=var_43, code=var_44, position=var_42)
    var_47 = [var_45]
    var_48 = module_0.ValidationError(messages=var_47)
    var_49 = [var_46]
    var_50 = module_0.ValidationError(messages=var_49)
    var_51 = 2
    var_52 = 10
    var_53 = 9
    var_54 = module_0.Position(var_51, var_52, var_53)
    var_55 = module_0.Message(text=var_43, code=var_44, position=var_54)
    var_56 = [var_55]
    var_57 = module_0.ValidationError(messages=var_56)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 'Error'
    var_34 = 'code'
    var_35 = module_0.ValidationError(text=var_33, code=var_34)
    var_36 = module_0.ParseError(text=var_33, code=var_34)
    var_37 = 1
    var_38 = 5
    var_39 = 4
    var_40 = module_0.Position(var_37, var_38, var_39)
    var_41 = module_0.ValidationError(text=var_33, code=var_34, position=var_40)
    var_42 = module_0.ValidationError(text=var_33, code=var_34, position=var_40)
    var_43 = 2
    var_44 = 10
    var_45 = 9
    var_46 = module_0.Position(var_43, var_44, var_45)
    var_47 = module_0.ValidationError(text=var_33, code=var_34, position=var_46)



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'field2'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_8 = [var_3, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    var_10 = var_9.messages()
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.messages(add_prefix=var_12)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 'users'
    var_16 = var_9.messages(add_prefix=var_15)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'Nested error'
    var_19 = 'nested'
    var_20 = 'name'
    var_21 = [var_15, var_12, var_20]
    var_22 = module_0.Message(text=var_18, code=var_19, index=var_21)
    var_23 = [var_22]
    var_24 = module_0.BaseError(messages=var_23)
    var_25 = 'data'
    var_26 = var_24.messages(add_prefix=var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = var_9.messages()
    var_29 = var_9.messages()
    var_30 = 'Root error'
    var_31 = 'root'
    var_32 = module_0.Message(text=var_30, code=var_31)
    var_33 = [var_32]
    var_34 = module_0.BaseError(messages=var_33)
    var_35 = var_34.messages()
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = var_34.messages(add_prefix=var_31)



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    var_6 = var_5.messages()
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'Error 2'
    var_9 = 'code2'
    var_10 = 'field2'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = [var_3, var_11]
    var_13 = module_0.BaseError(messages=var_12)
    var_14 = var_13.messages()
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'Nested error'
    var_17 = 'nested'
    var_18 = 'field3'
    var_19 = 'subfield'
    var_20 = [var_18, var_19]
    var_21 = module_0.Message(text=var_16, code=var_17, index=var_20)
    var_22 = [var_21]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = 'parent'
    var_25 = var_23.messages(add_prefix=var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = 'Array error'
    var_28 = 'array_err'
    var_29 = 'items'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_27, code=var_28, index=var_30)
    var_32 = [var_31]
    var_33 = module_0.BaseError(messages=var_32)
    var_34 = 0
    var_35 = var_33.messages(add_prefix=var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = 'Original'
    var_38 = 'orig'
    var_39 = 'key1'
    var_40 = module_0.Message(text=var_37, code=var_38, key=var_39)
    var_41 = [var_40]
    var_42 = module_0.BaseError(messages=var_41)
    var_43 = var_42._messages[var_34]
    var_44 = 'prefix'
    var_45 = var_42.messages(add_prefix=var_44)
    var_46 = 'Top level'
    var_47 = 'top'
    var_48 = module_0.Message(text=var_46, code=var_47)
    var_49 = [var_48]
    var_50 = module_0.BaseError(messages=var_49)
    var_51 = 'root'
    var_52 = var_50.messages(add_prefix=var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = 'Copy test'
    var_55 = 'copy'
    var_56 = 'test'
    var_57 = module_0.Message(text=var_54, code=var_55, key=var_56)
    var_58 = [var_57]
    var_59 = module_0.BaseError(messages=var_58)
    var_60 = var_59.messages()
    var_61 = var_59.messages()



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 3
    var_12 = 20
    var_13 = module_0.Position(var_11, var_2, var_12)
    var_14 = 0
    var_15 = module_0.Position(var_14, var_14, var_14)
    var_16 = module_0.Position(var_14, var_14, var_14)
    var_17 = -1
    var_18 = -1
    var_19 = -1
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = -1
    var_22 = -1
    var_23 = -1
    var_24 = module_0.Position(var_21, var_22, var_23)



# Parsed testcases at query #4
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Test error'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 42
    var_6 = module_0.ValidationResult(value=var_5)
    var_7 = iter(var_6)
    var_8 = next(var_7)
    assert var_8 == 42
    var_9 = next(var_7)
    assert var_9 is None
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_0.ValidationResult(value=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.ValidationResult(value=var_17)
    var_19 = list(var_18)
    var_20 = None
    var_21 = module_0.ValidationResult(value=var_20, error=var_20)



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = module_0.Position(var_5, var_7, var_9)
    var_12 = 0
    var_13 = module_0.Position(var_12, var_12, var_12)
    var_14 = module_0.Position(var_12, var_12, var_12)
    var_15 = -1
    var_16 = -1
    var_17 = -1
    var_18 = module_0.Position(var_15, var_16, var_17)
    var_19 = -1
    var_20 = -1
    var_21 = -1
    var_22 = module_0.Position(var_19, var_20, var_21)
    var_23 = 3
    var_24 = 7
    var_25 = 15
    var_26 = module_0.Position(var_23, var_24, var_25)
    var_27 = module_0.Position(var_23, var_24, var_25)
    var_28 = 4
    var_29 = 8
    var_30 = 20
    var_31 = module_0.Position(var_28, var_29, var_30)
    var_32 = module_0.Position(var_28, var_29, var_30)
    var_33 = module_0.Position(var_28, var_29, var_30)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = [var_10]
    var_20 = module_0.Message(text=var_8, code=var_9, index=var_19)
    var_21 = [var_15]
    var_22 = module_0.Message(text=var_13, code=var_14, index=var_21)
    var_23 = [var_20, var_22]
    var_24 = module_0.ValidationError(messages=var_18)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = [var_10]
    var_27 = module_0.Message(text=var_8, code=var_9, index=var_26)
    var_28 = [var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = [var_15]
    var_31 = module_0.Message(text=var_13, code=var_14, index=var_30)
    var_32 = [var_10]
    var_33 = module_0.Message(text=var_8, code=var_9, index=var_32)
    var_34 = [var_31, var_33]
    var_35 = module_0.ValidationError(messages=var_34)
    var_36 = 'Parse error'
    var_37 = 'parse_code'
    var_38 = module_0.ParseError(text=var_36, code=var_37)
    var_39 = module_0.ParseError(text=var_36, code=var_37)
    var_40 = 1
    var_41 = 5
    var_42 = module_0.Position(var_40, var_41, var_41)
    var_43 = 'Error with position'
    var_44 = 'pos_code'
    var_45 = module_0.ValidationError(text=var_43, code=var_44, position=var_42)
    var_46 = module_0.ValidationError(text=var_43, code=var_44, position=var_42)
    var_47 = 2
    var_48 = 10
    var_49 = module_0.Position(var_47, var_48, var_48)
    var_50 = module_0.ValidationError(text=var_43, code=var_44, position=var_49)



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
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 0
    var_12 = module_0.Position(var_11, var_11, var_11)
    var_13 = module_0.Position(var_11, var_11, var_11)
    var_14 = -1
    var_15 = -1
    var_16 = -1
    var_17 = module_0.Position(var_14, var_15, var_16)
    var_18 = -1
    var_19 = -1
    var_20 = -1
    var_21 = module_0.Position(var_18, var_19, var_20)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 6
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = module_0.Position(var_5, var_7, var_9)
    var_12 = module_0.Position(var_0, var_1, var_2)
    var_13 = 0
    var_14 = module_0.Position(var_13, var_13, var_13)
    var_15 = module_0.Position(var_13, var_13, var_13)
    var_16 = -1
    var_17 = -5
    var_18 = -10
    var_19 = module_0.Position(var_16, var_17, var_18)
    var_20 = -1
    var_21 = -5
    var_22 = -10
    var_23 = module_0.Position(var_20, var_21, var_22)



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_23 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_24 = [var_22, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = 'Error with position'
    var_38 = 'pos_code'
    var_39 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_40 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_41 = 2
    var_42 = 10
    var_43 = 9
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.ValidationError(text=var_37, code=var_38, position=var_44)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'key1'
    var_9 = module_0.ValidationError(text=var_0, code=var_1, key=var_8)
    var_10 = 'key2'
    var_11 = module_0.ValidationError(text=var_0, code=var_1, key=var_10)
    var_12 = 'Error 1'
    var_13 = 'code1'
    var_14 = module_0.Message(text=var_12, code=var_13)
    var_15 = 'Error 2'
    var_16 = 'code2'
    var_17 = module_0.Message(text=var_15, code=var_16)
    var_18 = [var_14, var_17]
    var_19 = module_0.Message(text=var_12, code=var_13)
    var_20 = module_0.Message(text=var_15, code=var_16)
    var_21 = [var_19, var_20]
    var_22 = module_0.ValidationError(messages=var_18)
    var_23 = module_0.ValidationError(messages=var_21)
    var_24 = module_0.Message(text=var_12, code=var_13)
    var_25 = [var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = 1
    var_28 = 5
    var_29 = 4
    var_30 = module_0.Position(var_27, var_28, var_29)
    var_31 = module_0.Position(var_27, var_28, var_29)
    var_32 = 'Error'
    var_33 = module_0.ValidationError(text=var_32, position=var_30)
    var_34 = module_0.ValidationError(text=var_32, position=var_31)
    var_35 = 2
    var_36 = module_0.Position(var_35, var_28, var_29)
    var_37 = module_0.ValidationError(text=var_32, position=var_36)
    var_38 = 'Parse error'
    var_39 = 'parse_code'
    var_40 = module_0.ParseError(text=var_38, code=var_39)
    var_41 = module_0.ParseError(text=var_38, code=var_39)
    var_42 = module_0.ValidationError(text=var_38, code=var_39)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationError(messages=var_16)
    var_19 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_20 = 'Different Error'
    var_21 = module_0.Message(text=var_20, code=var_13, key=var_14)
    var_22 = [var_19, var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Parse error'
    var_25 = 'parse_code'
    var_26 = module_0.ParseError(text=var_24, code=var_25)
    var_27 = module_0.ParseError(text=var_24, code=var_25)
    var_28 = module_0.ValidationError(text=var_24, code=var_25)
    var_29 = 'Nested error'
    var_30 = 'nested_code'
    var_31 = 'users'
    var_32 = 0
    var_33 = 'name'
    var_34 = [var_31, var_32, var_33]
    var_35 = module_0.Message(text=var_29, code=var_30, index=var_34)
    var_36 = [var_35]
    var_37 = [var_31, var_32, var_33]
    var_38 = module_0.Message(text=var_29, code=var_30, index=var_37)
    var_39 = [var_38]
    var_40 = module_0.ValidationError(messages=var_36)
    var_41 = module_0.ValidationError(messages=var_39)
    var_42 = 1
    var_43 = [var_31, var_42, var_33]
    var_44 = module_0.Message(text=var_29, code=var_30, index=var_43)
    var_45 = [var_44]
    var_46 = module_0.ValidationError(messages=var_45)



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = [var_8]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = 0
    var_15 = module_0.Position(var_13, var_14, var_14)
    var_16 = 2
    var_17 = module_0.Position(var_16, var_14, var_13)
    var_18 = module_0.Message(text=var_0, code=var_1, start_position=var_15)
    var_19 = module_0.Message(text=var_0, code=var_1, start_position=var_17)
    var_20 = module_0.Message(text=var_0, code=var_1, start_position=var_15)
    var_21 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_15)
    var_22 = module_0.Message(text=var_0, code=var_1, start_position=var_15, end_position=var_17)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_24 = module_0.Message(text=var_0, code=var_1, position=var_15)
    var_25 = 'username'
    var_26 = module_0.Message(text=var_0, code=var_1, key=var_25)
    var_27 = [var_25]
    var_28 = module_0.Message(text=var_0, code=var_1, index=var_27)
    var_29 = module_0.Message(text=var_0)
    var_30 = 'custom'
    var_31 = module_0.Message(text=var_0, code=var_30)
    var_32 = 'users'
    var_33 = 'name'
    var_34 = [var_32, var_14, var_33]
    var_35 = module_0.Message(text=var_0, code=var_1, index=var_34)
    var_36 = [var_32, var_14, var_33]
    var_37 = module_0.Message(text=var_0, code=var_1, index=var_36)
    var_38 = [var_32, var_14]
    var_39 = module_0.Message(text=var_0, code=var_1, index=var_38)



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Message 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Message 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = 'Parse error'
    var_23 = 'parse_code'
    var_24 = module_0.ParseError(text=var_22, code=var_23)
    var_25 = module_0.ParseError(text=var_22, code=var_23)
    var_26 = 1
    var_27 = 5
    var_28 = 4
    var_29 = module_0.Position(var_26, var_27, var_28)
    var_30 = 'Error with position'
    var_31 = 'pos_code'
    var_32 = module_0.Message(text=var_30, code=var_31, position=var_29)
    var_33 = module_0.Message(text=var_30, code=var_31, position=var_29)
    var_34 = [var_32]
    var_35 = module_0.ValidationError(messages=var_34)
    var_36 = [var_33]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = 2
    var_39 = 10
    var_40 = 9
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.Message(text=var_30, code=var_31, position=var_41)
    var_43 = [var_42]
    var_44 = module_0.ValidationError(messages=var_43)
    var_45 = 'Nested error'
    var_46 = 'nested'
    var_47 = 'users'
    var_48 = 0
    var_49 = 'email'
    var_50 = [var_47, var_48, var_49]
    var_51 = module_0.Message(text=var_45, code=var_46, index=var_50)
    var_52 = [var_47, var_48, var_49]
    var_53 = module_0.Message(text=var_45, code=var_46, index=var_52)
    var_54 = [var_51]
    var_55 = module_0.ValidationError(messages=var_54)
    var_56 = [var_53]
    var_57 = module_0.ValidationError(messages=var_56)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_15, var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_11]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Parse error'
    var_25 = 'parse_code'
    var_26 = module_0.ParseError(text=var_24, code=var_25)
    var_27 = module_0.ParseError(text=var_24, code=var_25)
    var_28 = 'Error'
    var_29 = 'code'
    var_30 = module_0.ValidationError(text=var_28, code=var_29)
    var_31 = module_0.ParseError(text=var_28, code=var_29)
    var_32 = 'users'
    var_33 = 0
    var_34 = 'name'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.Message(text=var_28, code=var_29, index=var_35)
    var_37 = [var_32, var_33, var_34]
    var_38 = module_0.Message(text=var_28, code=var_29, index=var_37)
    var_39 = [var_36]
    var_40 = module_0.ValidationError(messages=var_39)
    var_41 = [var_38]
    var_42 = module_0.ValidationError(messages=var_41)
    var_43 = 1
    var_44 = 5
    var_45 = 10
    var_46 = module_0.Position(var_43, var_44, var_45)
    var_47 = module_0.Message(text=var_28, code=var_29, position=var_46)
    var_48 = module_0.Message(text=var_28, code=var_29, position=var_46)
    var_49 = [var_47]
    var_50 = module_0.ValidationError(messages=var_49)
    var_51 = [var_48]
    var_52 = module_0.ValidationError(messages=var_51)



# Parsed testcases at query #15
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'key1'
    var_9 = module_0.ValidationError(text=var_0, code=var_1, key=var_8)
    var_10 = 'key2'
    var_11 = module_0.ValidationError(text=var_0, code=var_1, key=var_10)
    var_12 = 'Error 1'
    var_13 = 'code1'
    var_14 = 'field1'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = 'Error 2'
    var_17 = 'code2'
    var_18 = 'field2'
    var_19 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_20 = [var_15, var_19]
    var_21 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_22 = module_0.Message(text=var_16, code=var_17, key=var_18)
    var_23 = [var_21, var_22]
    var_24 = module_0.ValidationError(messages=var_20)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_27 = [var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = module_0.Position(var_33, var_34, var_35)
    var_38 = 'Error'
    var_39 = 'code'
    var_40 = module_0.ValidationError(text=var_38, code=var_39, position=var_36)
    var_41 = module_0.ValidationError(text=var_38, code=var_39, position=var_37)
    var_42 = 2
    var_43 = 10
    var_44 = 9
    var_45 = module_0.Position(var_42, var_43, var_44)
    var_46 = module_0.ValidationError(text=var_38, code=var_39, position=var_45)



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = 'Error 3'
    var_21 = 'code3'
    var_22 = 'field3'
    var_23 = module_0.Message(text=var_20, code=var_21, key=var_22)
    var_24 = [var_11, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = [var_11]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = 'Parse error'
    var_29 = 'parse_code'
    var_30 = module_0.ParseError(text=var_28, code=var_29)
    var_31 = module_0.ParseError(text=var_28, code=var_29)
    var_32 = 1
    var_33 = 5
    var_34 = 4
    var_35 = module_0.Position(var_32, var_33, var_34)
    var_36 = 'Error'
    var_37 = 'code'
    var_38 = module_0.ValidationError(text=var_36, code=var_37, position=var_35)
    var_39 = module_0.ValidationError(text=var_36, code=var_37, position=var_35)
    var_40 = 2
    var_41 = 10
    var_42 = 9
    var_43 = module_0.Position(var_40, var_41, var_42)
    var_44 = module_0.ValidationError(text=var_36, code=var_37, position=var_43)
    var_45 = 'users'
    var_46 = 0
    var_47 = 'name'
    var_48 = [var_45, var_46, var_47]
    var_49 = module_0.Message(text=var_36, code=var_37, index=var_48)
    var_50 = [var_45, var_46, var_47]
    var_51 = module_0.Message(text=var_36, code=var_37, index=var_50)
    var_52 = [var_49]
    var_53 = module_0.ValidationError(messages=var_52)
    var_54 = [var_51]
    var_55 = module_0.ValidationError(messages=var_54)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Different Error'
    var_27 = module_0.Message(text=var_26, code=var_13, key=var_14)
    var_28 = [var_25, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = module_0.ParseError(text=var_0, code=var_1)
    var_31 = 1
    var_32 = 5
    var_33 = 4
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = 'Error'
    var_36 = module_0.ValidationError(text=var_35, code=var_9, position=var_34)
    var_37 = module_0.ValidationError(text=var_35, code=var_9, position=var_34)
    var_38 = 2
    var_39 = 10
    var_40 = 9
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.ValidationError(text=var_35, code=var_9, position=var_41)



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = 'Parse error'
    var_26 = 'parse_code'
    var_27 = module_0.ParseError(text=var_25, code=var_26)
    var_28 = module_0.ParseError(text=var_25, code=var_26)
    var_29 = module_0.ValidationError(text=var_25, code=var_26)
    var_30 = module_0.ParseError(text=var_25, code=var_26)
    var_31 = 1
    var_32 = 5
    var_33 = 4
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = 'Error'
    var_36 = 'code'
    var_37 = module_0.ValidationError(text=var_35, code=var_36, position=var_34)
    var_38 = module_0.ValidationError(text=var_35, code=var_36, position=var_34)
    var_39 = 2
    var_40 = 10
    var_41 = 9
    var_42 = module_0.Position(var_39, var_40, var_41)
    var_43 = module_0.ValidationError(text=var_35, code=var_36, position=var_42)
    var_44 = 'username'
    var_45 = module_0.ValidationError(text=var_35, code=var_36, key=var_44)
    var_46 = module_0.ValidationError(text=var_35, code=var_36, key=var_44)
    var_47 = 'email'
    var_48 = module_0.ValidationError(text=var_35, code=var_36, key=var_47)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 'Error'
    var_34 = 'code'
    var_35 = module_0.ValidationError(text=var_33, code=var_34)
    var_36 = module_0.ParseError(text=var_33, code=var_34)
    var_37 = 1
    var_38 = 5
    var_39 = 4
    var_40 = module_0.Position(var_37, var_38, var_39)
    var_41 = module_0.ValidationError(text=var_33, code=var_34, position=var_40)
    var_42 = module_0.ValidationError(text=var_33, code=var_34, position=var_40)
    var_43 = 2
    var_44 = 10
    var_45 = 9
    var_46 = module_0.Position(var_43, var_44, var_45)
    var_47 = module_0.ValidationError(text=var_33, code=var_34, position=var_46)



# Parsed testcases at query #20
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = 'Error 3'
    var_21 = 'code3'
    var_22 = 'field3'
    var_23 = module_0.Message(text=var_20, code=var_21, key=var_22)
    var_24 = [var_11, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    var_26 = [var_11]
    var_27 = module_0.ValidationError(messages=var_26)
    var_28 = module_0.ParseError(text=var_0, code=var_1)
    var_29 = 1
    var_30 = 5
    var_31 = 4
    var_32 = module_0.Position(var_29, var_30, var_31)
    var_33 = module_0.Position(var_29, var_30, var_31)
    var_34 = 'Error'
    var_35 = 'code'
    var_36 = module_0.Message(text=var_34, code=var_35, position=var_32)
    var_37 = module_0.Message(text=var_34, code=var_35, position=var_33)
    var_38 = [var_36]
    var_39 = module_0.ValidationError(messages=var_38)
    var_40 = [var_37]
    var_41 = module_0.ValidationError(messages=var_40)
    var_42 = 2
    var_43 = 10
    var_44 = module_0.Position(var_42, var_30, var_43)
    var_45 = module_0.Message(text=var_34, code=var_35, position=var_44)
    var_46 = [var_45]
    var_47 = module_0.ValidationError(messages=var_46)



# Parsed testcases at query #21
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_15, var_11]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Parse error'
    var_25 = 'parse_code'
    var_26 = module_0.ParseError(text=var_24, code=var_25)
    var_27 = module_0.ParseError(text=var_24, code=var_25)
    var_28 = 1
    var_29 = 5
    var_30 = 4
    var_31 = module_0.Position(var_28, var_29, var_30)
    var_32 = 'Error'
    var_33 = 'pos_code'
    var_34 = module_0.Message(text=var_32, code=var_33, position=var_31)
    var_35 = module_0.Message(text=var_32, code=var_33, position=var_31)
    var_36 = [var_34]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = [var_35]
    var_39 = module_0.ValidationError(messages=var_38)
    var_40 = 2
    var_41 = 10
    var_42 = 9
    var_43 = module_0.Position(var_40, var_41, var_42)
    var_44 = module_0.Message(text=var_32, code=var_33, position=var_43)
    var_45 = [var_44]
    var_46 = module_0.ValidationError(messages=var_45)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = 1
    var_26 = 5
    var_27 = 4
    var_28 = module_0.Position(var_25, var_26, var_27)
    var_29 = 'Error'
    var_30 = 'pos_error'
    var_31 = module_0.ValidationError(text=var_29, code=var_30, position=var_28)
    var_32 = module_0.ValidationError(text=var_29, code=var_30, position=var_28)
    var_33 = 2
    var_34 = 10
    var_35 = 9
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = module_0.ValidationError(text=var_29, code=var_30, position=var_36)
    var_38 = 'Parse error'
    var_39 = 'parse_code'
    var_40 = module_0.ParseError(text=var_38, code=var_39)
    var_41 = module_0.ParseError(text=var_38, code=var_39)



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = [var_10]
    var_20 = module_0.Message(text=var_8, code=var_9, index=var_19)
    var_21 = [var_15]
    var_22 = module_0.Message(text=var_13, code=var_14, index=var_21)
    var_23 = [var_20, var_22]
    var_24 = module_0.ValidationError(messages=var_18)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = [var_10]
    var_27 = module_0.Message(text=var_8, code=var_9, index=var_26)
    var_28 = [var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = [var_10]
    var_31 = module_0.Message(text=var_8, code=var_9, index=var_30)
    var_32 = 'Different'
    var_33 = [var_15]
    var_34 = module_0.Message(text=var_32, code=var_14, index=var_33)
    var_35 = [var_31, var_34]
    var_36 = module_0.ValidationError(messages=var_35)
    var_37 = 'Parse error'
    var_38 = 'parse_code'
    var_39 = module_0.ParseError(text=var_37, code=var_38)
    var_40 = module_0.ParseError(text=var_37, code=var_38)
    var_41 = 1
    var_42 = 5
    var_43 = 10
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = 'Error'
    var_46 = 'code'
    var_47 = module_0.ValidationError(text=var_45, code=var_46, position=var_44)
    var_48 = module_0.ValidationError(text=var_45, code=var_46, position=var_44)
    var_49 = 2
    var_50 = 3
    var_51 = 15
    var_52 = module_0.Position(var_49, var_50, var_51)
    var_53 = module_0.ValidationError(text=var_45, code=var_46, position=var_52)



# Parsed testcases at query #24
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Error 1'
    var_5 = 'code1'
    var_6 = module_0.Message(text=var_4, code=var_5)
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = module_0.Message(text=var_7, code=var_8)
    var_10 = [var_6, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    var_12 = module_0.ValidationError(messages=var_10)
    var_13 = module_0.ValidationError(text=var_0, code=var_1)
    var_14 = 'Different message'
    var_15 = module_0.ValidationError(text=var_14, code=var_1)
    var_16 = module_0.ValidationError(text=var_0, code=var_5)
    var_17 = module_0.ValidationError(text=var_0, code=var_8)
    var_18 = module_0.Message(text=var_4, code=var_5)
    var_19 = [var_18]
    var_20 = module_0.ValidationError(messages=var_19)
    var_21 = module_0.Message(text=var_4, code=var_5)
    var_22 = module_0.Message(text=var_7, code=var_8)
    var_23 = [var_21, var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.ValidationError(text=var_0, code=var_1)
    var_26 = module_0.ParseError(text=var_0, code=var_1)
    var_27 = module_0.ValidationError(text=var_0, code=var_1)
    var_28 = 'Error'
    var_29 = 'code'
    var_30 = 'field1'
    var_31 = module_0.Message(text=var_28, code=var_29, key=var_30)
    var_32 = [var_31]
    var_33 = module_0.ValidationError(messages=var_32)
    var_34 = 'field2'
    var_35 = module_0.Message(text=var_28, code=var_29, key=var_34)
    var_36 = [var_35]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = 1
    var_39 = 5
    var_40 = 4
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.Position(var_38, var_39, var_40)
    var_43 = module_0.Message(text=var_28, code=var_29, position=var_41)
    var_44 = [var_43]
    var_45 = module_0.ValidationError(messages=var_44)
    var_46 = module_0.Message(text=var_28, code=var_29, position=var_42)
    var_47 = [var_46]
    var_48 = module_0.ValidationError(messages=var_47)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Different Error'
    var_27 = module_0.Message(text=var_26, code=var_13, key=var_14)
    var_28 = [var_25, var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = module_0.ParseError(text=var_0, code=var_1)
    var_31 = 1
    var_32 = 5
    var_33 = 4
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = 'Error'
    var_36 = 'code'
    var_37 = module_0.ValidationError(text=var_35, code=var_36, position=var_34)
    var_38 = module_0.ValidationError(text=var_35, code=var_36, position=var_34)
    var_39 = 2
    var_40 = 10
    var_41 = 9
    var_42 = module_0.Position(var_39, var_40, var_41)
    var_43 = module_0.ValidationError(text=var_35, code=var_36, position=var_42)



# Parsed testcases at query #26
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_26 = 'Error 3'
    var_27 = 'code3'
    var_28 = 'field3'
    var_29 = module_0.Message(text=var_26, code=var_27, key=var_28)
    var_30 = [var_25, var_29]
    var_31 = module_0.ValidationError(messages=var_30)
    var_32 = 'Parse error'
    var_33 = 'parse_code'
    var_34 = module_0.ParseError(text=var_32, code=var_33)
    var_35 = module_0.ParseError(text=var_32, code=var_33)
    var_36 = 'Error'
    var_37 = 'code'
    var_38 = module_0.ValidationError(text=var_36, code=var_37)
    var_39 = module_0.ParseError(text=var_36, code=var_37)
    var_40 = 1
    var_41 = 5
    var_42 = 4
    var_43 = module_0.Position(var_40, var_41, var_42)
    var_44 = module_0.ValidationError(text=var_36, code=var_37, position=var_43)
    var_45 = module_0.ValidationError(text=var_36, code=var_37, position=var_43)
    var_46 = 2
    var_47 = 10
    var_48 = 9
    var_49 = module_0.Position(var_46, var_47, var_48)
    var_50 = module_0.ValidationError(text=var_36, code=var_37, position=var_49)



# Parsed testcases at query #27
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_15, var_11]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_25 = 'Different'
    var_26 = module_0.Message(text=var_25, code=var_13, key=var_14)
    var_27 = [var_24, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = module_0.ParseError(text=var_0, code=var_1)
    var_30 = 1
    var_31 = 5
    var_32 = 4
    var_33 = module_0.Position(var_30, var_31, var_32)
    var_34 = 'Error'
    var_35 = 'code'
    var_36 = module_0.ValidationError(text=var_34, code=var_35, position=var_33)
    var_37 = module_0.ValidationError(text=var_34, code=var_35, position=var_33)
    var_38 = 2
    var_39 = 10
    var_40 = 9
    var_41 = module_0.Position(var_38, var_39, var_40)
    var_42 = module_0.ValidationError(text=var_34, code=var_35, position=var_41)
    var_43 = 0
    var_44 = module_0.Position(var_30, var_30, var_43)
    var_45 = module_0.Position(var_30, var_39, var_40)
    var_46 = 'field'
    var_47 = module_0.ValidationError(text=var_34, code=var_35, key=var_46)
    var_48 = module_0.ValidationError(text=var_34, code=var_35, key=var_46)



# Parsed testcases at query #28
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = module_0.ParseError(text=var_0, code=var_1)
    var_30 = 1
    var_31 = 5
    var_32 = 4
    var_33 = module_0.Position(var_30, var_31, var_32)
    var_34 = module_0.Position(var_30, var_31, var_32)
    var_35 = 'Error'
    var_36 = 'test'
    var_37 = module_0.ValidationError(text=var_35, code=var_36, position=var_33)
    var_38 = module_0.ValidationError(text=var_35, code=var_36, position=var_34)
    var_39 = 2
    var_40 = 10
    var_41 = 20
    var_42 = module_0.Position(var_39, var_40, var_41)
    var_43 = module_0.ValidationError(text=var_35, code=var_36, position=var_42)



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = 'Different Error'
    var_24 = module_0.Message(text=var_23, code=var_13, key=var_14)
    var_25 = [var_22, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_28 = [var_27]
    var_29 = module_0.ValidationError(messages=var_28)
    var_30 = module_0.ParseError(text=var_0, code=var_1)
    var_31 = 1
    var_32 = 5
    var_33 = 10
    var_34 = module_0.Position(var_31, var_32, var_33)
    var_35 = module_0.Position(var_31, var_32, var_33)
    var_36 = 'Error'
    var_37 = module_0.ValidationError(text=var_36, code=var_9, position=var_34)
    var_38 = module_0.ValidationError(text=var_36, code=var_9, position=var_35)
    var_39 = 2
    var_40 = 20
    var_41 = module_0.Position(var_39, var_33, var_40)
    var_42 = module_0.ValidationError(text=var_36, code=var_9, position=var_41)



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different error'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = [var_11, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_11]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_15, var_11]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Parse error'
    var_25 = 'parse_code'
    var_26 = module_0.ParseError(text=var_24, code=var_25)
    var_27 = module_0.ParseError(text=var_24, code=var_25)
    var_28 = 1
    var_29 = 5
    var_30 = 4
    var_31 = module_0.Position(var_28, var_29, var_30)
    var_32 = 'Error with position'
    var_33 = 'pos_code'
    var_34 = module_0.Message(text=var_32, code=var_33, position=var_31)
    var_35 = module_0.Message(text=var_32, code=var_33, position=var_31)
    var_36 = [var_34]
    var_37 = module_0.ValidationError(messages=var_36)
    var_38 = [var_35]
    var_39 = module_0.ValidationError(messages=var_38)
    var_40 = 2
    var_41 = 10
    var_42 = 9
    var_43 = module_0.Position(var_40, var_41, var_42)
    var_44 = module_0.Message(text=var_32, code=var_33, position=var_43)
    var_45 = [var_44]
    var_46 = module_0.ValidationError(messages=var_45)



# Parsed testcases at query #31
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
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
    var_13 = module_0.Message(text=var_4, code=var_5, key=var_6)
    var_14 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_15 = [var_13, var_14]
    var_16 = module_0.ValidationError(messages=var_12)
    var_17 = module_0.ValidationError(messages=var_15)
    var_18 = 'Error message 1'
    var_19 = module_0.ValidationError(text=var_18, code=var_1)
    var_20 = 'Error message 2'
    var_21 = module_0.ValidationError(text=var_20, code=var_1)
    var_22 = module_0.ValidationError(text=var_0, code=var_5)
    var_23 = module_0.ValidationError(text=var_0, code=var_9)
    var_24 = module_0.Message(text=var_4, code=var_5)
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_4, code=var_5)
    var_27 = module_0.Message(text=var_8, code=var_9)
    var_28 = [var_26, var_27]
    var_29 = module_0.ValidationError(messages=var_25)
    var_30 = module_0.ValidationError(messages=var_28)
    var_31 = module_0.ValidationError(text=var_0, code=var_1)
    var_32 = module_0.ValidationError(text=var_0, code=var_1)
    var_33 = module_0.ParseError(text=var_0, code=var_1)
    var_34 = 1
    var_35 = 5
    var_36 = 10
    var_37 = module_0.Position(var_34, var_35, var_36)
    var_38 = module_0.ValidationError(text=var_0, code=var_1, position=var_37)
    var_39 = module_0.ValidationError(text=var_0, code=var_1, position=var_37)
    var_40 = module_0.Position(var_34, var_35, var_36)
    var_41 = 2
    var_42 = 20
    var_43 = module_0.Position(var_41, var_36, var_42)
    var_44 = module_0.ValidationError(text=var_0, code=var_1, position=var_40)
    var_45 = module_0.ValidationError(text=var_0, code=var_1, position=var_43)



# Parsed testcases at query #32
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_18 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_19 = [var_17, var_18]
    var_20 = module_0.ValidationError(messages=var_16)
    var_21 = module_0.ValidationError(messages=var_19)
    var_22 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_26 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_27 = [var_25, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = 'Parse error'
    var_30 = 'parse_code'
    var_31 = module_0.ParseError(text=var_29, code=var_30)
    var_32 = module_0.ParseError(text=var_29, code=var_30)
    var_33 = 1
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_33, var_34, var_35)
    var_37 = 'Error with position'
    var_38 = 'pos_code'
    var_39 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_40 = module_0.ValidationError(text=var_37, code=var_38, position=var_36)
    var_41 = 2
    var_42 = 10
    var_43 = 9
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.ValidationError(text=var_37, code=var_38, position=var_44)



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    var_18 = module_0.ValidationError(messages=var_16)
    var_19 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_20 = 'Different Error'
    var_21 = module_0.Message(text=var_20, code=var_13, key=var_14)
    var_22 = [var_19, var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_25 = [var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = 'Parse error'
    var_28 = 'parse_code'
    var_29 = module_0.ParseError(text=var_27, code=var_28)
    var_30 = module_0.ParseError(text=var_27, code=var_28)
    var_31 = 'Error'
    var_32 = module_0.ValidationError(text=var_31, code=var_9)
    var_33 = module_0.ParseError(text=var_31, code=var_9)
    var_34 = 1
    var_35 = 5
    var_36 = 4
    var_37 = module_0.Position(var_34, var_35, var_36)
    var_38 = 'code'
    var_39 = module_0.ValidationError(text=var_31, code=var_38, position=var_37)
    var_40 = module_0.ValidationError(text=var_31, code=var_38, position=var_37)
    var_41 = 2
    var_42 = 10
    var_43 = 9
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.ValidationError(text=var_31, code=var_38, position=var_44)



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_8, code=var_9, index=var_11)
    var_13 = 'Error 2'
    var_14 = 'code2'
    var_15 = 'field2'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = [var_12, var_17]
    var_19 = [var_10]
    var_20 = module_0.Message(text=var_8, code=var_9, index=var_19)
    var_21 = [var_15]
    var_22 = module_0.Message(text=var_13, code=var_14, index=var_21)
    var_23 = [var_20, var_22]
    var_24 = module_0.ValidationError(messages=var_18)
    var_25 = module_0.ValidationError(messages=var_23)
    var_26 = [var_10]
    var_27 = module_0.Message(text=var_8, code=var_9, index=var_26)
    var_28 = 'Different Error'
    var_29 = [var_15]
    var_30 = module_0.Message(text=var_28, code=var_14, index=var_29)
    var_31 = [var_27, var_30]
    var_32 = module_0.ValidationError(messages=var_31)
    var_33 = [var_10]
    var_34 = module_0.Message(text=var_8, code=var_9, index=var_33)
    var_35 = [var_34]
    var_36 = module_0.ValidationError(messages=var_35)
    var_37 = 'Parse error'
    var_38 = 'parse_code'
    var_39 = module_0.ParseError(text=var_37, code=var_38)
    var_40 = module_0.ParseError(text=var_37, code=var_38)
    var_41 = 1
    var_42 = 5
    var_43 = 4
    var_44 = module_0.Position(var_41, var_42, var_43)
    var_45 = module_0.Position(var_41, var_42, var_43)
    var_46 = 'Error'
    var_47 = 'code'
    var_48 = module_0.ValidationError(text=var_46, code=var_47, position=var_44)
    var_49 = module_0.ValidationError(text=var_46, code=var_47, position=var_45)
    var_50 = 2
    var_51 = 10
    var_52 = 15
    var_53 = module_0.Position(var_50, var_51, var_52)
    var_54 = module_0.ValidationError(text=var_46, code=var_47, position=var_53)



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'test_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = 'Different Error'
    var_5 = module_0.Message(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.Message(text=var_0, code=var_6)
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, index=var_9)
    var_11 = [var_8]
    var_12 = module_0.Message(text=var_0, code=var_1, index=var_11)
    var_13 = 1
    var_14 = 5
    var_15 = 4
    var_16 = module_0.Position(var_13, var_14, var_15)
    var_17 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_18 = module_0.Message(text=var_0, code=var_1, position=var_16)
    var_19 = 2
    var_20 = 10
    var_21 = 20
    var_22 = module_0.Position(var_19, var_20, var_21)
    var_23 = module_0.Message(text=var_0, code=var_1, position=var_22)
    var_24 = 0
    var_25 = module_0.Position(var_13, var_13, var_24)
    var_26 = 9
    var_27 = module_0.Position(var_13, var_20, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_27)
    var_29 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_27)
    var_30 = 15
    var_31 = module_0.Position(var_19, var_14, var_30)
    var_32 = module_0.Message(text=var_0, code=var_1, start_position=var_25, end_position=var_31)
    var_33 = 'username'
    var_34 = module_0.Message(text=var_0, code=var_1, key=var_33)
    var_35 = [var_33]
    var_36 = module_0.Message(text=var_0, code=var_1, index=var_35)
    var_37 = []
    var_38 = module_0.Message(text=var_0, code=var_1, index=var_37)
    var_39 = module_0.Message(text=var_0, code=var_1)
    var_40 = 'users'
    var_41 = 'email'
    var_42 = [var_40, var_24, var_41]
    var_43 = module_0.Message(text=var_0, code=var_1, index=var_42)
    var_44 = [var_40, var_24, var_41]
    var_45 = module_0.Message(text=var_0, code=var_1, index=var_44)
    var_46 = [var_40, var_13, var_41]
    var_47 = module_0.Message(text=var_0, code=var_1, index=var_46)



