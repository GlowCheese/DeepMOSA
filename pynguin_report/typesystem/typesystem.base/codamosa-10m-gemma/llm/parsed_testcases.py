####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'Invalid input'
    var_7 = 'error_code'
    var_8 = module_0.ValidationError(text=var_6, code=var_7)
    var_9 = module_0.ValidationResult(error=var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = module_0.ValidationResult()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'success_data'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'error_text'
    var_5 = 'error_code'
    var_6 = module_0.ValidationError(text=var_4, code=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = module_0.ValidationResult()
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #3
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
    var_6 = 'error message'
    var_7 = 'err_code'
    var_8 = module_0.ValidationError(text=var_6, code=var_7)
    var_9 = 123
    var_10 = module_0.ValidationResult(value=var_9)
    var_11 = iter(var_10)
    var_12 = next(var_11)
    assert var_12 == 123
    var_13 = next(var_11)
    assert var_13 is None
    var_14 = next(var_11)



# Parsed testcases at query #4
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
    var_10 = module_0.ValidationError(text=var_8, code=var_9)
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
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Invalid syntax'
    var_5 = 'syntax_error'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_3)
    var_7 = module_0.ParseError(text=var_4, code=var_5, position=var_3)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = error_single.messages()[var_9]
    var_11 = var_10.text
    assert var_11 == 'Invalid syntax'
    var_12 = error_single.messages()[var_9]
    var_13 = var_12.start_position
    var_14 = error_single.messages()[var_9]
    var_15 = var_14.end_position
    var_16 = 'Error 1'
    var_17 = 'err1'
    var_18 = 'users'
    var_19 = [var_18, var_9]
    var_20 = module_0.Message(text=var_16, code=var_17, index=var_19)
    var_21 = 'Error 2'
    var_22 = 'err2'
    var_23 = [var_18, var_0]
    var_24 = module_0.Message(text=var_21, code=var_22, index=var_23)
    var_25 = [var_20, var_24]
    var_26 = module_0.ParseError(messages=var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = 'Bad field'
    var_29 = 'username'
    var_30 = module_0.ParseError(text=var_28, key=var_29)
    var_31 = module_0.Position(var_0, var_9, var_9)
    var_32 = module_0.Position(var_0, var_1, var_1)
    var_33 = 'Range error'
    var_34 = module_0.Message(text=var_33, start_position=var_31, end_position=var_32)
    var_35 = [var_34]
    var_36 = module_0.ParseError(messages=var_35)
    var_37 = error_range.messages()[var_9]
    var_38 = var_37.start_position
    var_39 = error_range.messages()[var_9]
    var_40 = var_39.end_position
    var_41 = 'Error'
    var_42 = [var_20]
    var_43 = module_0.ParseError(text=var_41, messages=var_42)
    var_44 = 'Error'
    var_45 = 'user'
    var_46 = 'list'
    var_47 = 0
    var_48 = [var_46, var_47]
    var_49 = module_0.ParseError(text=var_44, key=var_45)
    var_50 = 'Err'
    var_51 = module_0.Message(text=var_50, position=var_3, start_position=var_31)
    var_52 = 'Error'
    var_53 = [var_20]
    var_54 = module_0.ParseError(text=var_52, messages=var_53)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "Message(text='Error', code='err_code')"
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "Message(text='Error', code='custom', index=['username'])"
    var_7 = 'users'
    var_8 = 0
    var_9 = 'name'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.Message(text=var_0, index=var_10)
    var_12 = repr(var_11)
    assert var_12 == "Message(text='Error', code='custom', index=['users', 0, 'name'])"
    var_13 = 1
    var_14 = 5
    var_15 = 10
    var_16 = module_0.Position(var_13, var_14, var_15)
    var_17 = module_0.Message(text=var_0, position=var_16)
    var_18 = repr(var_2)
    var_19 = repr(var_17)
    var_20 = repr(var_17)
    var_21 = repr(var_17)
    var_22 = module_0.Position(var_13, var_13, var_13)
    var_23 = module_0.Position(var_13, var_14, var_14)
    var_24 = module_0.Message(text=var_0, start_position=var_22, end_position=var_23)
    var_25 = repr(var_24)
    var_26 = 'Simple'
    var_27 = module_0.Message(text=var_26)
    var_28 = repr(var_27)
    assert var_28 == "Message(text='Simple', code='custom')"



# Parsed testcases at query #7
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    var_2 = repr(var_1)
    assert var_2 == "Message(text='error', code='custom')"
    var_3 = 'max_length'
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, code=var_3, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "Message(text='error', code='max_length', index=['username'])"
    var_7 = 'users'
    var_8 = 0
    var_9 = 'name'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.Message(text=var_0, index=var_10)
    var_12 = repr(var_11)
    assert var_12 == "Message(text='error', code='custom', index=['users', 0, 'name'])"
    var_13 = 1
    var_14 = 5
    var_15 = 10
    var_16 = module_0.Position(var_13, var_14, var_15)
    var_17 = module_0.Message(text=var_0, position=var_16)
    var_18 = repr(var_17)
    var_19 = repr(var_17)
    var_20 = repr(var_17)
    var_21 = module_0.Position(var_13, var_8, var_8)
    var_22 = module_0.Position(var_13, var_14, var_14)
    var_23 = module_0.Message(text=var_0, start_position=var_21, end_position=var_22)
    var_24 = "Message(text='error', code='custom', start_position=Position(line_no=1, column_no=0, char_index=0), end_position=Position(line_no=1, column_no=5, char_index=5))"
    var_25 = repr(var_23)
    var_26 = 'no position'
    var_27 = 'none'
    var_28 = module_0.Message(text=var_26, code=var_27)
    var_29 = repr(var_28)
    var_30 = repr(var_28)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    var_4 = repr(var_3)
    assert var_4 == "ValidationError(text='Error 1', code='code1')"
    var_5 = 'Error 2'
    var_6 = 'user'
    var_7 = module_0.Message(text=var_5, key=var_6)
    var_8 = module_0.ValidationError(text=var_5, key=var_6)
    var_9 = "Message(text='Error 2', code='custom', index=['user'])"
    var_10 = repr(var_8)
    var_11 = 'Error 3'
    var_12 = 'code3'
    var_13 = 'items'
    var_14 = 0
    var_15 = [var_13, var_14]
    var_16 = module_0.Message(text=var_11, code=var_12, index=var_15)
    var_17 = 'Error 4'
    var_18 = 'code4'
    var_19 = module_0.Message(text=var_17, code=var_18)
    var_20 = [var_16, var_19]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = "Message(text='Error 3', code='code3', index=['items', 0])"
    var_23 = "Message(text='Error 4', code='code4')"
    var_24 = repr(var_21)
    var_25 = 1
    var_26 = module_0.Position(var_25, var_25, var_25)
    var_27 = 'Pos Error'
    var_28 = module_0.Message(text=var_27, position=var_26)
    var_29 = module_0.ValidationError(text=var_27, position=var_26)
    var_30 = "Message(text='Pos Error', code='custom', position=Position(line_no=1, column_no=1, char_index=1))"
    var_31 = repr(var_29)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid format'
    var_1 = 'format_error'
    var_2 = 'field_name'
    var_3 = module_0.ParseError(text=var_0, code=var_1, key=var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = error.messages()[var_5]
    var_7 = var_6.text
    var_8 = error.messages()[var_5]
    var_9 = var_8.code
    var_10 = error.messages()[var_5]
    var_11 = var_10.index
    var_12 = 1
    var_13 = 5
    var_14 = 10
    var_15 = module_0.Position(var_12, var_13, var_14)
    var_16 = 'Error at pos'
    var_17 = module_0.ParseError(text=var_16, position=var_15)
    var_18 = error_with_pos.messages()[var_5]
    var_19 = var_18.start_position
    var_20 = error_with_pos.messages()[var_5]
    var_21 = var_20.end_position
    var_22 = 'Error 1'
    var_23 = 'code1'
    var_24 = 'users'
    var_25 = [var_24, var_5]
    var_26 = module_0.Message(text=var_22, code=var_23, index=var_25)
    var_27 = 'Error 2'
    var_28 = 'code2'
    var_29 = 'name'
    var_30 = [var_24, var_5, var_29]
    var_31 = module_0.Message(text=var_27, code=var_28, index=var_30)
    var_32 = [var_26, var_31]
    var_33 = module_0.ParseError(messages=var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'text'
    var_36 = [var_26]
    var_37 = module_0.ParseError(text=var_35, messages=var_36)
    var_38 = 'text'
    var_39 = 'code'
    var_40 = [var_26]
    var_41 = module_0.ParseError(text=var_38, code=var_39, messages=var_40)
    var_42 = str(var_3)
    var_43 = repr(var_3)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 6
    var_6 = module_0.Position(var_0, var_5, var_2)
    var_7 = 2
    var_8 = module_0.Position(var_7, var_1, var_2)
    var_9 = 11
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 'not a position'



# Parsed testcases at query #12
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
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, start_position=var_3)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, start_position=var_8)
    var_31 = [var_12, var_13]
    var_32 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_31, end_position=var_3)
    var_33 = [var_12, var_13]
    var_34 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_33, end_position=var_8)
    var_35 = [var_12, var_13]
    var_36 = module_0.Message(text=var_9, key=var_11, index=var_35, position=var_3)



# Parsed testcases at query #13
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
    var_11 = 'Error 1'
    var_12 = 'users'
    var_13 = 0
    var_14 = 'name'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.Message(text=var_11, index=var_15)
    var_17 = 'Error 2'
    var_18 = 1
    var_19 = [var_12, var_18, var_14]
    var_20 = module_0.Message(text=var_17, index=var_19)
    var_21 = [var_16, var_20]
    var_22 = module_0.ValidationError(messages=var_21)
    var_23 = {var_14: var_11}
    var_24 = {var_14: var_17}
    var_25 = {var_13: var_23, var_18: var_24}
    var_26 = {var_12: var_25}
    var_27 = str(var_26)
    var_28 = str(var_22)
    var_29 = 'Invalid input'
    var_30 = 'age'
    var_31 = module_0.Message(text=var_29, key=var_30)
    var_32 = [var_31]
    var_33 = module_0.ValidationError(messages=var_32)
    var_34 = str(var_33)
    assert var_34 == "{'age': 'Invalid input'}"



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Simple error'
    var_5 = 'error_code'
    var_6 = module_0.Message(text=var_4, code=var_5)
    var_7 = module_0.ValidationError(text=var_4, code=var_5)
    var_8 = str(var_7)
    assert var_8 == 'Simple error'
    var_9 = 'Field error'
    var_10 = 'username'
    var_11 = module_0.Message(text=var_9, key=var_10)
    var_12 = [var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = str(var_13)
    assert var_14 == "{'username': 'Field error'}"
    var_15 = 'Invalid name'
    var_16 = 'users'
    var_17 = 0
    var_18 = 'name'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.Message(text=var_15, index=var_19)
    var_21 = 'Invalid email'
    var_22 = 'email'
    var_23 = [var_22]
    var_24 = module_0.Message(text=var_21, index=var_23)
    var_25 = [var_20, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = {var_18: var_15}
    var_28 = {var_17: var_27}
    var_29 = {var_16: var_28, var_22: var_21}
    var_30 = str(var_29)
    var_31 = str(var_26)
    var_32 = 'Position error'
    var_33 = module_0.Message(text=var_32, position=var_3)
    var_34 = module_0.ValidationError(text=var_32, position=var_3)
    var_35 = str(var_34)
    assert var_35 == 'Position error'
    var_36 = 'Range error'
    var_37 = module_0.Message(text=var_36, start_position=var_3, end_position=var_3)
    var_38 = module_0.ValidationError(text=var_36)
    var_39 = str(var_38)
    assert var_39 == 'Range error'



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = 3
    var_7 = 15
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



# Parsed testcases at query #17
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
    var_12 = 'profile'
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
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, end_position=var_8)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
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
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, start_position=var_3, end_position=var_8)



# Parsed testcases at query #20
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
    var_10 = 'field'
    var_11 = 'sub'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_12, position=var_3)
    var_14 = [var_11]
    var_15 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_14, position=var_4)
    var_16 = 'different'
    var_17 = [var_11]
    var_18 = module_0.Message(text=var_16, code=var_9, key=var_10, index=var_17, position=var_3)
    var_19 = [var_11]
    var_20 = module_0.Message(text=var_8, code=var_16, key=var_10, index=var_19, position=var_3)
    var_21 = 'other'
    var_22 = [var_21]
    var_23 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_22, position=var_3)
    var_24 = [var_11]
    var_25 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_24, start_position=var_3, end_position=var_7)
    var_26 = [var_11]
    var_27 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_26, start_position=var_7, end_position=var_3)
    var_28 = [var_11]
    var_29 = module_0.Message(text=var_8, code=var_9, key=var_10, index=var_28, position=var_7)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'error text'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'max_length'
    var_3 = 'username'
    var_4 = module_0.Message(text=var_0, code=var_2, key=var_3)
    var_5 = 'users'
    var_6 = 3
    var_7 = [var_5, var_6, var_3]
    var_8 = module_0.Message(text=var_0, index=var_7)
    var_9 = 1
    var_10 = 5
    var_11 = 10
    var_12 = module_0.Position(var_9, var_10, var_11)
    var_13 = module_0.Message(text=var_0, position=var_12)
    var_14 = 0
    var_15 = module_0.Position(var_9, var_14, var_14)
    var_16 = module_0.Position(var_9, var_10, var_10)
    var_17 = module_0.Message(text=var_0, start_position=var_15, end_position=var_16)
    var_18 = module_0.Message(text=var_0, code=var_2, key=var_3)
    var_19 = repr(var_4)
    assert var_19 == "Message(text='error text', code='max_length', index=['username'])"
    var_20 = hash(var_4)
    var_21 = hash(var_18)
    var_22 = 'err'
    var_23 = 'key'
    var_24 = 'index'
    var_25 = [var_24]
    var_26 = module_0.Message(text=var_22, key=var_23, index=var_25)
    var_27 = 'err'
    var_28 = module_0.Message(text=var_27, position=var_12, start_position=var_15)
    var_29 = 'err'
    var_30 = module_0.Message(text=var_29, position=var_12, end_position=var_16)
    var_31 = 'err'
    var_32 = module_0.Message(text=var_31, position=var_12, start_position=var_15)
    var_33 = 'err'
    var_34 = module_0.Message(text=var_33, position=var_12, end_position=var_16)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 5
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 11
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 6
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 'not a position'



# Parsed testcases at query #2
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Invalid syntax'
    var_5 = 'syntax_error'
    var_6 = module_0.Message(text=var_4, code=var_5, position=var_3)
    var_7 = module_0.ParseError(text=var_4, code=var_5, position=var_3)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = error_single.messages()[var_9]
    var_11 = var_10.text
    assert var_11 == 'Invalid syntax'
    var_12 = error_single.messages()[var_9]
    var_13 = var_12.start_position
    var_14 = error_single.messages()[var_9]
    var_15 = var_14.end_position
    var_16 = 'Error 2'
    var_17 = 'err2'
    var_18 = 'users'
    var_19 = 'name'
    var_20 = [var_18, var_9, var_19]
    var_21 = module_0.Message(text=var_16, code=var_17, index=var_20)
    var_22 = 'Error 3'
    var_23 = 'err3'
    var_24 = 'age'
    var_25 = [var_18, var_0, var_24]
    var_26 = module_0.Message(text=var_22, code=var_23, index=var_25)
    var_27 = [var_6, var_21, var_26]
    var_28 = module_0.ParseError(messages=var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'error'
    var_31 = [var_6]
    var_32 = module_0.ParseError(text=var_30, messages=var_31)
    var_33 = 'err'
    var_34 = 'user'
    var_35 = 1
    var_36 = 2
    var_37 = [var_35, var_36]
    var_38 = module_0.Message(text=var_33, key=var_34, index=var_37)
    var_39 = 'err'
    var_40 = module_0.Message(text=var_39, position=var_3, start_position=var_3)
    var_41 = 'a'
    var_42 = 1
    var_43 = {var_41: var_42}
    var_44 = module_0.ValidationResult(value=var_43, error=var_7)



# Parsed testcases at query #3
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'Invalid input'
    var_7 = 'error_code'
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



# Parsed testcases at query #4
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
    var_9 = 0
    var_10 = module_0.Position(var_9, var_9, var_9)
    var_11 = module_0.Position(var_9, var_1, var_1)
    var_12 = 'error'
    var_13 = 'err_code'
    var_14 = 'user'
    var_15 = 'users'
    var_16 = [var_15, var_9]
    var_17 = module_0.Message(text=var_12, code=var_13, key=var_14, index=var_16, position=var_3)
    var_18 = [var_15, var_9]
    var_19 = module_0.Message(text=var_12, code=var_13, key=var_14, index=var_18, position=var_4)
    var_20 = 'different'
    var_21 = [var_15, var_9]
    var_22 = module_0.Message(text=var_20, code=var_13, key=var_14, index=var_21, position=var_3)
    var_23 = 'other_code'
    var_24 = [var_15, var_9]
    var_25 = module_0.Message(text=var_12, code=var_23, key=var_14, index=var_24, position=var_3)
    var_26 = 'admin'
    var_27 = [var_15, var_9]
    var_28 = module_0.Message(text=var_12, code=var_13, key=var_26, index=var_27, position=var_3)
    var_29 = [var_15, var_0]
    var_30 = module_0.Message(text=var_12, code=var_13, index=var_29, position=var_3)
    var_31 = [var_15, var_9]
    var_32 = module_0.Message(text=var_12, code=var_13, key=var_14, index=var_31, start_position=var_10, end_position=var_11)
    var_33 = [var_15, var_9]
    var_34 = module_0.Message(text=var_12, code=var_13, key=var_14, index=var_33, start_position=var_3)
    var_35 = [var_15, var_9]
    var_36 = module_0.Message(text=var_12, code=var_13, key=var_14, index=var_35, position=var_3)
    var_37 = module_0.Position(var_0, var_1, var_2)



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
    var_7 = 'error'
    var_8 = 'err_code'
    var_9 = 'user'
    var_10 = 'users'
    var_11 = 0
    var_12 = [var_10, var_11]
    var_13 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_12, position=var_3)
    var_14 = [var_10, var_11]
    var_15 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_14, position=var_4)
    var_16 = 'different'
    var_17 = [var_10, var_11]
    var_18 = module_0.Message(text=var_16, code=var_8, key=var_9, index=var_17, position=var_3)
    var_19 = 'other_code'
    var_20 = [var_10, var_11]
    var_21 = module_0.Message(text=var_7, code=var_19, key=var_9, index=var_20, position=var_3)
    var_22 = [var_10, var_0]
    var_23 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_22, position=var_3)
    var_24 = [var_10, var_11]
    var_25 = module_0.Message(text=var_7, code=var_8, key=var_9, index=var_24, position=var_6)
    var_26 = module_0.Message(text=var_7)
    var_27 = 'custom'
    var_28 = []
    var_29 = module_0.Message(text=var_7, code=var_27, index=var_28)
    var_30 = module_0.Message(text=var_7, start_position=var_3, end_position=var_4)
    var_31 = module_0.Message(text=var_7, position=var_3)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    var_3 = 2
    var_4 = 5
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = 'Error 1'
    var_7 = 'err1'
    var_8 = 'a'
    var_9 = [var_8, var_1]
    var_10 = module_0.Message(text=var_6, code=var_7, index=var_9, position=var_2)
    var_11 = [var_8, var_1]
    var_12 = module_0.Message(text=var_6, code=var_7, index=var_11, position=var_2)
    var_13 = 'Error 2'
    var_14 = 'err2'
    var_15 = 'b'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16, position=var_5)
    var_18 = [var_10, var_17]
    var_19 = module_0.ValidationError(messages=var_18)
    var_20 = [var_10, var_17]
    var_21 = module_0.ValidationError(messages=var_20)
    var_22 = [var_10, var_12]
    var_23 = module_0.ValidationError(messages=var_22)
    var_24 = 'Single'
    var_25 = 'code'
    var_26 = module_0.ValidationError(text=var_24, code=var_25)
    var_27 = module_0.ValidationError(text=var_24, code=var_25)
    var_28 = 'Different'
    var_29 = module_0.ValidationError(text=var_28, code=var_25)
    var_30 = [var_17, var_10]
    var_31 = module_0.ValidationError(messages=var_30)



# Parsed testcases at query #7
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
    var_24 = 'admin'
    var_25 = [var_12, var_13]
    var_26 = module_0.Message(text=var_9, code=var_10, key=var_24, index=var_25, position=var_3)
    var_27 = [var_12, var_0]
    var_28 = module_0.Message(text=var_9, code=var_10, index=var_27, position=var_3)
    var_29 = []
    var_30 = module_0.Message(text=var_9, code=var_10, index=var_29, start_position=var_3, end_position=var_8)
    var_31 = []
    var_32 = module_0.Message(text=var_9, code=var_10, index=var_31, start_position=var_4, end_position=var_8)
    var_33 = module_0.Message(text=var_9)
    var_34 = 'custom'
    var_35 = []
    var_36 = None
    var_37 = module_0.Message(text=var_9, code=var_34, index=var_35, start_position=var_36, end_position=var_36)



# Parsed testcases at query #8
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 5
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_1, var_2)
    var_7 = 11
    var_8 = module_0.Position(var_0, var_7, var_2)
    var_9 = 6
    var_10 = module_0.Position(var_0, var_1, var_9)
    var_11 = 'not a position'



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid syntax'
    var_1 = 'syntax_error'
    var_2 = 'root'
    var_3 = 1
    var_4 = 5
    var_5 = 10
    var_6 = module_0.Position(var_3, var_4, var_5)
    var_7 = module_0.ParseError(text=var_0, code=var_1, key=var_2, position=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = error.messages()[var_9]
    var_11 = var_10.text
    var_12 = error.messages()[var_9]
    var_13 = var_12.code
    var_14 = error.messages()[var_9]
    var_15 = var_14.start_position
    var_16 = error.messages()[var_9]
    var_17 = var_16.end_position
    var_18 = 'Error 1'
    var_19 = 'err1'
    var_20 = 'users'
    var_21 = [var_20, var_9]
    var_22 = module_0.Message(text=var_18, code=var_19, index=var_21)
    var_23 = 'Error 2'
    var_24 = 'err2'
    var_25 = 'name'
    var_26 = [var_20, var_3, var_25]
    var_27 = module_0.Message(text=var_23, code=var_24, index=var_26)
    var_28 = [var_22, var_27]
    var_29 = module_0.ParseError(messages=var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = module_0.ParseError(text=var_0, code=var_1, key=var_2, position=var_6)
    var_32 = str(var_7)
    var_33 = repr(var_7)
    var_34 = len(var_7)
    assert var_34 == 1
    var_35 = 'text'
    var_36 = [var_22]
    var_37 = module_0.ParseError(text=var_35, messages=var_36)
    var_38 = 'text'
    var_39 = module_0.ParseError(text=var_38, position=var_6)



# Parsed testcases at query #10
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = 'Invalid format'
    var_5 = 'format_error'
    var_6 = 'email'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6, position=var_3)
    var_8 = module_0.ValidationError(text=var_4, code=var_5, key=var_6, position=var_3)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = error_single.messages()[var_10]
    var_12 = var_11.text
    assert var_12 == 'Invalid format'
    var_13 = error_single.messages()[var_10]
    var_14 = var_13.start_position
    var_15 = 'Too short'
    var_16 = 'min_length'
    var_17 = 'password'
    var_18 = [var_17]
    var_19 = module_0.Message(text=var_15, code=var_16, index=var_18)
    var_20 = 'Missing digit'
    var_21 = 'regex'
    var_22 = 'complexity'
    var_23 = [var_17, var_22]
    var_24 = module_0.Message(text=var_20, code=var_21, index=var_23)
    var_25 = [var_19, var_24]
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = 'Error'
    var_29 = [var_7]
    var_30 = module_0.ValidationError(text=var_28, messages=var_29)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    var_2 = 2
    var_3 = module_0.Position(var_2, var_2, var_2)
    var_4 = module_0.Position(var_0, var_0, var_0)
    var_5 = 'error'
    var_6 = 'err_code'
    var_7 = 'user'
    var_8 = 'users'
    var_9 = 0
    var_10 = [var_8, var_9]
    var_11 = module_0.Message(text=var_5, code=var_6, key=var_7, index=var_10, position=var_1)
    var_12 = [var_8, var_9]
    var_13 = module_0.Message(text=var_5, code=var_6, key=var_7, index=var_12, position=var_1)
    var_14 = 'different'
    var_15 = [var_8, var_9]
    var_16 = module_0.Message(text=var_14, code=var_6, key=var_7, index=var_15, position=var_1)
    var_17 = 'other_code'
    var_18 = [var_8, var_9]
    var_19 = module_0.Message(text=var_5, code=var_17, key=var_7, index=var_18, position=var_1)
    var_20 = [var_8, var_0]
    var_21 = module_0.Message(text=var_5, code=var_6, key=var_7, index=var_20, position=var_1)
    var_22 = [var_8, var_9]
    var_23 = module_0.Message(text=var_5, code=var_6, key=var_7, index=var_22, position=var_3)
    var_24 = [var_8, var_9]
    var_25 = module_0.Message(text=var_5, code=var_6, key=var_7, index=var_24, start_position=var_1, end_position=var_3)
    var_26 = module_0.Message(text=var_5, code=var_6)
    var_27 = module_0.Message(text=var_5, code=var_6)



# Parsed testcases at query #12
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
    var_30 = [var_12, var_13]
    var_31 = module_0.Message(text=var_9, code=var_10, index=var_30, start_position=var_3, end_position=var_3)



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
    var_12 = 'sub'
    var_13 = [var_12]
    var_14 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_13, position=var_3)
    var_15 = [var_12]
    var_16 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_15, position=var_4)
    var_17 = 'different'
    var_18 = [var_12]
    var_19 = module_0.Message(text=var_17, code=var_10, key=var_11, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_12]
    var_22 = module_0.Message(text=var_9, code=var_20, key=var_11, index=var_21, position=var_3)
    var_23 = 'other'
    var_24 = [var_23]
    var_25 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_24, position=var_3)
    var_26 = 'extra'
    var_27 = [var_12, var_26]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, position=var_3)
    var_29 = [var_12]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, position=var_8)
    var_31 = [var_12]
    var_32 = module_0.Message(text=var_9, code=var_10, index=var_31, start_position=var_3, end_position=var_8)
    var_33 = [var_12]
    var_34 = module_0.Message(text=var_9, code=var_10, index=var_33, start_position=var_3, end_position=var_4)
    var_35 = module_0.Message(text=var_9)
    var_36 = module_0.Message(text=var_9)



# Parsed testcases at query #14
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    var_2 = 2
    var_3 = module_0.Position(var_2, var_2, var_2)
    var_4 = module_0.Position(var_0, var_0, var_0)
    var_5 = 'error'
    var_6 = 'err_code'
    var_7 = 'key'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8, position=var_1)
    var_10 = [var_7]
    var_11 = module_0.Message(text=var_5, code=var_6, index=var_10, position=var_1)
    var_12 = 'different'
    var_13 = [var_7]
    var_14 = module_0.Message(text=var_12, code=var_6, index=var_13, position=var_1)
    var_15 = 'different_code'
    var_16 = [var_7]
    var_17 = module_0.Message(text=var_5, code=var_15, index=var_16, position=var_1)
    var_18 = 'other_key'
    var_19 = [var_18]
    var_20 = module_0.Message(text=var_5, code=var_6, index=var_19, position=var_1)
    var_21 = 123
    var_22 = [var_21]
    var_23 = module_0.Message(text=var_5, code=var_6, index=var_22, position=var_1)
    var_24 = [var_7]
    var_25 = module_0.Message(text=var_5, code=var_6, index=var_24, position=var_3)
    var_26 = [var_7]
    var_27 = module_0.Message(text=var_5, code=var_6, index=var_26, start_position=var_1, end_position=var_3)
    var_28 = module_0.Message(text=var_5, code=var_6)
    var_29 = module_0.Message(text=var_5, code=var_6)
    var_30 = module_0.Message(text=var_5, code=var_6, position=var_1)
    var_31 = module_0.Message(text=var_5, code=var_6, start_position=var_1, end_position=var_1)



# Parsed testcases at query #15
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
    var_12 = 'fields'
    var_13 = 0
    var_14 = [var_12, var_13]
    var_15 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_14, position=var_3)
    var_16 = [var_12, var_13]
    var_17 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_16, position=var_4)
    var_18 = 'different'
    var_19 = [var_12, var_13]
    var_20 = module_0.Message(text=var_18, code=var_10, key=var_11, index=var_19, position=var_3)
    var_21 = [var_12, var_13]
    var_22 = module_0.Message(text=var_9, code=var_18, key=var_11, index=var_21, position=var_3)
    var_23 = [var_12, var_13]
    var_24 = module_0.Message(text=var_9, code=var_10, key=var_18, index=var_23, position=var_3)
    var_25 = [var_12, var_0]
    var_26 = module_0.Message(text=var_9, code=var_10, index=var_25, position=var_3)
    var_27 = [var_12, var_13]
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, start_position=var_3, end_position=var_8)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, position=var_3)



# Parsed testcases at query #17
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
    var_27 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_26, start_position=var_8)
    var_28 = [var_12, var_13]
    var_29 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_28, end_position=var_8)



# Parsed testcases at query #18
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
    var_12 = None
    var_13 = 'test'
    var_14 = module_0.Message(text=var_13)



# Parsed testcases at query #19
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
    var_12 = 'profile'
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
    var_28 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_27, start_position=var_8)
    var_29 = [var_12, var_13]
    var_30 = module_0.Message(text=var_9, code=var_10, key=var_11, index=var_29, end_position=var_8)
    var_31 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #21
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
    var_11 = 'users'
    var_12 = 0
    var_13 = [var_11, var_12]
    var_14 = module_0.Message(text=var_9, code=var_10, index=var_13, position=var_3)
    var_15 = [var_11, var_12]
    var_16 = module_0.Message(text=var_9, code=var_10, index=var_15, position=var_3)
    var_17 = 'different'
    var_18 = [var_11, var_12]
    var_19 = module_0.Message(text=var_17, code=var_10, index=var_18, position=var_3)
    var_20 = 'other_code'
    var_21 = [var_11, var_12]
    var_22 = module_0.Message(text=var_9, code=var_20, index=var_21, position=var_3)
    var_23 = [var_11, var_0]
    var_24 = module_0.Message(text=var_9, code=var_10, index=var_23, position=var_3)
    var_25 = [var_11, var_12]
    var_26 = module_0.Message(text=var_9, code=var_10, index=var_25, position=var_8)
    var_27 = [var_11, var_12]
    var_28 = module_0.Message(text=var_9, code=var_10, index=var_27, position=var_4)
    var_29 = [var_11, var_12]
    var_30 = module_0.Message(text=var_9, code=var_10, index=var_29, start_position=var_3, end_position=var_4)
    var_31 = [var_11, var_12]
    var_32 = module_0.Message(text=var_9, code=var_10, index=var_31, position=var_3)
    var_33 = [var_11, var_12]
    var_34 = module_0.Message(text=var_9, code=var_10, index=var_33, start_position=var_3, end_position=var_3)



