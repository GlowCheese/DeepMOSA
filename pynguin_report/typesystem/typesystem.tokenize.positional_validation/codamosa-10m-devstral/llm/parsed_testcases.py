####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = {var_20: var_7, var_21: var_7}
    var_23 = 30
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_20: var_25, var_21: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_2.validate_with_positions(token=var_24, validator=var_28)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = 'field2'
    var_12 = module_1.Field()
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_6: var_12, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'invalid_value'
    var_20 = 12
    var_21 = module_0.Token(var_19)
    var_22 = 5
    var_23 = module_1.Field()
    var_24 = module_2.validate_with_positions(token=var_21, validator=var_23)
    var_25 = error.messages()[0]
    var_26 = 'nested'
    var_27 = 'field'
    var_28 = 'invalid'
    var_29 = {var_27: var_28}
    var_30 = {var_26: var_29}
    var_31 = 25
    var_32 = module_0.Token(var_30)
    var_33 = 10
    var_34 = module_1.Field()
    var_35 = {var_27: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = {var_26: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_32, validator=var_38)
    var_40 = error.messages()[0]



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[var_1]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = 9
    var_12 = 14
    var_13 = module_0.Token(var_4)
    var_14 = [var_13]
    var_15 = module_0.Token(var_9)
    var_16 = error.messages()[var_1]
    var_17 = 'field1'
    var_18 = 'field2'
    var_19 = 'invalid'
    var_20 = {var_17: var_4, var_18: var_19}
    var_21 = 30
    var_22 = module_0.Token(var_4)
    var_23 = 20
    var_24 = 27
    var_25 = module_0.Token(var_19)
    var_26 = [var_22, var_25]
    var_27 = module_0.Token(var_20)
    var_28 = lambda m: m.start_position.char_index



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'invalid'
    var_13 = 7
    var_14 = module_0.Token(var_12)
    var_15 = 10
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_14, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'nested'
    var_20 = {var_19: var_6}
    var_21 = 15
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Field()
    var_24 = {var_19: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_2.validate_with_positions(token=var_22, validator=var_25)
    var_27 = error.messages()[0]



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = error.messages()[var_1]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[var_1]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_6, var_22: var_6}
    var_24 = 30
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {var_0: var_1}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = list(error.messages())
    var_10 = 'user'
    var_11 = {var_0: var_8}
    var_12 = {var_10: var_11}
    var_13 = 20
    var_14 = {var_0: var_8}
    var_15 = 5
    var_16 = 15
    var_17 = module_0.Token(var_14)
    var_18 = [var_17]
    var_19 = module_0.Token(var_12)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = list(error.messages())
    var_22 = 'age'
    var_23 = 150
    var_24 = {var_22: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = module_1.validate_with_positions(token=var_25, validator=var_0)
    var_27 = list(error.messages())



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = error.messages()[0]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 20
    var_16 = module_0.Token(var_14)
    var_17 = module_1.Field()
    var_18 = {var_13: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = module_2.validate_with_positions(token=var_16, validator=var_19)
    var_21 = error.messages()[0]
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = 'invalid'
    var_25 = {var_22: var_7, var_23: var_24}
    var_26 = 30
    var_27 = module_0.Token(var_25)
    var_28 = module_1.Field()
    var_29 = module_1.Field()
    var_30 = {var_22: var_28, var_23: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_27, validator=var_31)
    var_33 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_18 = error.messages()[0]
    var_19 = {}
    var_20 = module_0.Token(var_19)
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = e.messages()[0]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = list(error.messages())
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = list(error.messages())
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = 'invalid'
    var_24 = {var_21: var_7, var_22: var_23}
    var_25 = 30
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = 10
    var_29 = module_1.Field()
    var_30 = {var_21: var_27, var_22: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_26, validator=var_31)
    var_33 = list(error.messages())



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = 'field'
    var_10 = 'invalid_value'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 20
    var_14 = {var_9: var_10}
    var_15 = 10
    var_16 = 2
    var_17 = 9
    var_18 = module_0.Token(var_10)
    var_19 = [var_18]
    var_20 = module_0.Token(var_14)
    var_21 = [var_20]
    var_22 = module_0.Token(var_12)
    var_23 = error.messages()[0]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = {var_24: var_4, var_25: var_10}
    var_27 = 30
    var_28 = module_0.Token(var_4)
    var_29 = 11
    var_30 = module_0.Token(var_10)
    var_31 = [var_28, var_30]
    var_32 = module_0.Token(var_26)
    var_33 = error.messages()[0]
    var_34 = error.messages()[1]



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = e.messages()[0]
    var_10 = 'user'
    var_11 = 'name'
    var_12 = 'user'
    var_13 = {}
    var_14 = {var_12: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)
    var_17 = e.messages()[0]
    var_18 = {}
    var_19 = module_0.Token(var_18)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'missing_field'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = 20
    var_10 = module_0.Token(var_8)
    var_11 = 'required_field'
    var_12 = True
    var_13 = module_1.Field()
    var_14 = {var_11: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_2.validate_with_positions(token=var_10, validator=var_15)
    var_17 = error.messages()[0]
    var_18 = 'nested'
    var_19 = 'invalid'
    var_20 = 'bad_value'
    var_21 = {var_19: var_20}
    var_22 = {var_18: var_21}
    var_23 = 30
    var_24 = module_0.Token(var_22)
    var_25 = 10
    var_26 = module_1.Field()
    var_27 = {var_19: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {var_18: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_24, validator=var_30)
    var_32 = error.messages()[0]
    var_33 = 'field1'
    var_34 = 'field2'
    var_35 = 'a'
    var_36 = 'b'
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = 25
    var_39 = module_0.Token(var_37)
    var_40 = module_1.Field()
    var_41 = module_1.Field()
    var_42 = {var_33: var_40, var_34: var_41}
    var_43 = module_3.Schema(var_42)
    var_44 = module_2.validate_with_positions(token=var_39, validator=var_43)
    var_45 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 15
    var_10 = 10
    var_11 = module_0.Token(var_4)
    var_12 = [var_11]
    var_13 = module_0.Token(var_8)
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = 'invalid'
    var_17 = {var_14: var_4, var_15: var_16}
    var_18 = 30
    var_19 = module_0.Token(var_4)
    var_20 = 20
    var_21 = 27
    var_22 = module_0.Token(var_16)
    var_23 = [var_19, var_22]
    var_24 = module_0.Token(var_17)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'value1'
    var_9 = None
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 20
    var_12 = module_0.Token(var_10)
    var_13 = module_1.Field()
    var_14 = True
    var_15 = module_1.Field()
    var_16 = {var_6: var_13, var_7: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_12, validator=var_17)
    var_19 = error.messages()[var_1]
    var_20 = [var_7]
    var_21 = var_12.lookup(var_20)
    var_22 = var_21.start
    var_23 = [var_7]
    var_24 = var_12.lookup(var_23)
    var_25 = var_24.end
    var_26 = 'nested'
    var_27 = {var_6: var_8, var_7: var_9}
    var_28 = {var_26: var_27}
    var_29 = 30
    var_30 = module_0.Token(var_28)
    var_31 = module_1.Field()
    var_32 = module_1.Field()
    var_33 = {var_6: var_31, var_7: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = {var_26: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_2.validate_with_positions(token=var_30, validator=var_36)
    var_38 = error.messages()[var_1]
    var_39 = [var_26, var_7]
    var_40 = var_30.lookup(var_39)
    var_41 = var_40.start
    var_42 = [var_26, var_7]
    var_43 = var_30.lookup(var_42)
    var_44 = var_43.end
    var_45 = {var_6: var_9, var_7: var_9}
    var_46 = module_0.Token(var_45)
    var_47 = module_1.Field()
    var_48 = module_1.Field()
    var_49 = {var_6: var_47, var_7: var_48}
    var_50 = module_3.Schema(var_49)
    var_51 = module_2.validate_with_positions(token=var_46, validator=var_50)
    var_52 = lambda m: m.start_position.char_index



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = ''
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[var_1]
    var_8 = 'nested'
    var_9 = 'field'
    var_10 = {var_9: var_4}
    var_11 = {var_8: var_10}
    var_12 = 20
    var_13 = module_0.Token(var_11)
    var_14 = error.messages()[var_1]
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = {var_15: var_4, var_16: var_4}
    var_18 = module_0.Token(var_17)
    var_19 = lambda m: m.start_position.char_index



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = list(e.messages())
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = {}
    var_15 = 5
    var_16 = 7
    var_17 = module_0.Token(var_14)
    var_18 = {var_11: var_17}
    var_19 = module_0.Token(var_13)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = list(e.messages())
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = list(e.messages())



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = exc_info.value.messages()[var_1]
    var_12 = var_11.text
    assert var_12 == "The field 'field_name' is required."
    var_13 = 'invalid_value'
    var_14 = 12
    var_15 = module_0.Token(var_13)
    var_16 = 'expected_value'
    var_17 = lambda v: v == var_16
    var_18 = [var_17]
    var_19 = module_1.Field()
    var_20 = module_2.validate_with_positions(token=var_15, validator=var_19)
    var_21 = exc_info.value.messages()[var_1]
    var_22 = var_21.start_position
    assert var_22 == 0
    var_23 = exc_info.value.messages()[var_1]
    var_24 = var_23.end_position
    assert var_24 == 12
    var_25 = 'nested'
    var_26 = {var_25: var_6}
    var_27 = 15
    var_28 = module_0.Token(var_26)
    var_29 = module_1.Field()
    var_30 = {var_25: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_28, validator=var_31)
    var_33 = exc_info.value.messages()[var_1]
    var_34 = var_33.text
    assert var_34 == "The field 'nested' is required."



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = exc_info.value.messages()[var_1]
    var_12 = 'invalid'
    var_13 = 7
    var_14 = module_0.Token(var_12)
    var_15 = 10
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_14, validator=var_16)
    var_18 = exc_info.value.messages()[var_1]
    var_19 = 'nested'
    var_20 = {var_19: var_7}
    var_21 = 15
    var_22 = 8
    var_23 = 13
    var_24 = module_0.Token(var_7)
    var_25 = [var_24]
    var_26 = module_0.Token(var_20)
    var_27 = module_1.Field()
    var_28 = {var_19: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_26, validator=var_29)
    var_31 = exc_info.value.messages()[var_1]



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'invalid_value'
    var_13 = module_0.Token(var_12)
    var_14 = 5
    var_15 = module_1.Field()
    var_16 = module_2.validate_with_positions(token=var_13, validator=var_15)
    var_17 = error.messages()[0]
    var_18 = 'nested'
    var_19 = {var_18: var_6}
    var_20 = module_0.Token(var_19)
    var_21 = module_1.Field()
    var_22 = {var_18: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_23)
    var_25 = error.messages()[0]



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = 'age'
    var_10 = module_0.Token(var_7)
    var_11 = [var_8, var_10]
    var_12 = module_0.Token(var_6)
    var_13 = module_1.validate_with_positions(token=var_12, validator=var_0)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = e.messages()[0]
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = e.messages()[0]
    var_17 = {}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = sorted(e.messages(), key=lambda m: m.start_position.char_index)
    var_21 = {var_0: var_19}
    var_22 = 15
    var_23 = module_0.Token(var_21)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = e.messages()[0]



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = list(error.messages())
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = 15
    var_15 = {}
    var_16 = 7
    var_17 = 14
    var_18 = module_0.Token(var_15)
    var_19 = [var_18]
    var_20 = module_0.Token(var_13)
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = list(error.messages())
    var_23 = {}
    var_24 = []
    var_25 = module_0.Token(var_23)
    var_26 = module_1.validate_with_positions(token=var_25, validator=var_0)
    var_27 = list(error.messages())
    var_28 = 123
    var_29 = {var_0: var_28}
    var_30 = 8
    var_31 = module_0.Token(var_28)
    var_32 = [var_31]
    var_33 = module_0.Token(var_29)
    var_34 = module_1.validate_with_positions(token=var_33, validator=var_0)
    var_35 = list(error.messages())



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = 'name'
    var_7 = True
    var_8 = error.messages()[0]
    var_9 = 'invalid_email'
    var_10 = 12
    var_11 = module_0.Token(var_9)
    var_12 = 'email'
    var_13 = module_1.Field()
    var_14 = module_2.validate_with_positions(token=var_11, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'invalid'
    var_17 = {var_6: var_4, var_12: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_4)
    var_20 = 6
    var_21 = module_0.Token(var_16)
    var_22 = [var_19, var_21]
    var_23 = module_0.Token(var_17)
    var_24 = module_1.Field()



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 15
    var_8 = module_0.Token(var_6)
    var_9 = 'age'
    var_10 = True
    var_11 = e.messages()[0]
    var_12 = 'not_a_number'
    var_13 = {var_9: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = e.messages()[0]
    var_17 = 'user'
    var_18 = 123
    var_19 = {var_4: var_18}
    var_20 = {var_17: var_19}
    var_21 = 25
    var_22 = module_0.Token(var_20)
    var_23 = e.messages()[0]
    var_24 = {var_4: var_18, var_9: var_12}
    var_25 = 30
    var_26 = module_0.Token(var_24)
    var_27 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test'
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'user'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = 'email'
    var_22 = True
    var_23 = module_1.Field()
    var_24 = {var_21: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_16: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_20, validator=var_27)
    var_29 = error.messages()[0]
    var_30 = 'a'
    var_31 = 'b'
    var_32 = None
    var_33 = {var_30: var_32, var_31: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = True
    var_36 = lambda x: x is not var_32
    var_37 = [var_36]
    var_38 = module_1.Field()
    var_39 = True
    var_40 = lambda x: x is not var_32
    var_41 = [var_40]
    var_42 = module_1.Field()
    var_43 = {var_30: var_38, var_31: var_42}
    var_44 = module_3.Schema(var_43)
    var_45 = module_2.validate_with_positions(token=var_34, validator=var_44)
    var_46 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = exc_info.value.messages()[var_1]
    var_12 = var_11.code
    assert var_12 == 'required'
    var_13 = exc_info.value.messages()[var_1]
    var_14 = var_13.text
    assert var_14 == "The field 'this' is required."
    var_15 = 'invalid_value'
    var_16 = module_0.Token(var_15)
    var_17 = 5
    var_18 = module_1.Field()
    var_19 = module_2.validate_with_positions(token=var_16, validator=var_18)
    var_20 = exc_info.value.messages()[var_1]
    var_21 = var_20.code
    assert var_21 == 'min_length'
    var_22 = exc_info.value.messages()[var_1]
    var_23 = var_22.start_position
    assert var_23 == 0
    var_24 = exc_info.value.messages()[var_1]
    var_25 = var_24.end_position
    assert var_25 == 10
    var_26 = 'nested'
    var_27 = {var_26: var_6}
    var_28 = 20
    var_29 = module_0.Token(var_27)
    var_30 = module_1.Field()
    var_31 = {var_26: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = module_2.validate_with_positions(token=var_29, validator=var_32)
    var_34 = exc_info.value.messages()[var_1]
    var_35 = var_34.code
    assert var_35 == 'required'
    var_36 = exc_info.value.messages()[var_1]
    var_37 = var_36.text
    assert var_37 == "The field 'nested' is required."
    var_38 = exc_info.value.messages()[var_1]
    var_39 = var_38.start_position
    assert var_39 == 0
    var_40 = exc_info.value.messages()[var_1]
    var_41 = var_40.end_position
    assert var_41 == 20



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'age'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_13)
    var_22 = error.messages()[0]
    var_23 = None
    var_24 = {var_9: var_23, var_16: var_17}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = True
    var_28 = module_1.Field()
    var_29 = module_2.validate_with_positions(token=var_26, validator=var_13)
    var_30 = [m for m in error.messages() if m.code == 'required'][0]
    var_31 = [m for m in error.messages() if m.code == 'type'][0]



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = error.messages()[0]
    var_13 = 'invalid_value'
    var_14 = 12
    var_15 = module_0.Token(var_13)
    var_16 = False
    var_17 = 'valid_value'
    var_18 = lambda x: x == var_17
    var_19 = [var_18]
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = error.messages()[0]
    var_23 = 'nested'
    var_24 = {var_23: var_7}
    var_25 = 15
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = {var_23: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_26, validator=var_29)
    var_31 = error.messages()[0]



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = 20
    var_10 = module_0.Token(var_8)
    var_11 = 'field2'
    var_12 = module_1.Field()
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_6: var_12, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = 'invalid_value'
    var_19 = {var_6: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = module_1.Field()
    var_22 = {var_6: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_23)
    var_25 = 'nested'
    var_26 = {var_6: var_18}
    var_27 = {var_25: var_26}
    var_28 = 30
    var_29 = module_0.Token(var_27)
    var_30 = module_1.Field()
    var_31 = {var_6: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = {var_25: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_29, validator=var_34)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = error.messages()[0]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 15
    var_16 = module_0.Token(var_14)
    var_17 = module_1.Field()
    var_18 = {var_13: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = module_2.validate_with_positions(token=var_16, validator=var_19)
    var_21 = error.messages()[0]
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = {var_22: var_7, var_23: var_7}
    var_25 = 30
    var_26 = 10
    var_27 = module_0.Token(var_7)
    var_28 = 25
    var_29 = module_0.Token(var_7)
    var_30 = [var_27, var_29]
    var_31 = module_0.Token(var_24)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_22: var_32, var_23: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 20
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 30
    var_15 = module_0.Token(var_13)
    var_16 = 'invalid_value'
    var_17 = module_0.Token(var_16)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 15
    var_10 = 7
    var_11 = 12
    var_12 = module_0.Token(var_4)
    var_13 = [var_12]
    var_14 = module_0.Token(var_8)
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = 'invalid'
    var_18 = {var_15: var_4, var_16: var_17}
    var_19 = 30
    var_20 = module_0.Token(var_4)
    var_21 = 18
    var_22 = 25
    var_23 = module_0.Token(var_17)
    var_24 = [var_20, var_23]
    var_25 = module_0.Token(var_18)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = {var_20: var_7, var_21: var_7}
    var_23 = 20
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_20: var_25, var_21: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_2.validate_with_positions(token=var_24, validator=var_28)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 5
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 15
    var_14 = {}
    var_15 = 7
    var_16 = 14
    var_17 = module_0.Token(var_14)
    var_18 = [var_17]
    var_19 = module_0.Token(var_12)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = {}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field(default=var_0)
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'age'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_13)
    var_22 = error.messages()[0]
    var_23 = None
    var_24 = {var_9: var_23, var_16: var_17}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = True
    var_28 = module_1.Field()
    var_29 = module_2.validate_with_positions(token=var_26, validator=var_13)
    var_30 = [m for m in error.messages() if m.code == 'required'][0]
    var_31 = [m for m in error.messages() if m.code == 'invalid_type'][0]
    var_32 = 'user'
    var_33 = {var_9: var_23}
    var_34 = {var_32: var_33}
    var_35 = 15
    var_36 = {var_9: var_23}
    var_37 = 7
    var_38 = 12
    var_39 = module_0.Token(var_23)
    var_40 = [var_39]
    var_41 = module_0.Token(var_36)
    var_42 = [var_41]
    var_43 = module_0.Token(var_34)
    var_44 = True
    var_45 = module_1.Field()
    var_46 = {var_9: var_45}
    var_47 = module_3.Schema(var_46)
    var_48 = {var_32: var_47}
    var_49 = module_3.Schema(var_48)
    var_50 = module_2.validate_with_positions(token=var_43, validator=var_49)
    var_51 = error.messages()[0]



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = 'name'
    var_9 = True
    var_10 = module_1.Field()
    var_11 = {var_8: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = module_2.validate_with_positions(token=var_7, validator=var_12)
    var_14 = error.messages()[0]
    var_15 = 'age'
    var_16 = 'test'
    var_17 = 'invalid'
    var_18 = {var_8: var_16, var_15: var_17}
    var_19 = 20
    var_20 = module_0.Token(var_18)
    var_21 = module_1.Field()
    var_22 = module_2.validate_with_positions(token=var_20, validator=var_12)
    var_23 = error.messages()[0]
    var_24 = 'user'
    var_25 = None
    var_26 = {var_8: var_25}
    var_27 = {var_24: var_26}
    var_28 = 30
    var_29 = module_0.Token(var_27)
    var_30 = module_1.Field()
    var_31 = {var_8: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = {var_24: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_29, validator=var_34)
    var_36 = error.messages()[0]



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = 'field'
    var_10 = 'invalid'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = error.messages()[0]
    var_16 = 'field1'
    var_17 = 'field2'
    var_18 = {var_16: var_10, var_17: var_4}
    var_19 = 30
    var_20 = module_0.Token(var_18)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = 'field2'
    var_12 = module_1.Field()
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_6: var_12, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'invalid_value'
    var_20 = 12
    var_21 = module_0.Token(var_19)
    var_22 = 5
    var_23 = module_1.Field()
    var_24 = module_2.validate_with_positions(token=var_21, validator=var_23)
    var_25 = error.messages()[0]
    var_26 = 'nested'
    var_27 = 'field'
    var_28 = {var_27: var_19}
    var_29 = {var_26: var_28}
    var_30 = 25
    var_31 = {var_27: var_19}
    var_32 = 8
    var_33 = 17
    var_34 = module_0.Token(var_19)
    var_35 = [var_34]
    var_36 = module_0.Token(var_31)
    var_37 = [var_36]
    var_38 = module_0.Token(var_29)
    var_39 = module_1.Field()
    var_40 = {var_27: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = {var_26: var_41}
    var_43 = module_3.Schema(var_42)
    var_44 = module_2.validate_with_positions(token=var_38, validator=var_43)
    var_45 = error.messages()[0]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = 'nested'
    var_12 = 'field'
    var_13 = {var_12: var_6}
    var_14 = {var_11: var_13}
    var_15 = 20
    var_16 = {var_12: var_6}
    var_17 = 5
    var_18 = 15
    var_19 = 10
    var_20 = module_0.Token(var_6)
    var_21 = [var_20]
    var_22 = module_0.Token(var_16)
    var_23 = [var_22]
    var_24 = module_0.Token(var_14)
    var_25 = module_1.Field()
    var_26 = {var_12: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = {var_11: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_24, validator=var_29)
    var_31 = 'field1'
    var_32 = 'field2'
    var_33 = {var_31: var_6, var_32: var_6}
    var_34 = module_0.Token(var_6)
    var_35 = module_0.Token(var_6)
    var_36 = [var_34, var_35]
    var_37 = module_0.Token(var_33)
    var_38 = module_1.Field()
    var_39 = module_1.Field()
    var_40 = {var_31: var_38, var_32: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = module_2.validate_with_positions(token=var_37, validator=var_41)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = e.messages()[0]
    var_11 = 'wrong'
    var_12 = 5
    var_13 = module_0.Token(var_11)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = e.messages()[0]
    var_16 = 'user'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = 12
    var_20 = module_0.Token(var_18)
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = e.messages()[0]



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'age'
    var_17 = -5
    var_18 = {var_16: var_17}
    var_19 = 7
    var_20 = module_0.Token(var_18)
    var_21 = module_1.Field()
    var_22 = {var_16: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_23)
    var_25 = error.messages()[0]
    var_26 = None
    var_27 = -5
    var_28 = {var_9: var_26, var_16: var_27}
    var_29 = 15
    var_30 = module_0.Token(var_28)
    var_31 = True
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_9: var_32, var_16: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_30, validator=var_35)
    var_37 = [m for m in error.messages() if m.code == 'required'][0]
    var_38 = [m for m in error.messages() if m.code == 'min_value'][0]



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_6}
    var_13 = 15
    var_14 = 10
    var_15 = module_0.Token(var_6)
    var_16 = [var_15]
    var_17 = module_0.Token(var_12)
    var_18 = module_1.Field()
    var_19 = {var_11: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = module_2.validate_with_positions(token=var_17, validator=var_20)
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = 'invalid'
    var_25 = {var_22: var_6, var_23: var_24}
    var_26 = 30
    var_27 = module_0.Token(var_6)
    var_28 = 20
    var_29 = 27
    var_30 = module_0.Token(var_24)
    var_31 = [var_27, var_30]
    var_32 = module_0.Token(var_25)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_22: var_33, var_23: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = ''
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 10
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 20
    var_17 = module_0.Token(var_15)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = ''
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_6}
    var_13 = 10
    var_14 = module_0.Token(var_12)
    var_15 = module_1.Field()
    var_16 = {var_11: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_14, validator=var_17)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0)
    var_4 = 'not_an_int'
    var_5 = 9
    var_6 = module_0.Token(var_4)
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 17
    var_10 = 10
    var_11 = module_0.Token(var_4)
    var_12 = [var_11]
    var_13 = module_0.Token(var_8)
    var_14 = 'required_field'
    var_15 = True
    var_16 = {}
    var_17 = module_0.Token(var_16)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_value'
    var_2 = 0
    var_3 = 5
    var_4 = module_1.Token(var_1)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_0)
    assert var_5 == 'valid_value'
    var_6 = 'name'
    var_7 = True
    var_8 = module_0.Field()
    var_9 = {var_6: var_8}
    var_10 = module_3.Schema(var_9)
    var_11 = 'age'
    var_12 = 25
    var_13 = {var_11: var_12}
    var_14 = 10
    var_15 = module_1.Token(var_13)
    var_16 = module_2.validate_with_positions(token=var_15, validator=var_10)
    var_17 = error.messages()[var_2]
    var_18 = module_0.Field()
    var_19 = 'short'
    var_20 = module_1.Token(var_19)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_18)
    var_22 = error.messages()[var_2]
    var_23 = 'user'
    var_24 = module_0.Field()
    var_25 = {var_6: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = {var_23: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {var_11: var_12}
    var_30 = {var_23: var_29}
    var_31 = 20
    var_32 = module_1.Token(var_30)
    var_33 = module_2.validate_with_positions(token=var_32, validator=var_28)
    var_34 = error.messages()[var_2]



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = 'invalid'
    var_24 = {var_21: var_7, var_22: var_23}
    var_25 = 30
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_21: var_27, var_22: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = 'age'
    var_12 = module_1.Field()
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_6: var_12, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'user'
    var_20 = 'email'
    var_21 = 'invalid'
    var_22 = {var_6: var_7, var_20: var_21}
    var_23 = {var_19: var_22}
    var_24 = 30
    var_25 = {var_6: var_7, var_20: var_21}
    var_26 = 7
    var_27 = 28
    var_28 = 14
    var_29 = 18
    var_30 = module_0.Token(var_7)
    var_31 = 25
    var_32 = 32
    var_33 = module_0.Token(var_21)
    var_34 = [var_30, var_33]
    var_35 = module_0.Token(var_25)
    var_36 = [var_35]
    var_37 = module_0.Token(var_23)
    var_38 = module_1.Field()
    var_39 = module_1.Field()
    var_40 = {var_6: var_38, var_20: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = {var_19: var_41}
    var_43 = module_3.Schema(var_42)
    var_44 = module_2.validate_with_positions(token=var_37, validator=var_43)
    var_45 = error.messages()[0]
    var_46 = ''
    var_47 = {var_6: var_46, var_11: var_21}
    var_48 = 20
    var_49 = 8
    var_50 = 9
    var_51 = module_0.Token(var_46)
    var_52 = 16
    var_53 = 23
    var_54 = module_0.Token(var_21)
    var_55 = [var_51, var_54]
    var_56 = module_0.Token(var_47)
    var_57 = module_1.Field()
    var_58 = module_2.validate_with_positions(token=var_56, validator=var_43)
    var_59 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = error.messages()[0]
    var_16 = {}
    var_17 = module_0.Token(var_16)
    var_18 = module_1.validate_with_positions(token=var_17, validator=var_0)
    var_19 = sorted(error.messages(), key=lambda m: m.index)
    var_20 = 123
    var_21 = {var_0: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)
    var_24 = error.messages()[0]



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {var_0: var_1}
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = list(error.messages())
    var_10 = 'user'
    var_11 = {var_0: var_8}
    var_12 = {var_10: var_11}
    var_13 = 20
    var_14 = {var_0: var_8}
    var_15 = 5
    var_16 = 15
    var_17 = module_0.Token(var_14)
    var_18 = [var_17]
    var_19 = module_0.Token(var_12)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = list(error.messages())
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = list(error.messages())



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = exc_info.value.messages()[var_4]
    var_13 = var_12.text
    assert var_13 == "The field 'this' is required."
    var_14 = 'nested'
    var_15 = {var_14: var_7}
    var_16 = 20
    var_17 = 5
    var_18 = 15
    var_19 = module_0.Token(var_7)
    var_20 = {var_14: var_19}
    var_21 = module_0.Token(var_15)
    var_22 = module_1.Field()
    var_23 = {var_14: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_2.validate_with_positions(token=var_21, validator=var_24)
    var_26 = exc_info.value.messages()[var_4]
    var_27 = 'field1'
    var_28 = 'field2'
    var_29 = {var_27: var_7, var_28: var_7}
    var_30 = 30
    var_31 = module_0.Token(var_7)
    var_32 = module_0.Token(var_7)
    var_33 = {var_27: var_31, var_28: var_32}
    var_34 = module_0.Token(var_29)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_27: var_35, var_28: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_34, validator=var_38)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test'
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'age'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = 15
    var_20 = module_0.Token(var_18)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_13)
    var_22 = error.messages()[0]
    var_23 = None
    var_24 = {var_9: var_23, var_16: var_17}
    var_25 = 25
    var_26 = module_0.Token(var_24)
    var_27 = True
    var_28 = module_1.Field()
    var_29 = module_2.validate_with_positions(token=var_26, validator=var_13)
    var_30 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = error.messages()[0]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 10
    var_16 = module_0.Token(var_14)
    var_17 = module_1.Field()
    var_18 = {var_13: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = module_2.validate_with_positions(token=var_16, validator=var_19)
    var_21 = error.messages()[0]
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = {var_22: var_7, var_23: var_7}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)
    var_32 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_6}
    var_13 = 15
    var_14 = 8
    var_15 = 13
    var_16 = module_0.Token(var_6)
    var_17 = [var_16]
    var_18 = module_0.Token(var_12)
    var_19 = module_1.Field()
    var_20 = {var_11: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = 'invalid'
    var_26 = {var_23: var_6, var_24: var_25}
    var_27 = 30
    var_28 = module_0.Token(var_6)
    var_29 = 20
    var_30 = 27
    var_31 = module_0.Token(var_25)
    var_32 = [var_28, var_31]
    var_33 = module_0.Token(var_26)
    var_34 = module_1.Field()
    var_35 = 10
    var_36 = module_1.Field()
    var_37 = {var_23: var_34, var_24: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_33, validator=var_38)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = list(e.messages())
    var_11 = 'age'
    var_12 = -5
    var_13 = {var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = list(e.messages())



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = 'invalid'
    var_24 = {var_21: var_6, var_22: var_23}
    var_25 = 30
    var_26 = module_0.Token(var_6)
    var_27 = 15
    var_28 = module_0.Token(var_23)
    var_29 = [var_26, var_28]
    var_30 = module_0.Token(var_24)
    var_31 = module_1.Field()
    var_32 = module_1.Field()
    var_33 = {var_21: var_31, var_22: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_30, validator=var_34)
    var_36 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_7, var_22: var_7}
    var_24 = 30
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test'
    var_6 = 'name'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = True
    var_12 = module_1.Field()
    var_13 = {var_6: var_12}
    var_14 = module_3.Schema(var_13)
    var_15 = module_2.validate_with_positions(token=var_10, validator=var_14)
    var_16 = error.messages()[0]
    var_17 = 'user'
    var_18 = {var_6: var_7}
    var_19 = {var_17: var_18}
    var_20 = 25
    var_21 = module_0.Token(var_19)
    var_22 = module_1.Field()
    var_23 = {var_6: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {var_17: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_21, validator=var_26)
    var_28 = error.messages()[0]
    var_29 = 'age'
    var_30 = 'invalid'
    var_31 = {var_6: var_7, var_29: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = module_1.Field()
    var_35 = module_2.validate_with_positions(token=var_33, validator=var_26)
    var_36 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = 'age'
    var_7 = 'invalid'
    var_8 = {var_6: var_7}
    var_9 = 20
    var_10 = module_0.Token(var_7)
    var_11 = [var_10]
    var_12 = module_0.Token(var_8)
    var_13 = module_1.validate_with_positions(token=var_12, validator=var_0)



