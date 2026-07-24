####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_16 = 'invalid'
    var_17 = {var_15: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_12)
    var_20 = error.messages()[0]
    var_21 = 'user'
    var_22 = None
    var_23 = {var_8: var_22}
    var_24 = {var_21: var_23}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = {var_8: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = {var_21: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_26, validator=var_31)
    var_33 = error.messages()[0]



# Parsed testcases at query #2
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
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = 20
    var_10 = module_0.Token(var_8)
    var_11 = 'field2'
    var_12 = True
    var_13 = module_1.Field()
    var_14 = module_1.Field()
    var_15 = {var_6: var_13, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'invalid_value'
    var_20 = {var_6: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = 'valid_value'
    var_23 = lambda x: x == var_22
    var_24 = [var_23]
    var_25 = module_1.Field()
    var_26 = module_2.validate_with_positions(token=var_21, validator=var_25)
    var_27 = error.messages()[0]
    var_28 = 'nested'
    var_29 = 'field'
    var_30 = {var_29: var_19}
    var_31 = {var_28: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = lambda x: x == var_22
    var_35 = [var_34]
    var_36 = module_1.Field()
    var_37 = {var_29: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = {var_28: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_33, validator=var_40)
    var_42 = error.messages()[0]



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_value'
    var_2 = 0
    var_3 = 10
    var_4 = module_1.Token(var_1)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_0)
    assert var_5 == 'valid_value'
    var_6 = True
    var_7 = module_0.Field()
    var_8 = None
    var_9 = module_1.Token(var_8)
    var_10 = module_2.validate_with_positions(token=var_9, validator=var_7)
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = module_0.Field()
    var_14 = {var_12: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = {}
    var_17 = 5
    var_18 = module_1.Token(var_8)
    var_19 = {var_12: var_18}
    var_20 = module_1.Token(var_16)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_15)
    var_22 = error.messages()[0]
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = module_0.Field()
    var_26 = module_0.Field()
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {}
    var_30 = 2
    var_31 = module_1.Token(var_8)
    var_32 = 3
    var_33 = 4
    var_34 = module_1.Token(var_8)
    var_35 = {var_23: var_31, var_24: var_34}
    var_36 = module_1.Token(var_29)
    var_37 = module_2.validate_with_positions(token=var_36, validator=var_28)
    var_38 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = {var_4: var_0}
    var_6 = 15
    var_7 = module_0.Token(var_5)
    var_8 = 'age'
    var_9 = True
    var_10 = 'user'
    var_11 = 'invalid'
    var_12 = {var_4: var_0, var_8: var_11}
    var_13 = {var_10: var_12}
    var_14 = 30
    var_15 = module_0.Token(var_13)
    var_16 = 'email'
    var_17 = {var_4: var_0, var_8: var_11, var_16: var_11}
    var_18 = 40
    var_19 = module_0.Token(var_17)



# Parsed testcases at query #5
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
    var_16 = 'user'
    var_17 = 'email'
    var_18 = 'invalid'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 25
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Field()
    var_24 = {var_17: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_16: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_22, validator=var_27)
    var_29 = error.messages()[0]
    var_30 = 'age'
    var_31 = ''
    var_32 = {var_9: var_31, var_30: var_18}
    var_33 = 20
    var_34 = module_0.Token(var_32)
    var_35 = module_1.Field()
    var_36 = 'integer'
    var_37 = module_1.Field()
    var_38 = {var_9: var_35, var_30: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = module_2.validate_with_positions(token=var_34, validator=var_39)
    var_41 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #6
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
    var_10 = module_0.Token(var_4)
    var_11 = [var_10]
    var_12 = module_0.Token(var_8)
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_4)
    var_18 = module_0.Token(var_4)
    var_19 = [var_17, var_18]
    var_20 = module_0.Token(var_15)



# Parsed testcases at query #7
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = 8
    var_16 = 14
    var_17 = module_0.Token(var_7)
    var_18 = {var_12: var_17}
    var_19 = module_0.Token(var_13)
    var_20 = module_1.Field()
    var_21 = {var_12: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_24 = error.messages()[0]
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = 'invalid'
    var_28 = {var_25: var_7, var_26: var_27}
    var_29 = 30
    var_30 = module_0.Token(var_7)
    var_31 = 16
    var_32 = 24
    var_33 = module_0.Token(var_27)
    var_34 = {var_25: var_30, var_26: var_33}
    var_35 = module_0.Token(var_28)
    var_36 = module_1.Field()
    var_37 = 10
    var_38 = module_1.Field()
    var_39 = {var_25: var_36, var_26: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_35, validator=var_40)
    var_42 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #8
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
    var_6 = 'invalid_value'
    var_7 = module_0.Token(var_6)
    var_8 = 'name'
    var_9 = True
    var_10 = module_1.Field()
    var_11 = {var_8: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = None
    var_14 = {var_8: var_13}
    var_15 = 5
    var_16 = 7
    var_17 = module_0.Token(var_13)
    var_18 = [var_17]
    var_19 = module_0.Token(var_14)
    var_20 = module_2.validate_with_positions(token=var_19, validator=var_12)
    var_21 = module_0.Token(var_6)
    var_22 = module_1.Field()
    var_23 = 'short'
    var_24 = module_0.Token(var_23)
    var_25 = module_2.validate_with_positions(token=var_24, validator=var_22)
    var_26 = 'age'
    var_27 = 'invalid'
    var_28 = {var_8: var_13, var_26: var_27}
    var_29 = 20
    var_30 = module_0.Token(var_13)
    var_31 = 17
    var_32 = module_0.Token(var_27)
    var_33 = [var_30, var_32]
    var_34 = module_0.Token(var_28)
    var_35 = module_1.Field()
    var_36 = module_2.validate_with_positions(token=var_34, validator=var_12)



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
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_7}
    var_13 = 10
    var_14 = module_0.Token(var_12)
    var_15 = module_1.Field()
    var_16 = {var_11: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_14, validator=var_17)
    var_19 = 'field1'
    var_20 = 'field2'
    var_21 = {var_19: var_7, var_20: var_7}
    var_22 = 20
    var_23 = module_0.Token(var_21)
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_19: var_24, var_20: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_23, validator=var_27)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = None
    var_12 = {var_0: var_11}
    var_13 = {var_10: var_12}
    var_14 = 20
    var_15 = {var_0: var_11}
    var_16 = 5
    var_17 = 15
    var_18 = module_0.Token(var_11)
    var_19 = [var_18]
    var_20 = module_0.Token(var_15)
    var_21 = [var_20]
    var_22 = module_0.Token(var_13)
    var_23 = error.messages()[0]



# Parsed testcases at query #11
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
    var_15 = 'user'
    var_16 = 'email'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = 20
    var_21 = module_0.Token(var_19)
    var_22 = module_1.Field()
    var_23 = {var_16: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {var_15: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_21, validator=var_26)
    var_28 = 'age'
    var_29 = None
    var_30 = 'not_a_number'
    var_31 = {var_9: var_29, var_28: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = True
    var_35 = module_1.Field()
    var_36 = 'integer'
    var_37 = module_1.Field()
    var_38 = {var_9: var_35, var_28: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = module_2.validate_with_positions(token=var_33, validator=var_39)



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
    var_18 = 'invalid_value'
    var_19 = {var_6: var_18}
    var_20 = 20
    var_21 = module_0.Token(var_19)
    var_22 = 10
    var_23 = module_1.Field()
    var_24 = {var_6: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_2.validate_with_positions(token=var_21, validator=var_25)
    var_27 = 'short'
    var_28 = 'value2'
    var_29 = {var_6: var_27, var_11: var_28}
    var_30 = 25
    var_31 = 5
    var_32 = module_0.Token(var_27)
    var_33 = module_0.Token(var_28)
    var_34 = [var_32, var_33]
    var_35 = module_0.Token(var_29)
    var_36 = 'field3'
    var_37 = module_1.Field()
    var_38 = module_1.Field()
    var_39 = module_1.Field()
    var_40 = {var_6: var_37, var_11: var_38, var_36: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = module_2.validate_with_positions(token=var_35, validator=var_41)



# Parsed testcases at query #13
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
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = e.messages()[0]
    var_16 = {}
    var_17 = module_0.Token(var_16)
    var_18 = module_1.validate_with_positions(token=var_17, validator=var_0)
    var_19 = sorted(e.messages(), key=lambda m: m.index)
    var_20 = 123
    var_21 = {var_0: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)
    var_24 = e.messages()[0]



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
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
    var_14 = 10
    var_15 = 5
    var_16 = module_0.Token(var_7)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = module_1.Field()
    var_20 = {var_12: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = error.messages()[0]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = 'invalid'
    var_27 = {var_24: var_7, var_25: var_26}
    var_28 = 20
    var_29 = module_0.Token(var_7)
    var_30 = 15
    var_31 = module_0.Token(var_26)
    var_32 = [var_29, var_31]
    var_33 = module_0.Token(var_27)
    var_34 = module_1.Field()
    var_35 = module_1.Field()
    var_36 = {var_24: var_34, var_25: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_2.validate_with_positions(token=var_33, validator=var_37)
    var_39 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[var_1]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[var_1]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 20
    var_17 = module_0.Token(var_15)



# Parsed testcases at query #17
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
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = module_1.validate_with_positions(token=var_16, validator=var_0)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = module_0.Token(var_4)
    var_8 = {var_1: var_3}
    var_9 = module_0.Token(var_8)
    var_10 = module_1.validate_with_positions(token=var_9, validator=var_0)
    var_11 = -5
    var_12 = {var_0: var_2, var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = 'user'
    var_16 = ''
    var_17 = {var_0: var_16, var_14: var_3}
    var_18 = {var_15: var_17}
    var_19 = 20
    var_20 = module_0.Token(var_18)
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = -5
    var_23 = {var_0: var_16, var_21: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.validate_with_positions(token=var_24, validator=var_0)
    var_26 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 15
    var_7 = module_0.Token(var_4)
    var_8 = {var_1: var_3}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = 'thirty'
    var_12 = {var_0: var_2, var_1: var_11}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = 'user'
    var_16 = 123
    var_17 = {var_0: var_16}
    var_18 = {var_15: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = {var_0: var_16, var_1: var_11}
    var_21 = module_0.Token(var_20)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
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



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = module_0.Token(var_6)
    var_8 = 'age'
    var_9 = True
    var_10 = e.messages()[0]
    var_11 = 'invalid'
    var_12 = {var_8: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = e.messages()[0]
    var_15 = 'user'
    var_16 = {var_4: var_5}
    var_17 = {var_15: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = e.messages()[0]



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
    var_10 = list(e.messages())
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = list(e.messages())
    var_17 = 'wrong'
    var_18 = 5
    var_19 = module_0.Token(var_17)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = list(e.messages())



# Parsed testcases at query #24
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
    var_20 = {var_6: var_19}
    var_21 = 20
    var_22 = module_0.Token(var_20)
    var_23 = 'valid_value'
    var_24 = [var_23]
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_6: var_25, var_11: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_2.validate_with_positions(token=var_22, validator=var_28)
    var_30 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #25
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
    var_15 = 15
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
    var_29 = lambda x: x != var_24
    var_30 = [var_29]
    var_31 = module_1.Field()
    var_32 = {var_22: var_28, var_23: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = module_2.validate_with_positions(token=var_27, validator=var_33)
    var_35 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #26
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
    var_19 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #27
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
    var_9 = 'user'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = module_1.validate_with_positions(token=var_12, validator=var_0)
    var_14 = {}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)
    var_17 = {var_0: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)



# Parsed testcases at query #28
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
    var_9 = list(e.messages())



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
    var_7 = 'nested'
    var_8 = 'field'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 20
    var_13 = {var_8: var_9}
    var_14 = 5
    var_15 = 15
    var_16 = 10
    var_17 = module_0.Token(var_9)
    var_18 = [var_17]
    var_19 = module_0.Token(var_13)
    var_20 = [var_19]
    var_21 = module_0.Token(var_11)
    var_22 = True
    var_23 = module_1.Field()
    var_24 = {var_8: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_7: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_21, validator=var_27)
    var_29 = error.messages()[0]
    var_30 = 'invalid_value'
    var_31 = {var_8: var_30}
    var_32 = 7
    var_33 = module_0.Token(var_30)
    var_34 = [var_33]
    var_35 = module_0.Token(var_31)
    var_36 = False
    var_37 = 'valid_value'
    var_38 = lambda x: x == var_37
    var_39 = [var_38]
    var_40 = module_1.Field()
    var_41 = module_2.validate_with_positions(token=var_35, validator=var_40)
    var_42 = error.messages()[0]



# Parsed testcases at query #30
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
    var_9 = 'user'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {}
    var_14 = 5
    var_15 = 15
    var_16 = [var_9]
    var_17 = module_0.Token(var_13)
    var_18 = [var_17]
    var_19 = module_0.Token(var_11)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = {}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
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



# Parsed testcases at query #32
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
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = 5
    var_11 = module_0.Token(var_4)
    var_12 = [var_11]
    var_13 = module_0.Token(var_8)
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = 'invalid'
    var_17 = {var_14: var_4, var_15: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_4)
    var_20 = 12
    var_21 = 19
    var_22 = module_0.Token(var_16)
    var_23 = [var_19, var_22]
    var_24 = module_0.Token(var_17)



# Parsed testcases at query #33
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
    var_33 = 'invalid_value'
    var_34 = 12
    var_35 = module_0.Token(var_33)
    var_36 = False
    var_37 = 'valid_value'
    var_38 = lambda x: x == var_37
    var_39 = [var_38]
    var_40 = module_1.Field()
    var_41 = module_2.validate_with_positions(token=var_35, validator=var_40)
    var_42 = error.messages()[0]



# Parsed testcases at query #34
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
    var_7 = error.messages()[0]
    var_8 = 'invalid_value'
    var_9 = module_0.Token(var_8)
    var_10 = error.messages()[0]
    var_11 = 'nested'
    var_12 = {var_11: var_4}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = error.messages()[0]
    var_16 = 'field1'
    var_17 = 'field2'
    var_18 = 'invalid'
    var_19 = {var_16: var_4, var_17: var_18}
    var_20 = 30
    var_21 = module_0.Token(var_19)
    var_22 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #35
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
    var_4 = {}
    var_5 = 1
    var_6 = module_0.Token(var_4)
    var_7 = 'name'
    var_8 = True
    var_9 = error.messages()[0]
    var_10 = 'invalid_email'
    var_11 = 12
    var_12 = module_0.Token(var_10)
    var_13 = 'email'
    var_14 = module_1.Field()
    var_15 = module_2.validate_with_positions(token=var_12, validator=var_14)
    var_16 = error.messages()[0]
    var_17 = 'user'
    var_18 = 'invalid'
    var_19 = {var_13: var_18}
    var_20 = {var_17: var_19}
    var_21 = 20
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Field()
    var_24 = {var_13: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_17: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_22, validator=var_27)
    var_29 = error.messages()[0]



# Parsed testcases at query #36
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
    var_18 = e.messages()[0]
    var_19 = 'invalid_email'
    var_20 = 13
    var_21 = module_0.Token(var_19)
    var_22 = 'email'
    var_23 = module_1.Field()
    var_24 = module_2.validate_with_positions(token=var_21, validator=var_23)
    var_25 = e.messages()[0]
    var_26 = ''
    var_27 = 'invalid'
    var_28 = {var_6: var_26, var_11: var_27}
    var_29 = 20
    var_30 = 5
    var_31 = module_0.Token(var_26)
    var_32 = 6
    var_33 = module_0.Token(var_27)
    var_34 = [var_31, var_33]
    var_35 = module_0.Token(var_28)
    var_36 = module_1.Field()
    var_37 = 'integer'
    var_38 = module_1.Field()
    var_39 = {var_6: var_36, var_11: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_35, validator=var_40)
    var_42 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Token(var_0)
    var_2 = module_1.Field()
    var_3 = module_2.validate_with_positions(token=var_1, validator=var_2)
    assert var_3 == 'test_value'
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = module_1.Field()
    var_8 = module_2.validate_with_positions(token=var_5, validator=var_7)
    var_9 = error.messages()[0]
    var_10 = 'nested'
    var_11 = {var_10: var_4}
    var_12 = 0
    var_13 = 10
    var_14 = module_0.Token(var_11)
    var_15 = module_1.Field()
    var_16 = {var_10: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_14, validator=var_17)
    var_19 = error.messages()[0]
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = 'invalid'
    var_23 = {var_20: var_4, var_21: var_22}
    var_24 = 20
    var_25 = 5
    var_26 = module_0.Token(var_4)
    var_27 = 6
    var_28 = module_0.Token(var_22)
    var_29 = [var_26, var_28]
    var_30 = module_0.Token(var_23)
    var_31 = module_1.Field()
    var_32 = module_1.Field()
    var_33 = {var_20: var_31, var_21: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_30, validator=var_34)
    var_36 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #38
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
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 20
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



# Parsed testcases at query #39
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
    var_21 = 123
    var_22 = {var_0: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = e.messages()[0]



# Parsed testcases at query #40
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
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 20
    var_15 = module_0.Token(var_13)



# Parsed testcases at query #41
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
    var_19 = module_0.Token(var_18)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = e.messages()[0]



# Parsed testcases at query #42
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
    var_18 = 'name'
    var_19 = 'age'
    var_20 = 'John'
    var_21 = 'invalid'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = 20
    var_24 = 4
    var_25 = module_0.Token(var_20)
    var_26 = 12
    var_27 = module_0.Token(var_21)
    var_28 = [var_25, var_27]
    var_29 = module_0.Token(var_22)
    var_30 = module_1.Field()
    var_31 = 'integer'
    var_32 = module_1.Field()
    var_33 = {var_18: var_30, var_19: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_29, validator=var_34)
    var_36 = error.messages()[0]



# Parsed testcases at query #43
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
    var_12 = list(error.messages())
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 10
    var_16 = 5
    var_17 = module_0.Token(var_7)
    var_18 = {var_13: var_17}
    var_19 = module_0.Token(var_14)
    var_20 = module_1.Field()
    var_21 = {var_13: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_24 = list(error.messages())
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = {var_25: var_7, var_26: var_7}
    var_28 = 20
    var_29 = module_0.Token(var_7)
    var_30 = 15
    var_31 = module_0.Token(var_7)
    var_32 = {var_25: var_29, var_26: var_31}
    var_33 = module_0.Token(var_27)
    var_34 = module_1.Field()
    var_35 = module_1.Field()
    var_36 = {var_25: var_34, var_26: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_2.validate_with_positions(token=var_33, validator=var_37)
    var_39 = list(error.messages())



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
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



# Parsed testcases at query #45
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
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_7}
    var_13 = 15
    var_14 = module_0.Token(var_12)
    var_15 = module_1.Field()
    var_16 = {var_11: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_14, validator=var_17)
    var_19 = 'field1'
    var_20 = 'field2'
    var_21 = 'invalid'
    var_22 = {var_19: var_7, var_20: var_21}
    var_23 = 30
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = 10
    var_27 = module_1.Field()
    var_28 = {var_19: var_25, var_20: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_24, validator=var_29)



# Parsed testcases at query #46
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = 'name'
    var_7 = True
    var_8 = error.messages()[0]
    var_9 = 'invalid'
    var_10 = module_0.Token(var_9)
    var_11 = error.messages()[0]
    var_12 = 'user'
    var_13 = {var_6: var_4}
    var_14 = {var_12: var_13}
    var_15 = 20
    var_16 = {var_6: var_4}
    var_17 = 5
    var_18 = 15
    var_19 = module_0.Token(var_4)
    var_20 = [var_19]
    var_21 = module_0.Token(var_16)
    var_22 = [var_21]
    var_23 = module_0.Token(var_14)
    var_24 = error.messages()[0]



# Parsed testcases at query #47
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
    var_24 = {var_22: var_7, var_23: var_7}
    var_25 = 30
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)
    var_32 = 'invalid_value'
    var_33 = 15
    var_34 = module_0.Token(var_32)
    var_35 = False
    var_36 = 'valid_value'
    var_37 = lambda x: x == var_36
    var_38 = [var_37]
    var_39 = module_1.Field()
    var_40 = module_2.validate_with_positions(token=var_34, validator=var_39)
    var_41 = error.messages()[0]



# Parsed testcases at query #48
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
    var_14 = 'field'
    var_15 = 'invalid'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = 10
    var_21 = module_1.Field()
    var_22 = {var_14: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = {var_13: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_2.validate_with_positions(token=var_19, validator=var_25)
    var_27 = error.messages()[0]
    var_28 = 'field1'
    var_29 = 'field2'
    var_30 = 'short'
    var_31 = {var_28: var_7, var_29: var_30}
    var_32 = module_0.Token(var_31)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_28: var_33, var_29: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)
    var_38 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #49
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
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
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
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #50
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
    var_11 = 'nested'
    var_12 = {var_11: var_6}
    var_13 = 10
    var_14 = module_0.Token(var_6)
    var_15 = [var_14]
    var_16 = module_0.Token(var_12)
    var_17 = module_1.Field()
    var_18 = {var_11: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = module_2.validate_with_positions(token=var_16, validator=var_19)
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_6, var_22: var_6}
    var_24 = 20
    var_25 = module_0.Token(var_6)
    var_26 = 15
    var_27 = module_0.Token(var_6)
    var_28 = [var_25, var_27]
    var_29 = module_0.Token(var_23)
    var_30 = module_1.Field()
    var_31 = module_1.Field()
    var_32 = {var_21: var_30, var_22: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = module_2.validate_with_positions(token=var_29, validator=var_33)



# Parsed testcases at query #51
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)
    var_6 = {}
    var_7 = module_0.Token(var_6)
    var_8 = error.messages()[0]
    var_9 = 'user'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = module_0.Token(var_11)
    var_14 = error.messages()[0]
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #52
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
    var_12 = error.messages()[0]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 20
    var_16 = module_0.Token(var_7)
    var_17 = [var_16]
    var_18 = module_0.Token(var_14)
    var_19 = module_1.Field()
    var_20 = {var_13: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = error.messages()[0]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = {var_24: var_7, var_25: var_7}
    var_27 = 30
    var_28 = module_0.Token(var_7)
    var_29 = module_0.Token(var_7)
    var_30 = [var_28, var_29]
    var_31 = module_0.Token(var_26)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_24: var_32, var_25: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)
    var_37 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #53
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
    var_8 = 'invalid_value'
    var_9 = 12
    var_10 = module_0.Token(var_8)
    var_11 = error.messages()[0]
    var_12 = 'name'
    var_13 = 'age'
    var_14 = 'invalid'
    var_15 = {var_12: var_4, var_13: var_14}
    var_16 = 20
    var_17 = 4
    var_18 = module_0.Token(var_4)
    var_19 = 6
    var_20 = module_0.Token(var_14)
    var_21 = [var_18, var_20]
    var_22 = module_0.Token(var_15)



# Parsed testcases at query #54
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_6, var_22: var_6}
    var_24 = 25
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)



# Parsed testcases at query #55
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
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = {var_0: var_14}
    var_16 = 15
    var_17 = module_0.Token(var_15)
    var_18 = module_1.validate_with_positions(token=var_17, validator=var_0)
    var_19 = 'email'
    var_20 = 123
    var_21 = {var_19: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)



# Parsed testcases at query #56
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
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = 10
    var_16 = module_0.Token(var_7)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = module_1.Field()
    var_20 = {var_12: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = {var_23: var_7, var_24: var_7}
    var_26 = 30
    var_27 = 20
    var_28 = module_0.Token(var_7)
    var_29 = module_0.Token(var_7)
    var_30 = [var_28, var_29]
    var_31 = module_0.Token(var_25)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_23: var_32, var_24: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)



# Parsed testcases at query #57
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
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 25
    var_17 = module_0.Token(var_15)
    var_18 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #58
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 20
    var_15 = module_0.Token(var_13)



# Parsed testcases at query #59
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
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = error.messages()[0]
    var_17 = {}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = sorted(error.messages(), key=lambda m: m.index[-1])



# Parsed testcases at query #60
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
    var_7 = 5
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = 'data'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = module_1.validate_with_positions(token=var_16, validator=var_0)



# Parsed testcases at query #61
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 15
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = e.messages()[0]
    var_12 = 'thirty'
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = e.messages()[0]
    var_17 = 'user'
    var_18 = {var_0: var_2, var_1: var_12}
    var_19 = {var_17: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = e.messages()[0]
    var_22 = 123
    var_23 = {var_0: var_22, var_1: var_12}
    var_24 = module_0.Token(var_23)
    var_25 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #62
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
    var_6 = 'nested'
    var_7 = 'field'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 20
    var_12 = module_0.Token(var_10)
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_7: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = {var_6: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_12, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'invalid'
    var_22 = 7
    var_23 = module_0.Token(var_21)
    var_24 = module_2.validate_with_positions(token=var_23, validator=var_4)
    var_25 = error.messages()[0]
    var_26 = 'field1'
    var_27 = 'field2'
    var_28 = {var_26: var_21, var_27: var_8}
    var_29 = 30
    var_30 = module_0.Token(var_28)
    var_31 = module_1.Field()
    var_32 = module_2.validate_with_positions(token=var_30, validator=var_18)



# Parsed testcases at query #63
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
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_15)



# Parsed testcases at query #64
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = {}
    var_5 = 1
    var_6 = module_0.Token(var_4)
    var_7 = 'name'
    var_8 = True
    var_9 = error.messages()[0]
    var_10 = 'invalid_email'
    var_11 = 12
    var_12 = module_0.Token(var_10)
    var_13 = 'type'
    var_14 = 'email'
    var_15 = {var_13: var_14}
    var_16 = [var_15]
    var_17 = error.messages()[0]
    var_18 = 'user'
    var_19 = 123
    var_20 = {var_7: var_19}
    var_21 = {var_18: var_20}
    var_22 = 15
    var_23 = module_0.Token(var_21)
    var_24 = error.messages()[0]



# Parsed testcases at query #65
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = 'name'
    var_7 = True
    var_8 = error.messages()[0]
    var_9 = 'user'
    var_10 = {var_6: var_4}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_6: var_4}
    var_14 = 5
    var_15 = 15
    var_16 = module_0.Token(var_4)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = [var_18]
    var_20 = module_0.Token(var_11)
    var_21 = error.messages()[0]
    var_22 = 'age'
    var_23 = 'invalid'
    var_24 = {var_6: var_4, var_22: var_23}
    var_25 = module_0.Token(var_4)
    var_26 = module_0.Token(var_23)
    var_27 = [var_25, var_26]
    var_28 = module_0.Token(var_24)
    var_29 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #66
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
    var_20 = {var_6: var_7}
    var_21 = {var_19: var_20}
    var_22 = 25
    var_23 = module_0.Token(var_21)
    var_24 = 'email'
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_6: var_25, var_24: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {var_19: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_23, validator=var_30)
    var_32 = error.messages()[0]
    var_33 = 'invalid'
    var_34 = {var_6: var_7, var_11: var_33}
    var_35 = 30
    var_36 = module_0.Token(var_34)
    var_37 = module_1.Field()
    var_38 = module_2.validate_with_positions(token=var_36, validator=var_30)
    var_39 = error.messages()[0]



# Parsed testcases at query #67
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = ''
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = 'field'
    var_9 = {var_8: var_4}
    var_10 = {var_7: var_9}
    var_11 = 20
    var_12 = {var_8: var_4}
    var_13 = 10
    var_14 = 15
    var_15 = module_0.Token(var_4)
    var_16 = [var_15]
    var_17 = module_0.Token(var_12)
    var_18 = [var_17]
    var_19 = module_0.Token(var_10)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = {var_20: var_4, var_21: var_4}
    var_23 = module_0.Token(var_4)
    var_24 = module_0.Token(var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Token(var_22)



# Parsed testcases at query #68
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
    var_17 = 'email'
    var_18 = 'invalid'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 20
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Field()
    var_24 = {var_17: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_16: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_22, validator=var_27)
    var_29 = error.messages()[0]
    var_30 = 'age'
    var_31 = None
    var_32 = {var_9: var_31, var_30: var_18}
    var_33 = 30
    var_34 = module_0.Token(var_32)
    var_35 = True
    var_36 = module_1.Field()
    var_37 = 'integer'
    var_38 = module_1.Field()
    var_39 = {var_9: var_36, var_30: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_34, validator=var_40)
    var_42 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #69
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
    var_8 = 5
    var_9 = module_0.Token(var_7)
    var_10 = True
    var_11 = module_1.Field()
    var_12 = module_2.validate_with_positions(token=var_9, validator=var_11)
    var_13 = 'nested'
    var_14 = 'field'
    var_15 = {var_14: var_7}
    var_16 = {var_13: var_15}
    var_17 = 20
    var_18 = module_0.Token(var_16)
    var_19 = module_1.Field()
    var_20 = {var_14: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = {var_13: var_21}
    var_23 = module_3.Schema(var_22)
    var_24 = module_2.validate_with_positions(token=var_18, validator=var_23)
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = {var_25: var_7, var_26: var_7}
    var_28 = 30
    var_29 = module_0.Token(var_27)
    var_30 = module_1.Field()
    var_31 = module_1.Field()
    var_32 = {var_25: var_30, var_26: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = module_2.validate_with_positions(token=var_29, validator=var_33)



# Parsed testcases at query #70
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
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)
    var_17 = e.messages()[0]
    var_18 = 'age'
    var_19 = 123
    var_20 = 'abc'
    var_21 = {var_0: var_19, var_18: var_20}
    var_22 = 20
    var_23 = module_0.Token(var_21)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #71
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
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_7, var_22: var_7}
    var_24 = 25
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #72
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
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = 5
    var_11 = module_0.Token(var_4)
    var_12 = [var_11]
    var_13 = module_0.Token(var_8)
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = {var_14: var_4, var_15: var_4}
    var_17 = 20
    var_18 = module_0.Token(var_4)
    var_19 = 15
    var_20 = module_0.Token(var_4)
    var_21 = [var_18, var_20]
    var_22 = module_0.Token(var_16)



# Parsed testcases at query #73
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_15)



# Parsed testcases at query #74
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
    var_19 = 'nested'
    var_20 = 'field'
    var_21 = {var_20: var_9}
    var_22 = {var_19: var_21}
    var_23 = {var_20: var_9}
    var_24 = 15
    var_25 = 10
    var_26 = module_0.Token(var_9)
    var_27 = [var_26]
    var_28 = module_0.Token(var_23)
    var_29 = [var_28]
    var_30 = module_0.Token(var_22)
    var_31 = module_1.Field()
    var_32 = {var_20: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = {var_19: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_30, validator=var_35)
    var_37 = {var_6: var_9, var_7: var_9}
    var_38 = module_0.Token(var_9)
    var_39 = module_0.Token(var_9)
    var_40 = [var_38, var_39]
    var_41 = module_0.Token(var_37)
    var_42 = module_1.Field()
    var_43 = module_1.Field()
    var_44 = {var_6: var_42, var_7: var_43}
    var_45 = module_3.Schema(var_44)
    var_46 = module_2.validate_with_positions(token=var_41, validator=var_45)



# Parsed testcases at query #75
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
    var_7 = 5
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = e.messages()[0]
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)
    var_17 = e.messages()[0]
    var_18 = {}
    var_19 = module_0.Token(var_18)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = sorted(e.messages(), key=lambda m: m.start_position.char_index)
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = e.messages()[0]



# Parsed testcases at query #76
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
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 20
    var_15 = module_0.Token(var_13)



# Parsed testcases at query #77
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
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = 'user'
    var_16 = 'email'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = 20
    var_21 = module_0.Token(var_19)
    var_22 = module_1.Field()
    var_23 = {var_16: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {var_15: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_21, validator=var_26)
    var_28 = 'age'
    var_29 = ''
    var_30 = 'not_a_number'
    var_31 = {var_9: var_29, var_28: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = True
    var_35 = module_1.Field()
    var_36 = 'integer'
    var_37 = module_1.Field()
    var_38 = {var_9: var_35, var_28: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = module_2.validate_with_positions(token=var_33, validator=var_39)



# Parsed testcases at query #78
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
    var_14 = 15
    var_15 = 7
    var_16 = 14
    var_17 = module_0.Token(var_7)
    var_18 = [var_17]
    var_19 = module_0.Token(var_13)
    var_20 = module_1.Field()
    var_21 = {var_12: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_24 = error.messages()[0]
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = 'invalid'
    var_28 = {var_25: var_7, var_26: var_27}
    var_29 = 30
    var_30 = module_0.Token(var_7)
    var_31 = 18
    var_32 = 25
    var_33 = module_0.Token(var_27)
    var_34 = [var_30, var_33]
    var_35 = module_0.Token(var_28)
    var_36 = module_1.Field()
    var_37 = 10
    var_38 = module_1.Field()
    var_39 = {var_25: var_36, var_26: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_35, validator=var_40)
    var_42 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #79
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = e.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = e.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_15)
    var_18 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #80
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
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'required_field'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]
    var_16 = 'nested'
    var_17 = 'field'
    var_18 = 'invalid_value'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 20
    var_22 = {var_17: var_18}
    var_23 = 10
    var_24 = module_0.Token(var_18)
    var_25 = [var_24]
    var_26 = module_0.Token(var_22)
    var_27 = [var_26]
    var_28 = module_0.Token(var_20)
    var_29 = 'valid_value'
    var_30 = lambda x: x == var_29
    var_31 = module_1.Field()
    var_32 = {var_17: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = {var_16: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_28, validator=var_35)
    var_37 = error.messages()[0]
    var_38 = 'field1'
    var_39 = 'field2'
    var_40 = 'invalid'
    var_41 = None
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = module_0.Token(var_40)
    var_44 = 15
    var_45 = module_0.Token(var_41)
    var_46 = [var_43, var_45]
    var_47 = module_0.Token(var_42)
    var_48 = 'valid'
    var_49 = lambda x: x == var_48
    var_50 = module_1.Field()
    var_51 = True
    var_52 = module_1.Field()
    var_53 = {var_38: var_50, var_39: var_52}
    var_54 = module_3.Schema(var_53)
    var_55 = module_2.validate_with_positions(token=var_47, validator=var_54)



# Parsed testcases at query #81
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
    var_15 = 'user'
    var_16 = {}
    var_17 = {var_15: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = module_1.Field()
    var_21 = {var_8: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = {var_15: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_2.validate_with_positions(token=var_19, validator=var_24)
    var_26 = error.messages()[0]
    var_27 = 'age'
    var_28 = ''
    var_29 = -1
    var_30 = {var_8: var_28, var_27: var_29}
    var_31 = 30
    var_32 = module_0.Token(var_30)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_8: var_33, var_27: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)
    var_38 = [m for m in error.messages() if m.index == ['name']][0]
    var_39 = [m for m in error.messages() if m.index == ['age']][0]



# Parsed testcases at query #82
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
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 10
    var_15 = module_0.Token(var_7)
    var_16 = {var_12: var_15}
    var_17 = module_0.Token(var_13)
    var_18 = module_1.Field()
    var_19 = {var_12: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = module_2.validate_with_positions(token=var_17, validator=var_20)
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = 'invalid'
    var_25 = {var_22: var_7, var_23: var_24}
    var_26 = 20
    var_27 = module_0.Token(var_7)
    var_28 = 12
    var_29 = module_0.Token(var_24)
    var_30 = {var_22: var_27, var_23: var_29}
    var_31 = module_0.Token(var_25)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_22: var_32, var_23: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)



# Parsed testcases at query #83
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
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = list(error.messages())
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = list(error.messages())
    var_16 = {var_0: var_14}
    var_17 = module_0.Token(var_16)
    var_18 = module_1.validate_with_positions(token=var_17, validator=var_0)
    var_19 = list(error.messages())



# Parsed testcases at query #84
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
    var_14 = 'field'
    var_15 = {var_14: var_7}
    var_16 = {var_13: var_15}
    var_17 = 20
    var_18 = {var_14: var_7}
    var_19 = 5
    var_20 = 15
    var_21 = module_0.Token(var_7)
    var_22 = [var_21]
    var_23 = module_0.Token(var_18)
    var_24 = [var_23]
    var_25 = module_0.Token(var_16)
    var_26 = module_1.Field()
    var_27 = {var_14: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {var_13: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_25, validator=var_30)
    var_32 = error.messages()[0]
    var_33 = 'field1'
    var_34 = 'field2'
    var_35 = 'invalid'
    var_36 = {var_33: var_7, var_34: var_35}
    var_37 = 30
    var_38 = module_0.Token(var_7)
    var_39 = 25
    var_40 = module_0.Token(var_35)
    var_41 = [var_38, var_40]
    var_42 = module_0.Token(var_36)
    var_43 = module_1.Field()
    var_44 = module_1.Field()
    var_45 = {var_33: var_43, var_34: var_44}
    var_46 = module_3.Schema(var_45)
    var_47 = module_2.validate_with_positions(token=var_42, validator=var_46)
    var_48 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #85
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
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = 10
    var_16 = module_0.Token(var_7)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = module_1.Field()
    var_20 = {var_12: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = {var_23: var_7, var_24: var_7}
    var_26 = 30
    var_27 = 20
    var_28 = module_0.Token(var_7)
    var_29 = module_0.Token(var_7)
    var_30 = [var_28, var_29]
    var_31 = module_0.Token(var_25)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_23: var_32, var_24: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)



# Parsed testcases at query #86
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
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
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
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #87
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
    var_7 = 5
    var_8 = module_0.Token(var_6)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = error.messages()[0]



# Parsed testcases at query #88
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



# Parsed testcases at query #89
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

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
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = 'invalid'
    var_15 = {var_12: var_7, var_13: var_14}
    var_16 = 20
    var_17 = module_0.Token(var_15)
    var_18 = module_1.Field()



# Parsed testcases at query #90
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
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
    var_17 = 123
    var_18 = {var_4: var_17, var_9: var_12}
    var_19 = 30
    var_20 = module_0.Token(var_18)
    var_21 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #91
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 30
    var_15 = module_0.Token(var_13)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = {}
    var_5 = 1
    var_6 = module_0.Token(var_4)
    var_7 = 'name'
    var_8 = True
    var_9 = 'age'
    var_10 = 'invalid'
    var_11 = {var_9: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = 123
    var_14 = {var_7: var_13, var_9: var_10}
    var_15 = 20
    var_16 = module_0.Token(var_14)



# Parsed testcases at query #2
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
    var_18 = error.messages()[0]
    var_19 = 'invalid_value'
    var_20 = module_0.Token(var_19)
    var_21 = 5
    var_22 = module_1.Field()
    var_23 = module_2.validate_with_positions(token=var_20, validator=var_22)
    var_24 = error.messages()[0]
    var_25 = 'nested'
    var_26 = 'field'
    var_27 = 'invalid'
    var_28 = {var_26: var_27}
    var_29 = {var_25: var_28}
    var_30 = 30
    var_31 = {var_26: var_27}
    var_32 = module_0.Token(var_27)
    var_33 = [var_32]
    var_34 = module_0.Token(var_31)
    var_35 = [var_34]
    var_36 = module_0.Token(var_29)
    var_37 = module_1.Field()
    var_38 = {var_26: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = {var_25: var_39}
    var_41 = module_3.Schema(var_40)
    var_42 = module_2.validate_with_positions(token=var_36, validator=var_41)
    var_43 = error.messages()[0]



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
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
    var_13 = 12
    var_14 = module_0.Token(var_12)
    var_15 = e.messages()[0]
    var_16 = 'user'
    var_17 = 123
    var_18 = {var_4: var_17}
    var_19 = {var_16: var_18}
    var_20 = 20
    var_21 = module_0.Token(var_19)
    var_22 = e.messages()[0]



# Parsed testcases at query #4
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
    var_11 = 'invalid_value'
    var_12 = 12
    var_13 = module_0.Token(var_11)
    var_14 = lambda x: x == var_10
    var_15 = [var_14]
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_13, validator=var_16)
    var_18 = 'nested'
    var_19 = 'field'
    var_20 = {var_19: var_6}
    var_21 = {var_18: var_20}
    var_22 = 20
    var_23 = module_0.Token(var_21)
    var_24 = module_1.Field()
    var_25 = {var_19: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = {var_18: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_2.validate_with_positions(token=var_23, validator=var_28)



# Parsed testcases at query #5
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
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = error.messages()[0]
    var_17 = {}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #6
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
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = 'age'
    var_11 = module_1.Field()
    var_12 = True
    var_13 = module_1.Field()
    var_14 = {var_6: var_11, var_10: var_13}
    var_15 = module_3.Schema(var_14)
    var_16 = module_2.validate_with_positions(token=var_9, validator=var_15)
    var_17 = 'invalid_email'
    var_18 = module_0.Token(var_17)
    var_19 = 'email'
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_18, validator=var_20)
    var_22 = 'user'
    var_23 = {var_6: var_7}
    var_24 = {var_22: var_23}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_6: var_27, var_10: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = {var_22: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = module_2.validate_with_positions(token=var_26, validator=var_32)



# Parsed testcases at query #7
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
    var_9 = 'user'
    var_10 = {var_0: var_8}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_0: var_8}
    var_14 = 5
    var_15 = 15
    var_16 = module_0.Token(var_13)
    var_17 = [var_16]
    var_18 = module_0.Token(var_11)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = {var_0: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = module_1.validate_with_positions(token=var_21, validator=var_0)
    var_23 = 'age'
    var_24 = 'invalid'
    var_25 = {var_0: var_22, var_23: var_24}
    var_26 = module_0.Token(var_25)
    var_27 = module_1.validate_with_positions(token=var_26, validator=var_0)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = {var_4: var_0}
    var_6 = 15
    var_7 = module_0.Token(var_5)
    var_8 = 'age'
    var_9 = 'user'
    var_10 = {var_4: var_0}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_4: var_0}
    var_14 = 5
    var_15 = 10
    var_16 = module_0.Token(var_0)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = [var_18]
    var_20 = module_0.Token(var_11)
    var_21 = 123
    var_22 = {var_4: var_21, var_8: var_0}
    var_23 = module_0.Token(var_21)
    var_24 = module_0.Token(var_0)
    var_25 = [var_23, var_24]
    var_26 = module_0.Token(var_22)



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
    var_6 = {}
    var_7 = 1
    var_8 = module_0.Token(var_6)
    var_9 = 'name'
    var_10 = True
    var_11 = module_1.Field()
    var_12 = {var_9: var_11}
    var_13 = module_3.Schema(var_12)
    var_14 = module_2.validate_with_positions(token=var_8, validator=var_13)
    var_15 = 'user'
    var_16 = 'email'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = 20
    var_21 = {var_16: var_17}
    var_22 = 7
    var_23 = 15
    var_24 = module_0.Token(var_17)
    var_25 = [var_24]
    var_26 = module_0.Token(var_21)
    var_27 = [var_26]
    var_28 = module_0.Token(var_19)
    var_29 = [var_16]
    var_30 = module_1.Field()
    var_31 = {var_16: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = {var_15: var_32}
    var_34 = module_3.Schema(var_33)
    var_35 = module_2.validate_with_positions(token=var_28, validator=var_34)
    var_36 = 'age'
    var_37 = ''
    var_38 = {var_9: var_37, var_36: var_17}
    var_39 = 8
    var_40 = module_0.Token(var_37)
    var_41 = module_0.Token(var_17)
    var_42 = [var_40, var_41]
    var_43 = module_0.Token(var_38)
    var_44 = 'min_length'
    var_45 = [var_44]
    var_46 = module_1.Field()
    var_47 = 'integer'
    var_48 = [var_47]
    var_49 = module_1.Field()
    var_50 = {var_9: var_46, var_36: var_49}
    var_51 = module_3.Schema(var_50)
    var_52 = module_2.validate_with_positions(token=var_43, validator=var_51)



# Parsed testcases at query #10
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
    var_16 = 'invalid'
    var_17 = {var_15: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_12)
    var_20 = error.messages()[0]
    var_21 = None
    var_22 = {var_8: var_21, var_15: var_16}
    var_23 = 20
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = module_2.validate_with_positions(token=var_24, validator=var_12)
    var_27 = [m for m in error.messages() if m.code == 'required'][0]
    var_28 = [m for m in error.messages() if m.code == 'invalid_type'][0]



# Parsed testcases at query #11
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
    var_9 = 'user'
    var_10 = {var_0: var_8}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_0: var_8}
    var_14 = 5
    var_15 = 15
    var_16 = module_0.Token(var_13)
    var_17 = [var_16]
    var_18 = module_0.Token(var_11)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = {}
    var_21 = module_0.Token(var_20)
    var_22 = module_1.validate_with_positions(token=var_21, validator=var_0)



# Parsed testcases at query #12
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
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_7, var_22: var_7}
    var_24 = 25
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0)
    var_4 = 'not_an_int'
    var_5 = 10
    var_6 = module_0.Token(var_4)
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 20
    var_10 = module_0.Token(var_4)
    var_11 = [var_10]
    var_12 = module_0.Token(var_8)
    var_13 = 'required_field'
    var_14 = True
    var_15 = {}
    var_16 = []
    var_17 = module_0.Token(var_15)



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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
    var_17 = None
    var_18 = {var_9: var_17}
    var_19 = {var_16: var_18}
    var_20 = 15
    var_21 = module_0.Token(var_19)
    var_22 = True
    var_23 = module_1.Field()
    var_24 = {var_9: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = {var_16: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_21, validator=var_27)
    var_29 = error.messages()[0]
    var_30 = 'age'
    var_31 = 'invalid'
    var_32 = {var_9: var_17, var_30: var_31}
    var_33 = 20
    var_34 = module_0.Token(var_32)
    var_35 = True
    var_36 = module_1.Field()
    var_37 = module_2.validate_with_positions(token=var_34, validator=var_27)
    var_38 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #16
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
    var_12 = 'nested'
    var_13 = 'field'
    var_14 = 'invalid_value'
    var_15 = {var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = 20
    var_18 = {var_13: var_14}
    var_19 = 18
    var_20 = 15
    var_21 = module_0.Token(var_14)
    var_22 = [var_21]
    var_23 = module_0.Token(var_18)
    var_24 = [var_23]
    var_25 = module_0.Token(var_16)
    var_26 = module_1.Field()
    var_27 = {var_13: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = {var_12: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_25, validator=var_30)
    var_32 = 'field1'
    var_33 = 'field2'
    var_34 = 'invalid'
    var_35 = {var_32: var_7, var_33: var_34}
    var_36 = module_0.Token(var_7)
    var_37 = 7
    var_38 = module_0.Token(var_34)
    var_39 = [var_36, var_38]
    var_40 = module_0.Token(var_35)
    var_41 = module_1.Field()
    var_42 = 'valid'
    var_43 = lambda x: x == var_42
    var_44 = [var_43]
    var_45 = module_1.Field()
    var_46 = {var_32: var_41, var_33: var_45}
    var_47 = module_3.Schema(var_46)
    var_48 = module_2.validate_with_positions(token=var_40, validator=var_47)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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
    var_10 = True
    var_11 = 'user'
    var_12 = {var_0: var_8}
    var_13 = {var_11: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = module_1.validate_with_positions(token=var_15, validator=var_0)
    var_17 = e.messages()[0]
    var_18 = {}
    var_19 = module_0.Token(var_18)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #19
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
    var_23 = {var_21: var_6, var_22: var_6}
    var_24 = 30
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #20
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
    var_25 = 'short'
    var_26 = {var_6: var_25, var_11: var_25}
    var_27 = module_0.Token(var_26)
    var_28 = module_1.Field()
    var_29 = module_1.Field()
    var_30 = {var_6: var_28, var_11: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_27, validator=var_31)



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
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = module_1.validate_with_positions(token=var_10, validator=var_0)



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
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'nested'
    var_7 = 'field'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 20
    var_12 = module_0.Token(var_10)
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_7: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = {var_6: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_12, validator=var_18)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = 'invalid'
    var_23 = {var_20: var_8, var_21: var_22}
    var_24 = 30
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_20: var_26, var_21: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)



# Parsed testcases at query #23
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
    var_18 = e.messages()[0]
    var_19 = 'invalid_value'
    var_20 = 5
    var_21 = 15
    var_22 = module_0.Token(var_19)
    var_23 = 'valid_value'
    var_24 = [var_23]
    var_25 = module_1.Field()
    var_26 = module_2.validate_with_positions(token=var_22, validator=var_25)
    var_27 = e.messages()[0]
    var_28 = 'invalid1'
    var_29 = 'invalid2'
    var_30 = {var_6: var_28, var_11: var_29}
    var_31 = 30
    var_32 = module_0.Token(var_30)
    var_33 = 'valid1'
    var_34 = [var_33]
    var_35 = module_1.Field()
    var_36 = 'valid2'
    var_37 = [var_36]
    var_38 = module_1.Field()
    var_39 = {var_6: var_35, var_11: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_2.validate_with_positions(token=var_32, validator=var_40)
    var_42 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



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
    var_16 = 'user'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = True
    var_22 = module_1.Field()
    var_23 = {var_9: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {var_16: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_20, validator=var_26)
    var_28 = error.messages()[0]
    var_29 = 'age'
    var_30 = ''
    var_31 = -1
    var_32 = {var_9: var_30, var_29: var_31}
    var_33 = 20
    var_34 = module_0.Token(var_32)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_9: var_35, var_29: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_34, validator=var_38)
    var_40 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3
import typesystem.base as module_4

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
    var_9 = 12
    var_10 = module_0.Token(var_8)
    var_11 = 'age'
    var_12 = module_1.Field()
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_6: var_12, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = error.messages()[0]
    var_19 = 'invalid_email'
    var_20 = module_0.Token(var_19)
    var_21 = 'Invalid email'
    var_22 = 'invalid'
    var_23 = module_4.Message(text=var_21, code=var_22)
    var_24 = [var_23]
    var_25 = module_2.validate_with_positions(token=var_20, validator=var_4)
    var_26 = error.messages()[0]
    var_27 = 'user'
    var_28 = {var_6: var_7}
    var_29 = {var_27: var_28}
    var_30 = 20
    var_31 = module_0.Token(var_29)
    var_32 = 'email'
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_6: var_33, var_32: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = {var_27: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_31, validator=var_38)
    var_40 = error.messages()[0]



# Parsed testcases at query #26
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
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = e.messages()[0]
    var_12 = 'invalid'
    var_13 = 5
    var_14 = 15
    var_15 = module_0.Token(var_12)
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_15, validator=var_16)
    var_18 = e.messages()[0]
    var_19 = 'nested'
    var_20 = {var_19: var_6}
    var_21 = 20
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Field()
    var_24 = {var_19: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_2.validate_with_positions(token=var_22, validator=var_25)
    var_27 = e.messages()[0]
    var_28 = 'field1'
    var_29 = 'field2'
    var_30 = 'short'
    var_31 = {var_28: var_6, var_29: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = module_1.Field()
    var_35 = module_1.Field()
    var_36 = {var_28: var_34, var_29: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_2.validate_with_positions(token=var_33, validator=var_37)
    var_39 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = {}
    var_5 = module_0.Token(var_4)
    var_6 = 'name'
    var_7 = True
    var_8 = list(error.messages())
    var_9 = 'user'
    var_10 = 123
    var_11 = {var_6: var_10}
    var_12 = {var_9: var_11}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = list(error.messages())
    var_16 = 'age'
    var_17 = 'invalid'
    var_18 = {var_6: var_10, var_16: var_17}
    var_19 = 30
    var_20 = module_0.Token(var_18)
    var_21 = list(error.messages())



# Parsed testcases at query #28
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = 'field'
    var_14 = {var_13: var_6}
    var_15 = {var_12: var_14}
    var_16 = 20
    var_17 = module_0.Token(var_15)
    var_18 = module_1.Field()
    var_19 = {var_13: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = {var_12: var_20}
    var_22 = module_3.Schema(var_21)
    var_23 = module_2.validate_with_positions(token=var_17, validator=var_22)
    var_24 = error.messages()[0]
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = 'invalid'
    var_28 = {var_25: var_6, var_26: var_27}
    var_29 = 30
    var_30 = 10
    var_31 = module_0.Token(var_6)
    var_32 = 15
    var_33 = module_0.Token(var_27)
    var_34 = [var_31, var_33]
    var_35 = module_0.Token(var_28)
    var_36 = module_1.Field()
    var_37 = module_1.Field()
    var_38 = {var_25: var_36, var_26: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = module_2.validate_with_positions(token=var_35, validator=var_39)



# Parsed testcases at query #29
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
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_7}
    var_13 = 15
    var_14 = 10
    var_15 = module_0.Token(var_7)
    var_16 = [var_15]
    var_17 = module_0.Token(var_12)
    var_18 = module_1.Field()
    var_19 = {var_11: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = module_2.validate_with_positions(token=var_17, validator=var_20)
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = 'invalid'
    var_25 = {var_22: var_7, var_23: var_24}
    var_26 = 30
    var_27 = module_0.Token(var_7)
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



# Parsed testcases at query #30
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
    var_11 = error.messages()[0]
    var_12 = 'user'
    var_13 = 'email'
    var_14 = 'invalid'
    var_15 = {var_4: var_5, var_13: var_14}
    var_16 = {var_12: var_15}
    var_17 = 30
    var_18 = module_0.Token(var_16)
    var_19 = error.messages()[0]
    var_20 = ''
    var_21 = 'not_a_number'
    var_22 = {var_4: var_20, var_9: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #31
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
    var_11 = 'invalid_value'
    var_12 = 12
    var_13 = module_0.Token(var_11)
    var_14 = 5
    var_15 = 10
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_13, validator=var_16)
    var_18 = 'name'
    var_19 = 'age'
    var_20 = 'test'
    var_21 = -5
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = 20
    var_24 = module_0.Token(var_20)
    var_25 = -5
    var_26 = 6
    var_27 = 8
    var_28 = module_0.Token(var_25)
    var_29 = [var_24, var_28]
    var_30 = module_0.Token(var_22)
    var_31 = 3
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_18: var_32, var_19: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_30, validator=var_35)



# Parsed testcases at query #32
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 15
    var_15 = 10
    var_16 = module_0.Token(var_6)
    var_17 = [var_16]
    var_18 = module_0.Token(var_13)
    var_19 = module_1.Field()
    var_20 = {var_12: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = error.messages()[0]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = 'invalid'
    var_27 = {var_24: var_6, var_25: var_26}
    var_28 = 30
    var_29 = module_0.Token(var_6)
    var_30 = 25
    var_31 = module_0.Token(var_26)
    var_32 = [var_29, var_31]
    var_33 = module_0.Token(var_27)
    var_34 = module_1.Field()
    var_35 = module_1.Field()
    var_36 = {var_24: var_34, var_25: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_2.validate_with_positions(token=var_33, validator=var_37)



# Parsed testcases at query #33
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
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 10
    var_11 = 5
    var_12 = module_0.Token(var_4)
    var_13 = [var_12]
    var_14 = module_0.Token(var_9)
    var_15 = error.messages()[0]
    var_16 = 'field1'
    var_17 = 'field2'
    var_18 = 'invalid'
    var_19 = {var_16: var_4, var_17: var_18}
    var_20 = 20
    var_21 = module_0.Token(var_4)
    var_22 = 12
    var_23 = module_0.Token(var_18)
    var_24 = [var_21, var_23]
    var_25 = module_0.Token(var_19)
    var_26 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #34
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
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)
    var_32 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #35
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
    var_6 = 'name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 15
    var_10 = module_0.Token(var_8)
    var_11 = 'age'
    var_12 = True
    var_13 = module_1.Field()
    var_14 = module_1.Field()
    var_15 = {var_6: var_13, var_11: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_10, validator=var_16)
    var_18 = 'user'
    var_19 = {var_6: var_7}
    var_20 = {var_18: var_19}
    var_21 = 20
    var_22 = {var_6: var_7}
    var_23 = 10
    var_24 = module_0.Token(var_7)
    var_25 = [var_24]
    var_26 = module_0.Token(var_22)
    var_27 = [var_26]
    var_28 = module_0.Token(var_20)
    var_29 = 'email'
    var_30 = module_1.Field()
    var_31 = module_1.Field()
    var_32 = {var_6: var_30, var_29: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = {var_18: var_33}
    var_35 = module_3.Schema(var_34)
    var_36 = module_2.validate_with_positions(token=var_28, validator=var_35)
    var_37 = 'invalid'
    var_38 = {var_6: var_7, var_11: var_37}
    var_39 = 25
    var_40 = module_0.Token(var_38)
    var_41 = module_1.Field()
    var_42 = module_2.validate_with_positions(token=var_40, validator=var_35)



# Parsed testcases at query #36
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_7, var_22: var_7}
    var_24 = 20
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #37
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
    var_18 = error.messages()[0]
    var_19 = 'invalid_value'
    var_20 = {var_6: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = 'valid_value'
    var_23 = lambda x: x == var_22
    var_24 = [var_23]
    var_25 = module_1.Field()
    var_26 = module_2.validate_with_positions(token=var_21, validator=var_25)
    var_27 = error.messages()[0]
    var_28 = {var_6: var_19, var_11: var_19}
    var_29 = module_0.Token(var_28)
    var_30 = lambda x: x == var_22
    var_31 = [var_30]
    var_32 = module_1.Field()
    var_33 = lambda x: x == var_22
    var_34 = [var_33]
    var_35 = module_1.Field()
    var_36 = {var_6: var_32, var_11: var_35}
    var_37 = module_3.Schema(var_36)
    var_38 = module_2.validate_with_positions(token=var_29, validator=var_37)
    var_39 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #38
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
    var_10 = error.messages()[0]
    var_11 = 'user'
    var_12 = {}
    var_13 = {var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = error.messages()[0]
    var_17 = {}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested_field'
    var_8 = {var_7: var_4}
    var_9 = 20
    var_10 = 5
    var_11 = 15
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
    var_22 = 28
    var_23 = module_0.Token(var_17)
    var_24 = [var_20, var_23]
    var_25 = module_0.Token(var_18)



# Parsed testcases at query #40
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
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = error.messages()[0]
    var_12 = 'invalid'
    var_13 = 7
    var_14 = module_0.Token(var_12)
    var_15 = 'valid'
    var_16 = lambda x: x == var_15
    var_17 = [var_16]
    var_18 = module_1.Field()
    var_19 = module_2.validate_with_positions(token=var_14, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'nested'
    var_22 = {var_21: var_7}
    var_23 = 15
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = {var_21: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_24, validator=var_27)
    var_29 = error.messages()[0]



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = {}
    var_5 = 1
    var_6 = module_0.Token(var_4)
    var_7 = 'name'
    var_8 = True
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = 'age'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = 20
    var_16 = module_0.Token(var_14)
    var_17 = error.messages()[0]
    var_18 = 123
    var_19 = {var_7: var_18, var_11: var_12}
    var_20 = 25
    var_21 = module_0.Token(var_19)
    var_22 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #42
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
    var_12 = e.messages()[0]
    var_13 = 'nested'
    var_14 = 'field'
    var_15 = 'invalid'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = 'valid'
    var_21 = lambda x: x == var_20
    var_22 = [var_21]
    var_23 = module_1.Field()
    var_24 = {var_14: var_23}
    var_25 = {var_13: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_19, validator=var_26)
    var_28 = e.messages()[0]
    var_29 = 'field1'
    var_30 = 'field2'
    var_31 = {var_29: var_7, var_30: var_15}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = module_1.Field()
    var_35 = lambda x: x == var_20
    var_36 = [var_35]
    var_37 = module_1.Field()
    var_38 = {var_29: var_34, var_30: var_37}
    var_39 = module_3.Schema(var_38)
    var_40 = module_2.validate_with_positions(token=var_33, validator=var_39)



# Parsed testcases at query #43
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
    var_14 = e.messages()[0]
    var_15 = 'user'
    var_16 = 'email'
    var_17 = 'invalid'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = {var_16: var_17}
    var_21 = 5
    var_22 = module_0.Token(var_17)
    var_23 = [var_22]
    var_24 = module_0.Token(var_20)
    var_25 = [var_24]
    var_26 = module_0.Token(var_19)
    var_27 = '^[^@]+@[^@]+\\.[^@]+$'
    var_28 = module_1.Field()
    var_29 = {var_16: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = {var_15: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = module_2.validate_with_positions(token=var_26, validator=var_32)
    var_34 = e.messages()[0]
    var_35 = 'age'
    var_36 = ''
    var_37 = -5
    var_38 = {var_8: var_36, var_35: var_37}
    var_39 = module_0.Token(var_36)
    var_40 = -5
    var_41 = 6
    var_42 = module_0.Token(var_40)
    var_43 = [var_39, var_42]
    var_44 = module_0.Token(var_38)
    var_45 = module_1.Field()
    var_46 = module_1.Field()
    var_47 = {var_8: var_45, var_35: var_46}
    var_48 = module_3.Schema(var_47)
    var_49 = module_2.validate_with_positions(token=var_44, validator=var_48)
    var_50 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[var_1]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[var_1]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_15)
    var_18 = lambda m: m.start_position.char_index



# Parsed testcases at query #45
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
    var_7 = module_0.Token(var_6)
    var_8 = module_1.validate_with_positions(token=var_7, validator=var_0)
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 20
    var_14 = {}
    var_15 = 5
    var_16 = 15
    var_17 = module_0.Token(var_14)
    var_18 = {var_10: var_17}
    var_19 = module_0.Token(var_12)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = error.messages()[0]
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_26 = 123
    var_27 = {var_0: var_26}
    var_28 = module_0.Token(var_27)
    var_29 = module_1.validate_with_positions(token=var_28, validator=var_0)
    var_30 = error.messages()[0]



# Parsed testcases at query #46
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 15
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = error.messages()[0]
    var_12 = 'thirty'
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = error.messages()[0]
    var_17 = 'user'
    var_18 = {var_0: var_2, var_1: var_12}
    var_19 = {var_17: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = error.messages()[0]
    var_22 = 123
    var_23 = {var_0: var_22, var_1: var_12}
    var_24 = module_0.Token(var_23)
    var_25 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #47
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
    var_18 = e.messages()[0]
    var_19 = 'invalid_value'
    var_20 = 5
    var_21 = 15
    var_22 = module_0.Token(var_19)
    var_23 = 'valid_value'
    var_24 = lambda x: x == var_23
    var_25 = [var_24]
    var_26 = module_1.Field()
    var_27 = module_2.validate_with_positions(token=var_22, validator=var_26)
    var_28 = e.messages()[0]
    var_29 = 'invalid1'
    var_30 = 'invalid2'
    var_31 = {var_6: var_29, var_11: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = 'valid1'
    var_35 = lambda x: x == var_34
    var_36 = [var_35]
    var_37 = module_1.Field()
    var_38 = 'valid2'
    var_39 = lambda x: x == var_38
    var_40 = [var_39]
    var_41 = module_1.Field()
    var_42 = {var_6: var_37, var_11: var_41}
    var_43 = module_3.Schema(var_42)
    var_44 = module_2.validate_with_positions(token=var_33, validator=var_43)
    var_45 = sorted(e.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #48
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'invalid'
    var_7 = {var_4: var_0, var_5: var_6}
    var_8 = 20
    var_9 = module_0.Token(var_7)
    var_10 = list(error.messages())
    var_11 = {var_4: var_0}
    var_12 = 10
    var_13 = module_0.Token(var_11)
    var_14 = True
    var_15 = list(error.messages())



# Parsed testcases at query #49
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = ''
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = 'nested'
    var_8 = {var_7: var_4}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 20
    var_15 = module_0.Token(var_13)



# Parsed testcases at query #50
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
    var_18 = 'user'
    var_19 = {var_6: var_7}
    var_20 = {var_18: var_19}
    var_21 = 20
    var_22 = module_0.Token(var_20)
    var_23 = 'email'
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_6: var_24, var_23: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = {var_18: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_22, validator=var_29)
    var_31 = 'invalid'
    var_32 = {var_6: var_7, var_11: var_31}
    var_33 = 25
    var_34 = module_0.Token(var_32)
    var_35 = module_1.Field()
    var_36 = module_2.validate_with_positions(token=var_34, validator=var_29)



# Parsed testcases at query #51
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
    var_11 = 'invalid_value'
    var_12 = 12
    var_13 = module_0.Token(var_11)
    var_14 = 5
    var_15 = 10
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_13, validator=var_16)
    var_18 = 'nested'
    var_19 = {var_18: var_6}
    var_20 = 15
    var_21 = module_0.Token(var_19)
    var_22 = module_1.Field()
    var_23 = {var_18: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = module_2.validate_with_positions(token=var_21, validator=var_24)



# Parsed testcases at query #52
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
    var_18 = 'invalid_value'
    var_19 = 12
    var_20 = module_0.Token(var_18)
    var_21 = module_1.Field()
    var_22 = module_2.validate_with_positions(token=var_20, validator=var_21)
    var_23 = 'nested'
    var_24 = 'field'
    var_25 = 'invalid'
    var_26 = {var_24: var_25}
    var_27 = {var_23: var_26}
    var_28 = 25
    var_29 = 10
    var_30 = 17
    var_31 = {var_24: var_25}
    var_32 = 7
    var_33 = 20
    var_34 = module_0.Token(var_31)
    var_35 = module_0.Token(var_25)
    var_36 = [var_35]
    var_37 = module_0.Token(var_27)
    var_38 = module_1.Field()
    var_39 = {var_24: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = {var_23: var_40}
    var_42 = module_3.Schema(var_41)
    var_43 = module_2.validate_with_positions(token=var_37, validator=var_42)



# Parsed testcases at query #53
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
    var_23 = 25
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_20: var_25, var_21: var_26}
    var_28 = module_3.Schema(var_27)
    var_29 = module_2.validate_with_positions(token=var_24, validator=var_28)



# Parsed testcases at query #54
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'John'
    var_7 = 'invalid'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 20
    var_10 = 7
    var_11 = 11
    var_12 = module_0.Token(var_6)
    var_13 = 17
    var_14 = 24
    var_15 = module_0.Token(var_7)
    var_16 = [var_12, var_15]
    var_17 = module_0.Token(var_8)
    var_18 = {var_4: var_6}
    var_19 = 15
    var_20 = module_0.Token(var_6)
    var_21 = [var_20]
    var_22 = module_0.Token(var_18)
    var_23 = True



# Parsed testcases at query #55
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'invalid'
    var_16 = {var_13: var_4, var_14: var_15}
    var_17 = 30
    var_18 = module_0.Token(var_16)
    var_19 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #56
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 15
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = 10
    var_10 = module_0.Token(var_8)
    var_11 = error.messages()[0]
    var_12 = 'thirty'
    var_13 = {var_0: var_2, var_1: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = error.messages()[0]
    var_17 = 'user'
    var_18 = {var_0: var_2, var_1: var_12}
    var_19 = {var_17: var_18}
    var_20 = 25
    var_21 = module_0.Token(var_19)
    var_22 = error.messages()[0]



# Parsed testcases at query #57
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3
import typesystem.base as module_4

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'nested'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = 'required_field'
    var_11 = True
    var_12 = module_1.Field()
    var_13 = {var_10: var_12}
    var_14 = module_3.Schema(var_13)
    var_15 = {var_6: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_2.validate_with_positions(token=var_9, validator=var_16)
    var_18 = 'invalid'
    var_19 = 7
    var_20 = module_0.Token(var_18)
    var_21 = 'Custom error'
    var_22 = module_4.Message(text=var_21, code=var_18)
    var_23 = [var_22]
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_4)
    var_25 = 'field1'
    var_26 = 'field2'
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = 20
    var_31 = module_0.Token(var_29)
    var_32 = 'Error1'
    var_33 = 'err1'
    var_34 = module_4.Message(text=var_32, code=var_33)
    var_35 = [var_34]
    var_36 = 'Error2'
    var_37 = 'err2'
    var_38 = module_4.Message(text=var_36, code=var_37)
    var_39 = [var_38]
    var_40 = module_2.validate_with_positions(token=var_31, validator=var_16)



# Parsed testcases at query #58
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = module_0.Token(var_4)
    var_8 = {var_0: var_2}
    var_9 = module_0.Token(var_8)
    var_10 = error.messages()[0]
    var_11 = 'thirty'
    var_12 = {var_0: var_2, var_1: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 'user'
    var_16 = {var_0: var_2, var_1: var_11}
    var_17 = {var_15: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = error.messages()[0]



# Parsed testcases at query #59
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'invalid'
    var_5 = 7
    var_6 = module_0.Token(var_4)
    var_7 = error.messages()[var_1]
    var_8 = 'field1'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = 15
    var_12 = module_0.Token(var_10)
    var_13 = error.messages()[var_1]
    var_14 = 'field2'
    var_15 = 'val1'
    var_16 = 'val2'
    var_17 = {var_8: var_15, var_14: var_16}
    var_18 = 20
    var_19 = 10
    var_20 = 14
    var_21 = [var_8]
    var_22 = module_0.Token(var_15)
    var_23 = 19
    var_24 = [var_14]
    var_25 = module_0.Token(var_16)
    var_26 = [var_22, var_25]
    var_27 = module_0.Token(var_17)
    var_28 = error.messages()[var_1]
    var_29 = 1
    var_30 = error.messages()[var_29]



# Parsed testcases at query #60
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



# Parsed testcases at query #61
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
    var_10 = module_0.Token(var_8)
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = {var_11: var_4, var_12: var_4}
    var_14 = 30
    var_15 = module_0.Token(var_13)



# Parsed testcases at query #62
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
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)
    var_32 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #63
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
    var_6 = None
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    var_11 = exc_info.value.messages()[var_1]
    var_12 = 'nested'
    var_13 = {var_12: var_6}
    var_14 = 10
    var_15 = module_0.Token(var_6)
    var_16 = [var_15]
    var_17 = module_0.Token(var_13)
    var_18 = module_1.Field()
    var_19 = {var_12: var_18}
    var_20 = module_3.Schema(var_19)
    var_21 = module_2.validate_with_positions(token=var_17, validator=var_20)
    var_22 = exc_info.value.messages()[var_1]
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = 'invalid'
    var_26 = {var_23: var_6, var_24: var_25}
    var_27 = 20
    var_28 = module_0.Token(var_6)
    var_29 = 12
    var_30 = module_0.Token(var_25)
    var_31 = [var_28, var_30]
    var_32 = module_0.Token(var_26)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_23: var_33, var_24: var_34}
    var_36 = module_3.Schema(var_35)
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)
    var_38 = lambda m: m.start_position



# Parsed testcases at query #64
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = {var_4: var_0}
    var_6 = 14
    var_7 = module_0.Token(var_5)
    var_8 = 'age'
    var_9 = True
    var_10 = list(error.messages())
    var_11 = 'invalid'
    var_12 = {var_8: var_11}
    var_13 = 16
    var_14 = module_0.Token(var_12)
    var_15 = list(error.messages())
    var_16 = 'user'
    var_17 = {var_4: var_0}
    var_18 = {var_16: var_17}
    var_19 = 22
    var_20 = module_0.Token(var_18)
    var_21 = list(error.messages())



# Parsed testcases at query #65
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = {var_13: var_4, var_14: var_4}
    var_16 = 30
    var_17 = module_0.Token(var_15)
    var_18 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #66
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = True
    var_6 = {}
    var_7 = 2
    var_8 = module_0.Token(var_6)
    var_9 = 'user'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = 10
    var_13 = module_0.Token(var_11)
    var_14 = 'age'
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = 'not_a_number'
    var_18 = 12
    var_19 = module_0.Token(var_17)



# Parsed testcases at query #67
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[0]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = error.messages()[0]
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'invalid'
    var_16 = {var_13: var_4, var_14: var_15}
    var_17 = 30
    var_18 = module_0.Token(var_16)
    var_19 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #68
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
    var_11 = 'invalid_value'
    var_12 = module_0.Token(var_11)
    var_13 = 5
    var_14 = module_1.Field()
    var_15 = module_2.validate_with_positions(token=var_12, validator=var_14)
    var_16 = 'nested'
    var_17 = {var_16: var_6}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.Field()
    var_20 = {var_16: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = 'short'
    var_26 = {var_23: var_6, var_24: var_25}
    var_27 = module_0.Token(var_26)
    var_28 = module_1.Field()
    var_29 = module_1.Field()
    var_30 = {var_23: var_28, var_24: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_27, validator=var_31)



# Parsed testcases at query #69
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
    var_11 = 'nested'
    var_12 = {var_11: var_7}
    var_13 = 10
    var_14 = module_0.Token(var_12)
    var_15 = module_1.Field()
    var_16 = {var_11: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_14, validator=var_17)
    var_19 = 'field1'
    var_20 = 'field2'
    var_21 = {var_19: var_7, var_20: var_7}
    var_22 = 20
    var_23 = module_0.Token(var_21)
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_19: var_24, var_20: var_25}
    var_27 = module_3.Schema(var_26)
    var_28 = module_2.validate_with_positions(token=var_23, validator=var_27)



# Parsed testcases at query #70
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
    var_13 = 'invalid'
    var_14 = {var_11: var_4, var_12: var_13}
    var_15 = 30
    var_16 = module_0.Token(var_4)
    var_17 = 15
    var_18 = module_0.Token(var_13)
    var_19 = [var_16, var_18]
    var_20 = module_0.Token(var_14)



# Parsed testcases at query #71
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = None
    var_5 = module_0.Token(var_4)
    var_6 = True
    var_7 = error.messages()[var_1]
    var_8 = 'nested'
    var_9 = {var_8: var_4}
    var_10 = 15
    var_11 = 7
    var_12 = 12
    var_13 = module_0.Token(var_4)
    var_14 = [var_13]
    var_15 = module_0.Token(var_9)
    var_16 = error.messages()[var_1]



# Parsed testcases at query #72
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
    var_6 = 'nested'
    var_7 = 'field'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 20
    var_12 = module_0.Token(var_10)
    var_13 = True
    var_14 = module_1.Field()
    var_15 = {var_7: var_14}
    var_16 = {var_6: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_2.validate_with_positions(token=var_12, validator=var_17)
    var_19 = 'field1'
    var_20 = 'field2'
    var_21 = 'invalid'
    var_22 = {var_19: var_8, var_20: var_21}
    var_23 = 30
    var_24 = module_0.Token(var_22)
    var_25 = module_1.Field()
    var_26 = 10
    var_27 = module_1.Field()
    var_28 = {var_19: var_25, var_20: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_24, validator=var_29)



# Parsed testcases at query #73
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
    var_4 = True
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    assert var_6 == 'valid_value'
    var_7 = None
    var_8 = module_0.Token(var_7)
    var_9 = module_1.Field()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = 'nested'
    var_12 = {var_11: var_7}
    var_13 = 15
    var_14 = 8
    var_15 = 13
    var_16 = module_0.Token(var_7)
    var_17 = {var_11: var_16}
    var_18 = module_0.Token(var_12)
    var_19 = module_1.Field()
    var_20 = {var_11: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = 'field1'
    var_24 = 'field2'
    var_25 = 'invalid'
    var_26 = {var_23: var_7, var_24: var_25}
    var_27 = 30
    var_28 = module_0.Token(var_7)
    var_29 = 16
    var_30 = 23
    var_31 = module_0.Token(var_25)
    var_32 = {var_23: var_28, var_24: var_31}
    var_33 = module_0.Token(var_26)
    var_34 = module_1.Field()
    var_35 = 10
    var_36 = module_1.Field()
    var_37 = {var_23: var_34, var_24: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_33, validator=var_38)



# Parsed testcases at query #74
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = True
    var_6 = 'age'
    var_7 = 25
    var_8 = {var_6: var_7}
    var_9 = 9
    var_10 = module_0.Token(var_6)
    var_11 = 5
    var_12 = 7
    var_13 = module_0.Token(var_7)
    var_14 = [var_10, var_13]
    var_15 = module_0.Token(var_8)
    var_16 = error.messages()[0]
    var_17 = 'user'
    var_18 = {}
    var_19 = {var_17: var_18}
    var_20 = 11
    var_21 = module_0.Token(var_17)
    var_22 = {}
    var_23 = 6
    var_24 = 10
    var_25 = []
    var_26 = module_0.Token(var_22)
    var_27 = [var_21, var_26]
    var_28 = module_0.Token(var_19)
    var_29 = error.messages()[0]
    var_30 = 'not_a_number'
    var_31 = 12
    var_32 = module_0.Token(var_30)
    var_33 = error.messages()[0]



# Parsed testcases at query #75
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
    var_22 = [var_17]
    var_23 = module_1.Field()
    var_24 = module_2.validate_with_positions(token=var_21, validator=var_23)
    var_25 = error.messages()[0]
    var_26 = 'nested'
    var_27 = 'field'
    var_28 = {var_27: var_19}
    var_29 = {var_26: var_28}
    var_30 = 25
    var_31 = {var_27: var_19}
    var_32 = 7
    var_33 = 16
    var_34 = module_0.Token(var_19)
    var_35 = [var_34]
    var_36 = module_0.Token(var_31)
    var_37 = [var_36]
    var_38 = module_0.Token(var_29)
    var_39 = [var_24]
    var_40 = module_1.Field()
    var_41 = {var_27: var_40}
    var_42 = module_3.Schema(var_41)
    var_43 = {var_26: var_42}
    var_44 = module_3.Schema(var_43)
    var_45 = module_2.validate_with_positions(token=var_38, validator=var_44)
    var_46 = error.messages()[0]



# Parsed testcases at query #76
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.base as module_3
import typesystem.schemas as module_4

def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'valid_value'
    var_6 = 'invalid_value'
    var_7 = 10
    var_8 = module_0.Token(var_6)
    var_9 = module_1.Field()
    var_10 = ()
    var_11 = 'Invalid'
    var_12 = 'invalid'
    var_13 = module_3.Message(text=var_11, code=var_12)
    var_14 = [var_13]
    var_15 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_16 = 'nested'
    var_17 = {var_16: var_12}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = module_1.Field()
    var_21 = {var_16: var_20}
    var_22 = module_4.Schema(var_21)
    var_23 = ()
    var_24 = 'Invalid nested'
    var_25 = [var_16]
    var_26 = module_3.Message(text=var_24, code=var_12, index=var_25)
    var_27 = [var_26]
    var_28 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_29 = {}
    var_30 = 2
    var_31 = module_0.Token(var_29)
    var_32 = 'required_field'
    var_33 = True
    var_34 = module_1.Field()
    var_35 = {var_32: var_34}
    var_36 = module_4.Schema(var_35)
    var_37 = ()
    var_38 = 'Required'
    var_39 = 'required'
    var_40 = [var_32]
    var_41 = module_3.Message(text=var_38, code=var_39, index=var_40)
    var_42 = [var_41]
    var_43 = module_2.validate_with_positions(token=var_31, validator=var_36)



# Parsed testcases at query #77
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0)
    var_4 = 'not_an_int'
    var_5 = 10
    var_6 = module_0.Token(var_4)
    var_7 = error.messages()[var_1]
    var_8 = 'name'
    var_9 = True
    var_10 = {}
    var_11 = module_0.Token(var_10)
    var_12 = error.messages()[var_1]
    var_13 = 'user'
    var_14 = 'age'
    var_15 = 'invalid'
    var_16 = {var_14: var_15}
    var_17 = {var_13: var_16}
    var_18 = 20
    var_19 = {var_14: var_15}
    var_20 = 7
    var_21 = 19
    var_22 = 13
    var_23 = 18
    var_24 = module_0.Token(var_15)
    var_25 = [var_24]
    var_26 = module_0.Token(var_19)
    var_27 = [var_26]
    var_28 = module_0.Token(var_17)
    var_29 = error.messages()[var_1]



# Parsed testcases at query #78
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



# Parsed testcases at query #79
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
    var_12 = error.messages()[var_4]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 20
    var_16 = module_0.Token(var_7)
    var_17 = [var_16]
    var_18 = module_0.Token(var_14)
    var_19 = module_1.Field()
    var_20 = {var_13: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = module_2.validate_with_positions(token=var_18, validator=var_21)
    var_23 = error.messages()[var_4]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = 'invalid'
    var_27 = {var_24: var_7, var_25: var_26}
    var_28 = 30
    var_29 = 5
    var_30 = 15
    var_31 = module_0.Token(var_7)
    var_32 = module_0.Token(var_26)
    var_33 = [var_31, var_32]
    var_34 = module_0.Token(var_27)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_24: var_35, var_25: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_34, validator=var_38)
    var_40 = error.messages()[var_4]
    var_41 = error.messages()[var_9]



# Parsed testcases at query #80
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
    var_11 = error.messages()[0]
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 15
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Field()
    var_17 = {var_12: var_16}
    var_18 = module_3.Schema(var_17)
    var_19 = module_2.validate_with_positions(token=var_15, validator=var_18)
    var_20 = error.messages()[0]
    var_21 = 'field1'
    var_22 = 'field2'
    var_23 = {var_21: var_7, var_22: var_7}
    var_24 = 25
    var_25 = module_0.Token(var_23)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_22: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #81
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
    var_9 = 'user'
    var_10 = {var_0: var_8}
    var_11 = {var_9: var_10}
    var_12 = 20
    var_13 = {var_0: var_8}
    var_14 = 5
    var_15 = 15
    var_16 = module_0.Token(var_13)
    var_17 = [var_16]
    var_18 = module_0.Token(var_11)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = 'age'
    var_21 = 'invalid'
    var_22 = {var_0: var_19, var_20: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)



# Parsed testcases at query #82
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
    var_15 = module_1.validate_with_positions(token=var_14, validator=var_0)
    var_16 = error.messages()[0]
    var_17 = {}
    var_18 = module_0.Token(var_17)
    var_19 = module_1.validate_with_positions(token=var_18, validator=var_0)
    var_20 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_21 = {var_0: var_19}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.validate_with_positions(token=var_22, validator=var_0)
    var_24 = error.messages()[0]



# Parsed testcases at query #83
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
    var_6 = ''
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = e.messages()[0]
    var_11 = {}
    var_12 = module_0.Token(var_11)
    var_13 = module_1.validate_with_positions(token=var_12, validator=var_0)
    var_14 = e.messages()[0]
    var_15 = 'nested'
    var_16 = {var_0: var_6}
    var_17 = {var_15: var_16}
    var_18 = 20
    var_19 = module_0.Token(var_17)
    var_20 = {var_0: var_6}
    var_21 = module_0.Token(var_20)
    var_22 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_23 = e.messages()[0]
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = {var_24: var_6, var_25: var_6}
    var_27 = module_0.Token(var_26)
    var_28 = module_0.Token(var_6)
    var_29 = module_0.Token(var_6)
    var_30 = module_1.validate_with_positions(token=var_27, validator=var_0)



# Parsed testcases at query #84
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test_value'
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



# Parsed testcases at query #85
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
    var_6 = ''
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = error.messages()[0]
    var_11 = {}
    var_12 = 5
    var_13 = module_0.Token(var_11)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = error.messages()[0]
    var_16 = 'user'
    var_17 = {var_0: var_6}
    var_18 = {var_16: var_17}
    var_19 = 20
    var_20 = {var_0: var_6}
    var_21 = 15
    var_22 = module_0.Token(var_20)
    var_23 = [var_22]
    var_24 = module_0.Token(var_18)
    var_25 = module_1.validate_with_positions(token=var_24, validator=var_0)
    var_26 = error.messages()[0]



# Parsed testcases at query #86
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = {}
    var_5 = 2
    var_6 = module_0.Token(var_4)
    var_7 = 'name'
    var_8 = True
    var_9 = error.messages()[0]
    var_10 = 'age'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = 15
    var_14 = module_0.Token(var_12)
    var_15 = error.messages()[0]
    var_16 = None
    var_17 = {var_7: var_16, var_10: var_11}
    var_18 = 25
    var_19 = module_0.Token(var_17)
    var_20 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #87
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
    var_8 = 'required_field'
    var_9 = True
    var_10 = module_1.Field()
    var_11 = {var_8: var_10}
    var_12 = module_3.Schema(var_11)
    var_13 = module_2.validate_with_positions(token=var_7, validator=var_12)
    var_14 = 'nested'
    var_15 = 'field'
    var_16 = 'invalid'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = 20
    var_20 = module_0.Token(var_18)
    var_21 = 5
    var_22 = module_1.Field()
    var_23 = {var_15: var_22}
    var_24 = module_3.Schema(var_23)
    var_25 = {var_14: var_24}
    var_26 = module_3.Schema(var_25)
    var_27 = module_2.validate_with_positions(token=var_20, validator=var_26)
    var_28 = 'field1'
    var_29 = 'field2'
    var_30 = 'a'
    var_31 = 'b'
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = 15
    var_34 = module_0.Token(var_32)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_28: var_35, var_29: var_36}
    var_38 = module_3.Schema(var_37)
    var_39 = module_2.validate_with_positions(token=var_34, validator=var_38)



# Parsed testcases at query #88
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
    var_10 = 'user'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = module_1.validate_with_positions(token=var_16, validator=var_0)



# Parsed testcases at query #89
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
    var_12 = 'nested'
    var_13 = {var_12: var_7}
    var_14 = 20
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



# Parsed testcases at query #90
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
    var_16 = 'invalid'
    var_17 = {var_15: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_12)
    var_20 = error.messages()[0]
    var_21 = 'user'
    var_22 = None
    var_23 = {var_8: var_22}
    var_24 = {var_21: var_23}
    var_25 = 20
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = {var_8: var_27}
    var_29 = module_3.Schema(var_28)
    var_30 = {var_21: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_2.validate_with_positions(token=var_26, validator=var_31)
    var_33 = error.messages()[0]



# Parsed testcases at query #91
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
    var_24 = {var_22: var_7, var_23: var_7}
    var_25 = 30
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)
    var_32 = sorted(error.messages(), key=lambda m: m.start_position.char_index)



# Parsed testcases at query #92
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
    var_12 = exc_info.value.messages()[var_4]
    var_13 = 'nested'
    var_14 = {var_13: var_7}
    var_15 = 15
    var_16 = module_0.Token(var_14)
    var_17 = module_1.Field()
    var_18 = {var_13: var_17}
    var_19 = module_3.Schema(var_18)
    var_20 = module_2.validate_with_positions(token=var_16, validator=var_19)
    var_21 = exc_info.value.messages()[var_4]
    var_22 = 'field1'
    var_23 = 'field2'
    var_24 = {var_22: var_7, var_23: var_7}
    var_25 = 25
    var_26 = module_0.Token(var_24)
    var_27 = module_1.Field()
    var_28 = module_1.Field()
    var_29 = {var_22: var_27, var_23: var_28}
    var_30 = module_3.Schema(var_29)
    var_31 = module_2.validate_with_positions(token=var_26, validator=var_30)



