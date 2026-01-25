####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'text'
    var_2 = 0
    var_3 = 10
    var_4 = module_0.Token(var_0)
    var_5 = module_1.String()
    var_6 = module_2.validate_with_positions(token=var_4, validator=var_5)
    assert var_6 == 'test_value'
    var_7 = 'not_an_integer'
    var_8 = 14
    var_9 = module_0.Token(var_7)
    var_10 = module_1.Integer()
    var_11 = module_2.validate_with_positions(token=var_9, validator=var_10)
    var_12 = {}
    var_13 = 5
    var_14 = module_0.Token(var_12)
    var_15 = {}
    var_16 = module_0.Token(var_15)
    var_17 = 'invalid'
    var_18 = 12
    var_19 = module_0.Token(var_17)
    var_20 = module_1.Integer()
    var_21 = module_2.validate_with_positions(token=var_19, validator=var_20)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_with_positions with valid input.'
    var_1 = '42'
    var_2 = module_0.Integer()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_with_positions with invalid type.'
    var_1 = 'not_a_number'
    var_2 = module_0.Integer()

def test_case_0():
    var_0 = 'Test validate_with_positions with required field missing.'
    var_1 = {}
    var_2 = 'required'

def test_case_0():
    var_0 = 'Test that validation error messages are sorted by position.'
    var_1 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_with_positions with valid string field.'
    var_1 = 'hello'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that message code is preserved in validation error.'
    var_1 = 'not_a_number'
    var_2 = module_0.Integer()
    var_3 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_with_positions with a schema validator.'
    var_1 = module_0.Integer()
    var_2 = 'age'
    var_3 = 25
    var_4 = {var_2: var_3}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_with_positions with invalid schema data.'
    var_1 = module_0.Integer()
    var_2 = 'age'
    var_3 = 'not_an_int'
    var_4 = {var_2: var_3}



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = None
    var_2 = module_0.Token(var_0)
    var_3 = module_1.Field()
    var_4 = module_2.validate_with_positions(token=var_2, validator=var_3)
    assert var_4 == 'test_value'
    var_5 = {}
    var_6 = module_3.Schema(var_5)
    var_7 = {}
    var_8 = module_0.Token(var_7)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_6)
    var_10 = 'age'
    var_11 = 25
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = 'obj'
    var_15 = 'char_index'
    var_16 = 0
    var_17 = {var_15: var_16}
    var_18 = 10
    var_19 = {var_15: var_18}
    var_20 = module_2.validate_with_positions(token=var_13, validator=var_0)
    var_21 = 'required'
    var_22 = 'start_position'
    var_23 = 'end_position'
    var_24 = lambda m: m.start_position.char_index if hasattr(m.start_position, var_15) else var_16
    var_25 = 'invalid_int'
    var_26 = {var_10: var_25}
    var_27 = module_0.Token(var_26)
    var_28 = {var_15: var_16}
    var_29 = 15
    var_30 = {var_15: var_29}
    var_31 = module_2.validate_with_positions(token=var_27, validator=var_0)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'field_name'
    var_2 = 0
    var_3 = 10

def test_case_0():
    var_0 = 'field'
    var_1 = 0
    var_2 = 10

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 0
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = 5
    var_7 = 15

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 20
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_2.Integer()
    var_17 = 'not_an_int'
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = 10
    var_20 = module_0.Position(var_19, var_1, var_19)
    var_21 = module_1.Token(var_17)
    var_22 = module_3.validate_with_positions(token=var_21, validator=var_16)
    var_23 = None
    var_24 = {var_9: var_23}
    var_25 = module_0.Position(var_1, var_1, var_1)
    var_26 = module_0.Position(var_13, var_1, var_13)
    var_27 = module_1.Token(var_24)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_2)
    var_4 = 5
    var_5 = module_0.Position(var_4)
    var_6 = module_1.Token(var_0)
    var_7 = module_2.String()
    var_8 = module_3.validate_with_positions(token=var_6, validator=var_7)
    assert var_8 == 'hello'
    var_9 = module_2.String()
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = module_0.Position(var_2)
    var_14 = 20
    var_15 = module_0.Position(var_14)
    var_16 = module_1.Token(var_12)
    var_17 = None
    var_18 = {var_10: var_17}
    var_19 = module_0.Position(var_2)
    var_20 = 10
    var_21 = module_0.Position(var_20)
    var_22 = module_1.Token(var_18)
    var_23 = 'not_an_integer'
    var_24 = module_0.Position(var_2)
    var_25 = 14
    var_26 = module_0.Position(var_25)
    var_27 = module_1.Token(var_23)
    var_28 = module_2.Integer()
    var_29 = module_3.validate_with_positions(token=var_27, validator=var_28)
    var_30 = {}
    var_31 = module_0.Position(var_2)
    var_32 = module_0.Position(var_20)
    var_33 = module_1.Token(var_30)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = 0
    var_3 = 10
    var_4 = 5
    var_5 = 15
    var_6 = 'This field is required.'
    var_7 = 'required'
    var_8 = 'field'
    var_9 = [var_0, var_8]
    var_10 = module_0.Message(text=var_6, code=var_7, index=var_9)
    var_11 = [var_10]
    var_12 = module_0.ValidationError(messages=var_11)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = 'invalid'
    var_2 = 0
    var_3 = 20
    var_4 = 10
    var_5 = 25
    var_6 = 'Invalid format.'
    var_7 = [var_0]
    var_8 = module_0.Message(text=var_6, code=var_1, index=var_7)
    var_9 = [var_8]
    var_10 = module_0.ValidationError(messages=var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'invalid'
    var_3 = 20
    var_4 = 30
    var_5 = 5
    var_6 = 15
    var_7 = 'Error 1'
    var_8 = 'error1'
    var_9 = [var_0]
    var_10 = module_0.Message(text=var_7, code=var_8, index=var_9)
    var_11 = 'Error 2'
    var_12 = 'error2'
    var_13 = [var_1]
    var_14 = module_0.Message(text=var_11, code=var_12, index=var_13)
    var_15 = [var_10, var_14]
    var_16 = module_0.ValidationError(messages=var_15)

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'validated_value'
    var_3 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'nested'
    var_3 = 'field_name'

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 'field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 10
    var_3 = 20
    var_4 = 5
    var_5 = 8



# Parsed testcases at query #9
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1)
    var_13 = 20
    var_14 = module_0.Position(var_13)
    var_15 = module_1.Token(var_11)
    var_16 = 'not_a_number'
    var_17 = module_0.Position(var_1)
    var_18 = 12
    var_19 = module_0.Position(var_18)
    var_20 = module_1.Token(var_16)
    var_21 = module_2.Integer()
    var_22 = module_3.validate_with_positions(token=var_20, validator=var_21)
    var_23 = {}
    var_24 = module_0.Position(var_1)
    var_25 = 10
    var_26 = module_0.Position(var_25)
    var_27 = module_1.Token(var_23)
    var_28 = 'test'
    var_29 = module_0.Position(var_1)
    var_30 = 4
    var_31 = module_0.Position(var_30)
    var_32 = module_1.Token(var_28)
    var_33 = 2
    var_34 = module_2.String(max_length=var_33)
    var_35 = module_3.validate_with_positions(token=var_32, validator=var_34)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = '123'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0)
    var_4 = 'Position'
    var_5 = ()
    var_6 = 'char_index'
    var_7 = {var_6: var_1}
    var_8 = type(var_4, var_5, var_7)
    var_9 = ()
    var_10 = {var_6: var_2}
    var_11 = type(var_4, var_9, var_10)
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_3, validator=var_12)
    assert var_13 == '123'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = [var_3]

import typesystem.base as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'required'
    var_2 = 'field_name'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]

import typesystem.base as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'invalid'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'field2'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_0, code=var_1, index=var_6)
    var_8 = [var_4, var_7]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = {}
    var_2 = ''
    var_3 = 'required'
    var_4 = 'inner_field'
    var_5 = [var_0, var_4]
    var_6 = module_0.Message(text=var_2, code=var_3, index=var_5)
    var_7 = [var_6]



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_an_integer'
    var_1 = 0
    var_2 = 14
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Integer()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    var_6 = 'start_position'
    var_7 = 'end_position'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = False
    var_5 = module_1.String()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = module_1.Token(var_6)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'char_index'



# Parsed testcases at query #12
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'Field is required'
    var_4 = 'required'
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_3, code=var_4, index=var_6)
    var_8 = [var_7]
    var_9 = 5
    var_10 = 'age'
    var_11 = 'invalid'
    var_12 = 20
    var_13 = 'Not a valid integer'
    var_14 = 'type_error'
    var_15 = [var_10]
    var_16 = module_0.Message(text=var_13, code=var_14, index=var_15)
    var_17 = [var_16]
    var_18 = 12
    var_19 = 'field1'
    var_20 = 'field2'
    var_21 = 50
    var_22 = 'Error 1'
    var_23 = 'error'
    var_24 = [var_19]
    var_25 = module_0.Message(text=var_22, code=var_23, index=var_24)
    var_26 = 'Error 2'
    var_27 = [var_20]
    var_28 = module_0.Message(text=var_26, code=var_23, index=var_27)
    var_29 = [var_25, var_28]
    var_30 = 25
    var_31 = 15



# Parsed testcases at query #13
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1)
    var_13 = 20
    var_14 = module_0.Position(var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_3.validate_with_positions(token=var_15, validator=var_6)
    var_17 = 'not_a_number'
    var_18 = module_0.Position(var_1)
    var_19 = 12
    var_20 = module_0.Position(var_19)
    var_21 = module_1.Token(var_17)
    var_22 = module_2.Integer()
    var_23 = module_3.validate_with_positions(token=var_21, validator=var_22)
    var_24 = {}
    var_25 = module_0.Position(var_1)
    var_26 = 2
    var_27 = module_0.Position(var_26)
    var_28 = module_1.Token(var_24)
    var_29 = module_3.validate_with_positions(token=var_28, validator=var_22)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'
    var_3 = 'subfield'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test successful validation'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'Test validation error with field validation'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 5
    var_5 = var_3 * var_4
    var_6 = 30
    var_7 = {var_1: var_5, var_2: var_6}

def test_case_0():
    var_0 = 'Test validation error for required field'
    var_1 = 'age'
    var_2 = 30
    var_3 = {var_1: var_2}
    var_4 = 'required'

def test_case_0():
    var_0 = 'Test that error messages are sorted by position'
    var_1 = {}

def test_case_0():
    var_0 = 'Test that returned messages have position information'
    var_1 = 'age'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = 'start_position'
    var_5 = 'end_position'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation with a single Field validator'
    var_1 = 'not_a_number'
    var_2 = module_0.Integer()

def test_case_0():
    var_0 = 'Test that message codes are preserved'
    var_1 = {}
    var_2 = 'code'



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = module_1.String()
    var_7 = 'name'
    var_8 = 'John'
    var_9 = {var_7: var_8}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = None
    var_13 = {var_7: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = 'not_an_integer'
    var_16 = 14
    var_17 = module_0.Token(var_15)
    var_18 = module_1.Integer()
    var_19 = module_2.validate_with_positions(token=var_17, validator=var_18)
    var_20 = {}
    var_21 = 50
    var_22 = module_0.Token(var_20)



# Parsed testcases at query #17
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 10
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_3.validate_with_positions(token=var_15, validator=var_6)
    var_17 = 'not_a_number'
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = 12
    var_20 = module_0.Position(var_19, var_1, var_19)
    var_21 = module_1.Token(var_17)
    var_22 = module_2.Integer()
    var_23 = module_3.validate_with_positions(token=var_21, validator=var_22)
    var_24 = list(error.messages())
    var_25 = {}
    var_26 = module_0.Position(var_1, var_1, var_1)
    var_27 = 2
    var_28 = module_0.Position(var_27, var_1, var_27)
    var_29 = module_1.Token(var_25)
    var_30 = module_3.validate_with_positions(token=var_29, validator=var_22)
    var_31 = list(error.messages())
    var_32 = module_0.Position(var_1, var_1, var_1)
    var_33 = 4
    var_34 = module_0.Position(var_33, var_1, var_33)
    var_35 = module_1.Token(var_10)
    var_36 = module_2.String()
    var_37 = module_3.validate_with_positions(token=var_35, validator=var_36)
    assert var_37 == 'test'



# Parsed testcases at query #18
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 10
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'test_value'
    var_8 = 'age'
    var_9 = 25
    var_10 = {var_8: var_9}
    var_11 = module_0.Position(var_1, var_1, var_1)
    var_12 = 20
    var_13 = module_0.Position(var_12, var_1, var_12)
    var_14 = module_1.Token(var_10)
    var_15 = module_3.validate_with_positions(token=var_14, validator=var_0)
    var_16 = 'required'
    var_17 = 'start_position'
    var_18 = 'end_position'
    var_19 = 'invalid'
    var_20 = {var_8: var_19}
    var_21 = module_0.Position(var_15, var_15, var_15)
    var_22 = 30
    var_23 = module_0.Position(var_22, var_15, var_22)
    var_24 = module_1.Token(var_20)
    var_25 = module_3.validate_with_positions(token=var_24, validator=var_0)
    var_26 = 'name'
    var_27 = 123
    var_28 = 'not_an_int'
    var_29 = {var_26: var_27, var_8: var_28}
    var_30 = module_0.Position(var_25, var_25, var_25)
    var_31 = 40
    var_32 = module_0.Position(var_31, var_25, var_31)
    var_33 = module_1.Token(var_29)
    var_34 = module_3.validate_with_positions(token=var_33, validator=var_0)



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_2)
    var_4 = 11
    var_5 = 10
    var_6 = module_0.Position(var_5)
    var_7 = module_1.Token(var_0)
    var_8 = module_2.String()
    var_9 = module_3.validate_with_positions(token=var_7, validator=var_8)
    assert var_9 == 'test_value'
    var_10 = module_2.String()
    var_11 = 'name'
    var_12 = 'John'
    var_13 = {var_11: var_12}
    var_14 = module_0.Position(var_2)
    var_15 = 20
    var_16 = 19
    var_17 = module_0.Position(var_16)
    var_18 = module_1.Token(var_13)
    var_19 = module_2.Integer()
    var_20 = 'age'
    var_21 = 'not_an_int'
    var_22 = {var_20: var_21}
    var_23 = module_0.Position(var_2)
    var_24 = 25
    var_25 = 24
    var_26 = module_0.Position(var_25)
    var_27 = module_1.Token(var_22)
    var_28 = {}
    var_29 = module_0.Position(var_2)
    var_30 = 9
    var_31 = module_0.Position(var_30)
    var_32 = module_1.Token(var_28)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'hello'
    var_6 = module_1.String()
    var_7 = 'name'
    var_8 = 'John'
    var_9 = {var_7: var_8}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = module_1.Integer()
    var_13 = 'not_an_int'
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = module_2.validate_with_positions(token=var_15, validator=var_12)
    var_17 = 'start_position'
    var_18 = 'end_position'
    var_19 = {}
    var_20 = module_0.Token(var_19)
    var_21 = 'required'
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = 'test'
    var_25 = 14
    var_26 = module_0.Token(var_24)
    var_27 = module_1.String()
    var_28 = module_2.validate_with_positions(token=var_26, validator=var_27)
    assert var_28 == 'test'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_string'
    var_1 = 0
    var_2 = 11
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_string'
    var_6 = 123
    var_7 = 3
    var_8 = module_0.Token(var_6)
    var_9 = module_1.String()
    var_10 = module_2.validate_with_positions(token=var_8, validator=var_9)
    var_11 = {}
    var_12 = 2
    var_13 = module_0.Token(var_11)
    var_14 = 42
    var_15 = 5
    var_16 = 7
    var_17 = module_0.Token(var_14)
    var_18 = module_1.Integer()
    var_19 = module_2.validate_with_positions(token=var_17, validator=var_18)
    assert var_19 == 42
    var_20 = 'not_an_int'
    var_21 = 10
    var_22 = module_0.Token(var_20)
    var_23 = module_1.Integer()
    var_24 = module_2.validate_with_positions(token=var_22, validator=var_23)
    var_25 = {}
    var_26 = module_0.Token(var_25)



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 10
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'test_value'
    var_8 = module_2.String()
    var_9 = module_2.Integer()
    var_10 = 'age'
    var_11 = 25
    var_12 = {var_10: var_11}
    var_13 = module_0.Position(var_1, var_1, var_1)
    var_14 = 20
    var_15 = module_0.Position(var_14, var_1, var_14)
    var_16 = module_1.Token(var_12)
    var_17 = list(error.messages())
    var_18 = 'not_an_integer'
    var_19 = module_0.Position(var_1, var_1, var_1)
    var_20 = 14
    var_21 = module_0.Position(var_20, var_1, var_20)
    var_22 = module_1.Token(var_18)
    var_23 = module_2.Integer()
    var_24 = module_3.validate_with_positions(token=var_22, validator=var_23)
    var_25 = list(error.messages())
    var_26 = module_2.String()
    var_27 = module_2.String()
    var_28 = {}
    var_29 = module_0.Position(var_1, var_1, var_1)
    var_30 = 50
    var_31 = module_0.Position(var_30, var_1, var_30)
    var_32 = module_1.Token(var_28)
    var_33 = list(error.messages())
    var_34 = [m.start_position.char_index for m in var_33]
    var_35 = 'field1'
    var_36 = 'field2'
    var_37 = 'value1'
    var_38 = 'value2'
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = module_0.Position(var_1, var_1, var_1)
    var_41 = module_0.Position(var_30, var_1, var_30)
    var_42 = module_1.Token(var_39)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test successful validation without errors.'
    var_1 = module_0.String()
    var_2 = 'valid_string'

def test_case_0():
    var_0 = 'Test validation error with non-required error code.'
    var_1 = 'invalid_value'

def test_case_0():
    var_0 = 'Test validation error with required error code.'
    var_1 = 'some_value'

def test_case_0():
    var_0 = 'Test validation error with nested field index.'
    var_1 = 'value'

def test_case_0():
    var_0 = 'Test multiple validation errors are sorted by position.'
    var_1 = 'value'

def test_case_0():
    var_0 = 'Test validation error with empty index.'
    var_1 = 'value'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'test_value'
    var_3 = 'nested'
    var_4 = 'field'
    var_5 = [var_3]
    var_6 = 5
    var_7 = 15
    var_8 = 8
    var_9 = 'field1'
    var_10 = 'field2'



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'obj'
    var_5 = 'char_index'
    var_6 = {var_5: var_1}
    var_7 = {var_5: var_2}
    var_8 = module_1.String()
    var_9 = module_2.validate_with_positions(token=var_3, validator=var_8)
    assert var_9 == 'test_value'
    var_10 = {}
    var_11 = 5
    var_12 = module_0.Token(var_10)
    var_13 = {var_5: var_1}
    var_14 = {var_5: var_11}
    var_15 = 'invalid'
    var_16 = 7
    var_17 = module_0.Token(var_15)
    var_18 = {var_5: var_1}
    var_19 = {var_5: var_16}
    var_20 = module_1.Integer()
    var_21 = module_2.validate_with_positions(token=var_17, validator=var_20)
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = {var_5: var_1}
    var_25 = {var_5: var_2}
    var_26 = module_2.validate_with_positions(token=var_23, validator=var_21)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 5
    var_3 = 'Field is required'
    var_4 = 'required'
    var_5 = 'name'
    var_6 = 'field'
    var_7 = [var_5, var_6]
    var_8 = False
    var_9 = True
    var_10 = 'invalid'
    var_11 = 10
    var_12 = 'Invalid value'
    var_13 = [var_5]
    var_14 = 12
    var_15 = False
    var_16 = True
    var_17 = 'field1'
    var_18 = 'field2'
    var_19 = 'bad'
    var_20 = 20
    var_21 = 'Error 1'
    var_22 = 'error'
    var_23 = [var_17]
    var_24 = 15
    var_25 = 'Error 2'
    var_26 = [var_18]
    var_27 = False
    var_28 = True



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'test_value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = ''
    var_3 = 'test@example.com'
    var_4 = 0
    var_5 = 10
    var_6 = 5
    var_7 = 15
    var_8 = 'This field is required.'
    var_9 = 'required'
    var_10 = [var_0]
    var_11 = module_0.Message(text=var_8, code=var_9, index=var_10)
    var_12 = [var_11]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 'invalid'
    var_2 = 0
    var_3 = 20
    var_4 = 8
    var_5 = 17
    var_6 = 'Not a valid integer.'
    var_7 = 'type_error'
    var_8 = [var_0]
    var_9 = module_0.Message(text=var_6, code=var_7, index=var_8)
    var_10 = [var_9]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'invalid'
    var_3 = 20
    var_4 = 30
    var_5 = 5
    var_6 = 15
    var_7 = 'Error 1'
    var_8 = 'error1'
    var_9 = [var_0]
    var_10 = module_0.Message(text=var_7, code=var_8, index=var_9)
    var_11 = 'Error 2'
    var_12 = 'error2'
    var_13 = [var_1]
    var_14 = module_0.Message(text=var_11, code=var_12, index=var_13)
    var_15 = [var_10, var_14]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = ''
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 30
    var_6 = 10
    var_7 = 20
    var_8 = 'This field is required.'
    var_9 = 'required'
    var_10 = [var_0, var_1]
    var_11 = module_0.Message(text=var_8, code=var_9, index=var_10)
    var_12 = [var_11]
    var_13 = [var_0]



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field_name'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'nested'
    var_5 = 'field'

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #29
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 10
    var_4 = module_0.Position(var_1, var_3, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'test_value'

import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not_a_number'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 12
    var_4 = module_0.Position(var_1, var_3, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.Integer()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    var_8 = 'start_position'
    var_9 = 'end_position'

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = {}
    var_2 = 0
    var_3 = module_1.Position(var_2, var_2, var_2)
    var_4 = 2
    var_5 = module_1.Position(var_2, var_4, var_4)
    var_6 = module_2.Token(var_1)
    var_7 = 'start_position'
    var_8 = 'end_position'

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = {}
    var_3 = 0
    var_4 = module_1.Position(var_3, var_3, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_3, var_5, var_5)
    var_7 = module_2.Token(var_2)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = {}
    var_2 = 0
    var_3 = module_1.Position(var_2, var_2, var_2)
    var_4 = 5
    var_5 = module_1.Position(var_2, var_4, var_4)
    var_6 = module_2.Token(var_1)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'field1'
    var_1 = None
    var_2 = 0
    var_3 = 10
    var_4 = 'nested'
    var_5 = 'field_name'

def test_case_0():
    var_0 = 'field'
    var_1 = 'invalid'
    var_2 = 5
    var_3 = 15

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'invalid1'
    var_3 = 'invalid2'
    var_4 = 20
    var_5 = 30
    var_6 = 5
    var_7 = 10



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'hello'
    var_6 = 12345
    var_7 = module_0.Token(var_6)
    var_8 = True
    var_9 = module_1.String()
    var_10 = module_2.validate_with_positions(token=var_7, validator=var_9)
    assert var_10 == '12345'
    var_11 = module_1.String()
    var_12 = module_1.Integer()
    var_13 = {}
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = False
    var_17 = module_2.validate_with_positions(token=var_15, validator=var_0)
    var_18 = True
    assert var_18 is True
    var_19 = list(error.messages())
    var_20 = 'age'
    var_21 = 'invalid'
    var_22 = {var_20: var_21}
    var_23 = 20
    var_24 = module_0.Token(var_22)
    var_25 = False
    var_26 = module_2.validate_with_positions(token=var_24, validator=var_0)
    var_27 = True
    assert var_27 is True
    var_28 = list(error.messages())
    var_29 = [msg.start_position.char_index for msg in var_28 if hasattr(msg.start_position, 'char_index')]
    var_30 = 42
    var_31 = 2
    var_32 = module_0.Token(var_30)
    var_33 = module_1.Integer()
    var_34 = module_2.validate_with_positions(token=var_32, validator=var_33)
    assert var_34 == 42



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'parent'
    var_3 = 'field_name'

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 'field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 20
    var_3 = 25
    var_4 = 5
    var_5 = 10

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = 'validated_value'
    var_3 = {var_0: var_1}



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'hello'
    var_6 = module_1.String()
    var_7 = 'name'
    var_8 = 'John'
    var_9 = {var_7: var_8}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = None
    var_13 = {var_7: var_12}
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Integer()
    var_17 = 'not_an_integer'
    var_18 = 14
    var_19 = module_0.Token(var_17)
    var_20 = module_2.validate_with_positions(token=var_19, validator=var_16)
    var_21 = {}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.String(max_length=var_2)
    var_24 = 'this_is_too_long'
    var_25 = 16
    var_26 = module_0.Token(var_24)
    var_27 = module_2.validate_with_positions(token=var_26, validator=var_23)



# Parsed testcases at query #34
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position()
    var_4 = 10
    var_5 = module_0.Position()
    var_6 = module_1.Token(var_0)
    var_7 = module_2.String()
    var_8 = module_3.validate_with_positions(token=var_6, validator=var_7)
    assert var_8 == 'test_value'
    var_9 = module_2.String()
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = module_0.Position()
    var_14 = 20
    var_15 = module_0.Position()
    var_16 = module_1.Token(var_12)
    var_17 = None
    var_18 = {var_10: var_17}
    var_19 = module_0.Position()
    var_20 = 15
    var_21 = module_0.Position()
    var_22 = module_1.Token(var_18)
    var_23 = 'not_an_int'
    var_24 = module_0.Position()
    var_25 = module_0.Position()
    var_26 = module_1.Token(var_23)
    var_27 = module_2.Integer()
    var_28 = module_3.validate_with_positions(token=var_26, validator=var_27)
    var_29 = 'field1'
    var_30 = 'field2'
    var_31 = {var_29: var_17, var_30: var_17}
    var_32 = module_0.Position()
    var_33 = 30
    var_34 = module_0.Position()
    var_35 = module_1.Token(var_31)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test_value'
    var_2 = 0
    var_3 = 10
    var_4 = module_1.Token(var_1)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_0)
    assert var_5 == 'test_value'

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 'not_an_integer'
    var_2 = 0
    var_3 = 14
    var_4 = module_1.Token(var_1)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_0)
    var_6 = 'start_position'
    var_7 = 'end_position'

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = 5
    var_4 = module_1.Token(var_2)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_1)
    var_6 = 'required'

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 20
    var_6 = module_1.Token(var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'this_is_too_long'
    var_3 = 0
    var_4 = 16
    var_5 = module_1.Token(var_2)
    var_6 = module_2.validate_with_positions(token=var_5, validator=var_1)



# Parsed testcases at query #36
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 4
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'test'
    var_8 = '123'
    var_9 = module_0.Position(var_1, var_1, var_1)
    var_10 = 3
    var_11 = module_0.Position(var_10, var_1, var_10)
    var_12 = module_1.Token(var_8)
    var_13 = module_2.Integer()
    var_14 = module_3.validate_with_positions(token=var_12, validator=var_13)
    assert var_14 == 123
    var_15 = {}
    var_16 = module_0.Position(var_1, var_1, var_1)
    var_17 = 2
    var_18 = module_0.Position(var_17, var_1, var_17)
    var_19 = module_1.Token(var_15)
    var_20 = 'not_a_number'
    var_21 = module_0.Position(var_1, var_1, var_1)
    var_22 = 12
    var_23 = module_0.Position(var_22, var_1, var_22)
    var_24 = module_1.Token(var_20)
    var_25 = module_2.Integer()
    var_26 = module_3.validate_with_positions(token=var_24, validator=var_25)
    var_27 = {}
    var_28 = module_0.Position(var_1, var_1, var_1)
    var_29 = module_0.Position(var_17, var_1, var_17)
    var_30 = module_1.Token(var_27)



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = module_1.String()
    var_7 = module_1.Integer()
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'John'
    var_11 = 30
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 20
    var_14 = module_0.Token(var_12)
    var_15 = 123
    var_16 = 5
    var_17 = module_0.Token(var_15)
    var_18 = module_1.String()
    var_19 = module_2.validate_with_positions(token=var_17, validator=var_18)
    var_20 = {}
    var_21 = module_0.Token(var_20)
    var_22 = 'required'
    var_23 = 'start_position'
    var_24 = 'end_position'
    var_25 = {}
    var_26 = module_0.Token(var_25)



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'field_name'
    var_4 = exc_info.value.messages()[var_1]
    var_5 = var_4.code
    assert var_5 == 'required'
    var_6 = exc_info.value.messages()[var_1]
    var_7 = var_6.text
    var_8 = 5
    var_9 = 12
    var_10 = 'field'
    var_11 = exc_info.value.messages()[var_1]
    var_12 = var_11.code
    assert var_12 == 'invalid_type'
    var_13 = exc_info.value.messages()[var_1]
    var_14 = var_13.start_position.char_index
    assert var_14 == 5
    var_15 = exc_info.value.messages()[var_1]
    var_16 = var_15.end_position.char_index
    assert var_16 == 12
    var_17 = 'field1'
    var_18 = 'field2'
    var_19 = 20
    var_20 = 25
    var_21 = 'nested'
    var_22 = 'data'
    var_23 = 15
    var_24 = 'nested_field'
    var_25 = exc_info.value.messages()[var_1]
    var_26 = var_25.code
    assert var_26 == 'required'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field_name'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'nested'
    var_5 = 'field'

def test_case_0():
    var_0 = 20
    var_1 = 25
    var_2 = 5
    var_3 = 10
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #40
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = 123
    var_9 = module_0.Position(var_1)
    var_10 = 3
    var_11 = module_0.Position(var_10)
    var_12 = module_1.Token(var_8)
    var_13 = module_2.String()
    var_14 = module_3.validate_with_positions(token=var_12, validator=var_13)
    var_15 = module_2.String()
    var_16 = {}
    var_17 = module_0.Position(var_1)
    var_18 = module_0.Position(var_3)
    var_19 = module_1.Token(var_16)
    var_20 = 'test'
    var_21 = module_0.Position(var_1)
    var_22 = 4
    var_23 = module_0.Position(var_22)
    var_24 = module_1.Token(var_20)
    var_25 = module_2.Integer()
    var_26 = module_3.validate_with_positions(token=var_24, validator=var_25)
    var_27 = 42
    var_28 = module_0.Position(var_1)
    var_29 = 2
    var_30 = module_0.Position(var_29)
    var_31 = module_1.Token(var_27)
    var_32 = module_2.Integer()
    var_33 = module_3.validate_with_positions(token=var_31, validator=var_32)
    assert var_33 == 42



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = 'obj'
    var_5 = 'char_index'
    var_6 = {var_5: var_1}
    var_7 = {var_5: var_2}
    var_8 = module_1.String()
    var_9 = module_2.validate_with_positions(token=var_3, validator=var_8)
    assert var_9 == 'hello'
    var_10 = module_1.String()
    var_11 = 'name'
    var_12 = 'John'
    var_13 = {var_11: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = {var_5: var_1}
    var_17 = {var_5: var_14}
    var_18 = 'not_an_int'
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = {var_5: var_1}
    var_22 = {var_5: var_19}
    var_23 = module_1.Integer()
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_23)
    var_25 = {}
    var_26 = module_0.Token(var_25)
    var_27 = {var_5: var_1}
    var_28 = {var_5: var_19}



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test with valid value that passes validation'
    var_1 = 'test_value'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error with non-required error code'
    var_1 = 'invalid'
    var_2 = module_0.Integer()
    var_3 = 'start_position'
    var_4 = 'end_position'

def test_case_0():
    var_0 = 'Test validation error with required field missing'
    var_1 = {}
    var_2 = 'required'

def test_case_0():
    var_0 = 'Test that messages are sorted by start position'
    var_1 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that message code is preserved'
    var_1 = 'invalid_int'
    var_2 = module_0.Integer()
    var_3 = None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error in nested schema'
    var_1 = module_0.Integer()
    var_2 = 'inner'
    var_3 = 'value'
    var_4 = 'not_an_int'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'start_position'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'
    var_3 = 'subfield'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = module_0.Token(var_2)
    var_6 = module_1.Field()
    var_7 = {var_0: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_5, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'name'
    var_5 = False
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = {var_4: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_3, validator=var_8)
    var_10 = 'start_position'
    var_11 = 'end_position'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = module_0.Token(var_0)
    var_4 = 5
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    var_7 = 'start_position'
    var_8 = 'end_position'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 100
    var_3 = module_0.Token(var_0)
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = False
    var_7 = module_1.Field(allow_null=var_6)
    var_8 = False
    var_9 = module_1.Field(allow_null=var_8)
    var_10 = {var_4: var_7, var_5: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_3, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'email'
    var_5 = False
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = {var_4: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_3, validator=var_8)
    var_10 = 'required'



# Parsed testcases at query #5
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 10
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = 'not_a_number'
    var_17 = module_0.Position(var_1, var_1, var_1)
    var_18 = 12
    var_19 = module_0.Position(var_18, var_1, var_18)
    var_20 = module_1.Token(var_16)
    var_21 = module_2.Integer()
    var_22 = module_3.validate_with_positions(token=var_20, validator=var_21)
    var_23 = 'required_field'
    var_24 = None
    var_25 = {var_23: var_24}
    var_26 = module_0.Position(var_1, var_1, var_1)
    var_27 = 20
    var_28 = module_0.Position(var_27, var_1, var_27)
    var_29 = module_1.Token(var_25)
    var_30 = module_2.String()
    var_31 = module_2.String()
    var_32 = 'field_a'
    var_33 = 'field_b'
    var_34 = 123
    var_35 = 456
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.Position(var_1, var_1, var_1)
    var_38 = 30
    var_39 = module_0.Position(var_38, var_1, var_38)
    var_40 = module_1.Token(var_36)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test'
    var_2 = 0
    var_3 = 4
    var_4 = module_1.Token(var_1)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_0)
    assert var_5 == 'test'
    var_6 = module_0.Integer()
    var_7 = ''
    var_8 = module_1.Token(var_7)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_6)
    var_10 = 5
    var_11 = module_0.String(max_length=var_10)
    var_12 = 'toolongstring'
    var_13 = 13
    var_14 = module_1.Token(var_12)
    var_15 = module_2.validate_with_positions(token=var_14, validator=var_11)
    var_16 = module_0.String()
    var_17 = {}
    var_18 = 2
    var_19 = module_1.Token(var_17)
    var_20 = module_0.String(max_length=var_10)
    var_21 = module_1.Token(var_12)
    var_22 = module_2.validate_with_positions(token=var_21, validator=var_20)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'nested'
    var_4 = 'field'
    var_5 = exc_info.value.messages()[var_1]
    var_6 = 5
    var_7 = 15
    var_8 = exc_info.value.messages()[var_1]
    var_9 = 20
    var_10 = 'field1'
    var_11 = 'field2'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'name'
    var_2 = 'John'
    var_3 = 0
    var_4 = 10
    var_5 = 'field_name'
    var_6 = False
    var_7 = True
    var_8 = 'age'
    var_9 = 'not_a_number'
    var_10 = 5
    var_11 = 20
    var_12 = False
    var_13 = True
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = 'invalid'
    var_17 = 'also_invalid'
    var_18 = 30
    var_19 = 15
    var_20 = False
    var_21 = True



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test successful validation returns the validated value.'
    var_1 = '123'
    var_2 = module_0.Integer()

def test_case_0():
    var_0 = 'Test validation error for required field.'
    var_1 = {}
    var_2 = 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error for invalid type.'
    var_1 = 'not_a_number'
    var_2 = module_0.Integer()

def test_case_0():
    var_0 = 'Test that error messages are sorted by start position.'
    var_1 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that positional messages have correct attributes.'
    var_1 = 'invalid'
    var_2 = module_0.Integer()
    var_3 = 0
    var_4 = 'text'
    var_5 = 'code'
    var_6 = 'index'
    var_7 = 'start_position'
    var_8 = 'end_position'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error in nested schema.'
    var_1 = module_0.Integer()
    var_2 = 'inner'
    var_3 = 'value'
    var_4 = 'not_int'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



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
    var_6 = None
    var_7 = 5
    var_8 = module_0.Token(var_6)
    var_9 = True
    var_10 = module_1.Field()
    var_11 = module_2.validate_with_positions(token=var_8, validator=var_10)
    var_12 = 'invalid'
    var_13 = 7
    var_14 = module_0.Token(var_12)
    var_15 = lambda x: x != var_12
    var_16 = module_1.Field()
    var_17 = module_2.validate_with_positions(token=var_14, validator=var_16)
    var_18 = 'name'
    var_19 = module_1.Field()
    var_20 = {var_18: var_19}
    var_21 = module_3.Schema(var_20)
    var_22 = {}
    var_23 = module_0.Token(var_22)
    var_24 = module_2.validate_with_positions(token=var_23, validator=var_21)
    var_25 = 'a'
    var_26 = 'b'
    var_27 = {var_25: var_6, var_26: var_6}
    var_28 = 20
    var_29 = module_0.Token(var_27)
    var_30 = module_1.Field()
    var_31 = module_1.Field()
    var_32 = {var_25: var_30, var_26: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = module_2.validate_with_positions(token=var_29, validator=var_33)



# Parsed testcases at query #11
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 10
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_2.Integer()
    var_17 = 'not_an_int'
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = module_0.Position(var_13, var_1, var_13)
    var_20 = module_1.Token(var_17)
    var_21 = module_3.validate_with_positions(token=var_20, validator=var_16)
    var_22 = {}
    var_23 = module_0.Position(var_1, var_1, var_1)
    var_24 = module_0.Position(var_3, var_1, var_3)
    var_25 = module_1.Token(var_22)
    var_26 = {}
    var_27 = module_0.Position(var_1, var_1, var_1)
    var_28 = module_0.Position(var_13, var_1, var_13)
    var_29 = module_1.Token(var_26)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 'age'
    var_7 = 25
    var_8 = {var_6: var_7}
    var_9 = 20
    var_10 = module_0.Token(var_8)
    var_11 = module_2.validate_with_positions(token=var_10, validator=var_0)
    var_12 = 'required'
    var_13 = 'start_position'
    var_14 = 'end_position'
    var_15 = module_1.Integer()
    var_16 = 'not_an_integer'
    var_17 = 5
    var_18 = module_0.Token(var_16)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_15)
    var_20 = module_2.validate_with_positions(token=var_18, validator=var_15)
    var_21 = module_1.String()
    var_22 = 'name'
    var_23 = 'John'
    var_24 = {var_22: var_23}
    var_25 = module_0.Token(var_24)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = 0
    var_3 = 10
    var_4 = 5
    var_5 = 15
    var_6 = 'nested'

def test_case_0():
    var_0 = 0
    var_1 = 7

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'val1'
    var_3 = 'val2'
    var_4 = 20
    var_5 = 30
    var_6 = 10
    var_7 = 15



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'nested'
    var_3 = 'field'
    var_4 = 5
    var_5 = 15
    var_6 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field'
    var_3 = 2
    var_4 = 8
    var_5 = [var_2]

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 10
    var_3 = 15
    var_4 = 5
    var_5 = 8



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'field1'
    var_4 = 'subfield'
    var_5 = 5
    var_6 = 15
    var_7 = 20
    var_8 = 8
    var_9 = 'field2'
    var_10 = [var_9]



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'parent'
    var_3 = 'field'

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 'field'

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 10
    var_3 = 15
    var_4 = 'field2'
    var_5 = 'field1'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 'name'
    var_2 = ''
    var_3 = 0
    var_4 = 10
    var_5 = 'age'
    var_6 = 'invalid'
    var_7 = 5
    var_8 = 15
    var_9 = 'field1'
    var_10 = 'field2'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 'field_name'

def test_case_0():
    var_0 = 10
    var_1 = 17

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = 20
    var_4 = 25
    var_5 = 10
    var_6 = 15

def test_case_0():
    var_0 = 'nested'
    var_1 = 'field'
    var_2 = None
    var_3 = {var_1: var_2}
    var_4 = 5
    var_5 = 10
    var_6 = [var_0, var_1]



# Parsed testcases at query #19
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 20
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_2.Integer()
    var_17 = 'not_an_int'
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = 10
    var_20 = module_0.Position(var_19, var_1, var_19)
    var_21 = module_1.Token(var_17)
    var_22 = module_3.validate_with_positions(token=var_21, validator=var_16)
    var_23 = {}
    var_24 = module_0.Position(var_1, var_1, var_1)
    var_25 = 2
    var_26 = module_0.Position(var_25, var_1, var_25)
    var_27 = module_1.Token(var_23)
    var_28 = {}
    var_29 = module_0.Position(var_1, var_1, var_1)
    var_30 = module_0.Position(var_19, var_1, var_19)
    var_31 = module_1.Token(var_28)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'nested'
    var_3 = 'field_name'
    var_4 = 5
    var_5 = 15
    var_6 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field'
    var_3 = 2
    var_4 = 8
    var_5 = [var_2]

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'nested'
    var_3 = 10
    var_4 = 20
    var_5 = 5
    var_6 = 15



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.Token(var_0)
    var_3 = False
    var_4 = module_1.Field(allow_null=var_3)
    var_5 = module_2.validate_with_positions(token=var_2, validator=var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = 5
    var_7 = module_2.Token(var_5)
    var_8 = module_3.validate_with_positions(token=var_7, validator=var_4)
    var_9 = 'required'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = module_0.Token(var_2)
    var_6 = module_1.Field()
    var_7 = {var_0: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_5, validator=var_8)
    var_10 = 'start_position'
    var_11 = 'end_position'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = module_0.Field(allow_null=var_2)
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = {}
    var_8 = 100
    var_9 = module_2.Token(var_7)
    var_10 = module_3.validate_with_positions(token=var_9, validator=var_6)
    var_11 = 'char_index'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'John'
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = 20
    var_8 = module_2.Token(var_5)
    var_9 = module_3.validate_with_positions(token=var_8, validator=var_3)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = 'c'
    var_5 = 0
    var_6 = 1
    var_7 = module_1.Token(var_4)
    var_8 = module_2.validate_with_positions(token=var_7, validator=var_3)
    var_9 = 'choice'



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 10
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = None
    var_17 = {var_9: var_16}
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = module_0.Position(var_13, var_1, var_13)
    var_20 = module_1.Token(var_17)
    var_21 = module_2.Integer()
    var_22 = 'not_an_int'
    var_23 = module_0.Position(var_1, var_1, var_1)
    var_24 = module_0.Position(var_13, var_1, var_13)
    var_25 = module_1.Token(var_22)
    var_26 = module_3.validate_with_positions(token=var_25, validator=var_21)
    var_27 = {}
    var_28 = module_0.Position(var_1, var_1, var_1)
    var_29 = 20
    var_30 = module_0.Position(var_29, var_1, var_29)
    var_31 = module_1.Token(var_27)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = module_1.String()
    var_7 = 'name'
    var_8 = 'John'
    var_9 = {var_7: var_8}
    var_10 = 20
    var_11 = module_0.Token(var_9)
    var_12 = module_2.validate_with_positions(token=var_11, validator=var_4)
    var_13 = None
    var_14 = {var_7: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = 'obj'
    var_17 = 'char_index'
    var_18 = {var_17: var_1}
    var_19 = {var_17: var_10}
    var_20 = module_2.validate_with_positions(token=var_15, validator=var_4)
    var_21 = 'not_an_integer'
    var_22 = 14
    var_23 = module_0.Token(var_21)
    var_24 = {var_17: var_1}
    var_25 = {var_17: var_22}
    var_26 = module_1.Integer()
    var_27 = module_2.validate_with_positions(token=var_23, validator=var_26)
    var_28 = 'field1'
    var_29 = 'field2'
    var_30 = 'invalid'
    var_31 = {var_28: var_13, var_29: var_30}
    var_32 = 30
    var_33 = module_0.Token(var_31)
    var_34 = {var_17: var_1}
    var_35 = {var_17: var_32}
    var_36 = module_2.validate_with_positions(token=var_33, validator=var_26)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = 20
    var_9 = module_1.Token(var_6)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_a_number'
    var_1 = 0
    var_2 = 12
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Integer()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    var_6 = 'start_position'
    var_7 = 'end_position'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = 'name'
    var_3 = 'address'
    var_4 = 'John'
    var_5 = 'city'
    var_6 = 'NYC'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = 0
    var_10 = 30
    var_11 = module_1.Token(var_8)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.Integer()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = 'required'

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = 50
    var_9 = module_1.Token(var_6)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Integer()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'invalid'
    var_5 = 'also_invalid'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = 100
    var_9 = module_1.Token(var_6)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    var_6 = 'text'
    var_7 = 'code'
    var_8 = 'start_position'
    var_9 = 'end_position'



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test successful validation returns the validated value.'
    var_1 = 'valid_string'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error with Field validator.'
    var_1 = 'x'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = 10
    var_5 = module_0.String(max_length=var_4)
    var_6 = 'start_position'
    var_7 = 'end_position'

def test_case_0():
    var_0 = 'Test validation error with required field missing.'
    var_1 = {}
    var_2 = 'required'

def test_case_0():
    var_0 = 'Test that error messages are sorted by character position.'
    var_1 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validation error with Integer field.'
    var_1 = 'not_an_integer'
    var_2 = module_0.Integer()
    var_3 = 'start_position'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that positional messages have all required attributes.'
    var_1 = 'invalid'
    var_2 = module_0.Integer()
    var_3 = 'text'
    var_4 = 'code'
    var_5 = 'index'
    var_6 = 'start_position'
    var_7 = 'end_position'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'
    var_3 = 'subfield'

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 'field1'

def test_case_0():
    var_0 = 20
    var_1 = 25
    var_2 = 5
    var_3 = 10
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'nested'
    var_5 = 'field'

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 10
    var_3 = 'field1'
    var_4 = 'field2'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'
    var_3 = 'field2'

def test_case_0():
    var_0 = 5
    var_1 = 15
    var_2 = 'field1'

def test_case_0():
    var_0 = 20
    var_1 = 25
    var_2 = 5
    var_3 = 10
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'subfield'



# Parsed testcases at query #30
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 15
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = module_3.validate_with_positions(token=var_15, validator=var_6)
    var_17 = {}
    var_18 = module_0.Position(var_1, var_1, var_1)
    var_19 = 2
    var_20 = module_0.Position(var_19, var_1, var_19)
    var_21 = module_1.Token(var_17)
    var_22 = module_3.validate_with_positions(token=var_21, validator=var_6)
    var_23 = 'not_an_int'
    var_24 = module_0.Position(var_1, var_1, var_1)
    var_25 = 10
    var_26 = module_0.Position(var_25, var_1, var_25)
    var_27 = module_1.Token(var_23)
    var_28 = module_2.Integer()
    var_29 = module_3.validate_with_positions(token=var_27, validator=var_28)
    var_30 = {}
    var_31 = module_0.Position(var_1, var_1, var_1)
    var_32 = module_0.Position(var_19, var_1, var_19)
    var_33 = module_1.Token(var_30)
    var_34 = module_3.validate_with_positions(token=var_33, validator=var_28)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = False

def test_case_0():
    var_0 = 'test_value'
    var_1 = True
    var_2 = 'invalid'
    var_3 = 'field'
    var_4 = [var_3]
    var_5 = 0

def test_case_0():
    var_0 = 'test_value'
    var_1 = True
    var_2 = 'required'
    var_3 = 'parent'
    var_4 = 'field'
    var_5 = [var_3, var_4]
    var_6 = 0

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'Error 1'
    var_2 = 'invalid'
    var_3 = 'field1'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_1, code=var_2, index=var_4)
    var_6 = 'Error 2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_6, code=var_2, index=var_8)
    var_10 = [var_5, var_9]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'Error 2'
    var_2 = 'invalid'
    var_3 = 'field2'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_1, code=var_2, index=var_4)
    var_6 = 'Error 1'
    var_7 = 'field1'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_6, code=var_2, index=var_8)
    var_10 = [var_5, var_9]



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'valid_value'
    var_1 = 'name'
    var_2 = ''
    var_3 = 0
    var_4 = 10
    var_5 = 'field'
    var_6 = 'age'
    var_7 = 'invalid'
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = 5
    var_11 = 15
    var_12 = 20
    var_13 = 30
    var_14 = [var_9]



# Parsed testcases at query #33
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_1)
    var_4 = 10
    var_5 = 11
    var_6 = module_0.Position(var_2, var_4)
    var_7 = module_1.Token(var_0)
    var_8 = module_2.String()
    var_9 = module_3.validate_with_positions(token=var_7, validator=var_8)
    assert var_9 == 'test_value'
    var_10 = module_2.String()
    var_11 = 'name'
    var_12 = 'John'
    var_13 = {var_11: var_12}
    var_14 = module_0.Position(var_2, var_1)
    var_15 = 20
    var_16 = 21
    var_17 = module_0.Position(var_2, var_15)
    var_18 = module_1.Token(var_13)
    var_19 = {}
    var_20 = module_0.Position(var_2, var_1)
    var_21 = module_0.Position(var_2, var_4)
    var_22 = module_1.Token(var_19)
    var_23 = 'required'
    var_24 = 'not_an_int'
    var_25 = module_0.Position(var_2, var_1)
    var_26 = module_0.Position(var_2, var_4)
    var_27 = module_1.Token(var_24)
    var_28 = module_2.Integer()
    var_29 = module_3.validate_with_positions(token=var_27, validator=var_28)
    var_30 = 'start_position'
    var_31 = 'end_position'
    var_32 = 'a'
    var_33 = 'invalid'
    var_34 = {var_32: var_33}
    var_35 = module_0.Position(var_2, var_1)
    var_36 = module_0.Position(var_2, var_15)
    var_37 = module_1.Token(var_34)
    var_38 = module_2.Integer()



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field1'
    var_3 = 'subfield'
    var_4 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = [var_4]

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'validated_value'
    var_3 = {var_0: var_1}



# Parsed testcases at query #35
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'text'
    var_1 = 'hello'
    var_2 = 0
    var_3 = module_0.Position(var_2)
    var_4 = 5
    var_5 = module_0.Position(var_4)
    var_6 = module_1.Token(var_1)
    var_7 = module_2.String()
    var_8 = module_3.validate_with_positions(token=var_6, validator=var_7)
    assert var_8 == 'hello'
    var_9 = module_2.String()
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = module_0.Position(var_2)
    var_14 = 10
    var_15 = module_0.Position(var_14)
    var_16 = module_1.Token(var_12)
    var_17 = module_3.validate_with_positions(token=var_16, validator=var_7)
    var_18 = None
    var_19 = {var_10: var_18}
    var_20 = module_0.Position(var_2)
    var_21 = module_0.Position(var_14)
    var_22 = module_1.Token(var_19)
    var_23 = module_3.validate_with_positions(token=var_22, validator=var_7)
    var_24 = list(error.messages())
    var_25 = 'not_an_integer'
    var_26 = module_0.Position(var_2)
    var_27 = 14
    var_28 = module_0.Position(var_27)
    var_29 = module_1.Token(var_25)
    var_30 = module_2.Integer()
    var_31 = module_3.validate_with_positions(token=var_29, validator=var_30)
    var_32 = list(error.messages())
    var_33 = module_2.Integer()
    var_34 = module_2.Integer()
    var_35 = 'field1'
    var_36 = 'field2'
    var_37 = 'bad'
    var_38 = 'also_bad'
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = module_0.Position(var_2)
    var_41 = 30
    var_42 = module_0.Position(var_41)
    var_43 = module_1.Token(var_39)
    var_44 = module_3.validate_with_positions(token=var_43, validator=var_30)
    var_45 = list(error.messages())



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'test_value'
    var_6 = 123
    var_7 = module_0.Token(var_6)
    var_8 = module_1.String()
    var_9 = module_2.validate_with_positions(token=var_7, validator=var_8)
    var_10 = module_1.String()
    var_11 = module_1.Integer()
    var_12 = 'age'
    var_13 = 25
    var_14 = {var_12: var_13}
    var_15 = 5
    var_16 = 20
    var_17 = module_0.Token(var_14)
    var_18 = module_1.String()
    var_19 = module_1.Integer()
    var_20 = module_1.String()
    var_21 = {}
    var_22 = 50
    var_23 = module_0.Token(var_21)
    var_24 = module_1.String()
    var_25 = 'name'
    var_26 = 'John'
    var_27 = {var_25: var_26}
    var_28 = module_0.Token(var_27)



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'test_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'field1'
    var_4 = 'nested_field'
    var_5 = 5
    var_6 = 12
    var_7 = 1
    var_8 = 'field'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10
    var_3 = 'nested'
    var_4 = 'field'
    var_5 = 'name'
    var_6 = 'invalid'
    var_7 = 5
    var_8 = 15
    var_9 = 20
    var_10 = 30
    var_11 = 'field1'
    var_12 = 'field2'



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = 'obj'
    var_5 = 'char_index'
    var_6 = {var_5: var_1}
    var_7 = {var_5: var_2}
    var_8 = module_1.String()
    var_9 = module_2.validate_with_positions(token=var_3, validator=var_8)
    assert var_9 == 'hello'
    var_10 = '123'
    var_11 = 3
    var_12 = module_0.Token(var_10)
    var_13 = {var_5: var_1}
    var_14 = {var_5: var_11}
    var_15 = module_1.Integer()
    var_16 = module_2.validate_with_positions(token=var_12, validator=var_15)
    assert var_16 == 123
    var_17 = 'not_a_number'
    var_18 = 12
    var_19 = module_0.Token(var_17)
    var_20 = {var_5: var_1}
    var_21 = {var_5: var_18}
    var_22 = module_1.Integer()
    var_23 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_24 = 'start_position'
    var_25 = 'end_position'
    var_26 = module_1.String()
    var_27 = {}
    var_28 = 2
    var_29 = module_0.Token(var_27)
    var_30 = {var_5: var_1}
    var_31 = {var_5: var_28}
    var_32 = 'invalid'
    var_33 = 7
    var_34 = module_0.Token(var_32)
    var_35 = {var_5: var_1}
    var_36 = {var_5: var_33}
    var_37 = module_1.Integer()
    var_38 = module_2.validate_with_positions(token=var_34, validator=var_37)
    var_39 = list(error.messages())
    var_40 = [m.start_position.char_index for m in var_39]



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0)
    var_4 = 'obj'
    var_5 = 'char_index'
    var_6 = {var_5: var_1}
    var_7 = {var_5: var_2}
    var_8 = module_1.String()
    var_9 = module_2.validate_with_positions(token=var_3, validator=var_8)
    assert var_9 == 'test'
    var_10 = module_1.String()
    var_11 = 'name'
    var_12 = 'John'
    var_13 = {var_11: var_12}
    var_14 = 20
    var_15 = module_0.Token(var_13)
    var_16 = {var_5: var_1}
    var_17 = {var_5: var_14}
    var_18 = 'not_an_int'
    var_19 = 10
    var_20 = module_0.Token(var_18)
    var_21 = {var_5: var_1}
    var_22 = {var_5: var_19}
    var_23 = module_1.Integer()
    var_24 = module_2.validate_with_positions(token=var_20, validator=var_23)
    var_25 = {}
    var_26 = 5
    var_27 = module_0.Token(var_25)
    var_28 = {var_5: var_1}
    var_29 = {var_5: var_26}



# Parsed testcases at query #41
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_1, var_3, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'

import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 10
    var_4 = module_0.Position(var_1, var_3, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.Integer()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = {}
    var_2 = 0
    var_3 = module_1.Position(var_2, var_2, var_2)
    var_4 = 2
    var_5 = module_1.Position(var_2, var_4, var_4)
    var_6 = module_2.Token(var_1)
    var_7 = True

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = module_1.Position(var_7, var_7, var_7)
    var_9 = module_1.Position(var_7, var_5, var_5)
    var_10 = module_2.Token(var_6)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 123
    var_5 = 456
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0
    var_8 = module_1.Position(var_7, var_7, var_7)
    var_9 = 50
    var_10 = module_1.Position(var_7, var_9, var_9)
    var_11 = module_2.Token(var_6)

import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 4
    var_4 = module_0.Position(var_1, var_3, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = 2
    var_7 = module_2.String(max_length=var_6)
    var_8 = module_3.validate_with_positions(token=var_5, validator=var_7)
    var_9 = 'text'
    var_10 = 'code'
    var_11 = 'start_position'
    var_12 = 'end_position'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'parent'
    var_3 = 'field_name'
    var_4 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'field'
    var_5 = [var_4]

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 5
    var_3 = module_0.Token(var_0)
    var_4 = module_1.String()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)
    assert var_5 == 'hello'
    var_6 = module_1.String()
    var_7 = 'name'
    var_8 = 'John'
    var_9 = {var_7: var_8}
    var_10 = 15
    var_11 = module_0.Token(var_9)
    var_12 = module_2.validate_with_positions(token=var_11, validator=var_4)
    var_13 = 'not_an_int'
    var_14 = 10
    var_15 = module_0.Token(var_13)
    var_16 = module_1.Integer()
    var_17 = module_2.validate_with_positions(token=var_15, validator=var_16)
    var_18 = {}
    var_19 = module_0.Token(var_18)
    var_20 = module_2.validate_with_positions(token=var_19, validator=var_16)
    var_21 = 'invalid'
    var_22 = 7
    var_23 = module_0.Token(var_21)
    var_24 = module_1.Integer()
    var_25 = module_2.validate_with_positions(token=var_23, validator=var_24)



# Parsed testcases at query #44
#--------------------------


import typesystem.base as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.fields as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = module_0.Position(var_1, var_1, var_1)
    var_3 = 5
    var_4 = module_0.Position(var_3, var_1, var_3)
    var_5 = module_1.Token(var_0)
    var_6 = module_2.String()
    var_7 = module_3.validate_with_positions(token=var_5, validator=var_6)
    assert var_7 == 'hello'
    var_8 = module_2.String()
    var_9 = 'name'
    var_10 = 'John'
    var_11 = {var_9: var_10}
    var_12 = module_0.Position(var_1, var_1, var_1)
    var_13 = 10
    var_14 = module_0.Position(var_13, var_1, var_13)
    var_15 = module_1.Token(var_11)
    var_16 = 'not_a_number'
    var_17 = module_0.Position(var_1, var_1, var_1)
    var_18 = 12
    var_19 = module_0.Position(var_18, var_1, var_18)
    var_20 = module_1.Token(var_16)
    var_21 = module_2.Integer()
    var_22 = module_3.validate_with_positions(token=var_20, validator=var_21)
    var_23 = None
    var_24 = {var_9: var_23}
    var_25 = module_0.Position(var_1, var_1, var_1)
    var_26 = module_0.Position(var_13, var_1, var_13)
    var_27 = module_1.Token(var_24)
    var_28 = 'invalid'
    var_29 = module_0.Position(var_1, var_1, var_1)
    var_30 = 7
    var_31 = module_0.Position(var_30, var_1, var_30)
    var_32 = module_1.Token(var_28)
    var_33 = module_2.Integer()
    var_34 = module_3.validate_with_positions(token=var_32, validator=var_33)



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 'field_name'

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 5
    var_3 = 15
    var_4 = 'field'

def test_case_0():
    var_0 = 20
    var_1 = 30
    var_2 = 5
    var_3 = 15
    var_4 = 'field1'
    var_5 = 'field2'



