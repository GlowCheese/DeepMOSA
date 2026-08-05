####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/8 statements.
# Partially parsed test_validate_with_positions_error_mapping_type. Retrieved 8/13 statements.
# Partially parsed test_validate_with_positions_required_error_logic. Retrieved 10/15 statements.
# Partially parsed test_validate_with_positions_sorting_by_index. Retrieved 11/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'val'
    var_3 = 'ok'

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'val'
    var_3 = 'Not a string'
    var_4 = 'type'
    var_5 = module_0.Message(text=var_3, code=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].text
    assert var_9 == 'Not a string'
    var_10 = e.messages()[0].start_position

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'val'
    var_3 = 'Missing'
    var_4 = 'required'
    var_5 = 'field_name'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_3, code=var_4, index=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].text
    assert var_11 == "The field 'field_name' is required."

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 0
    var_3 = 'val'
    var_4 = 'Second error'
    var_5 = 'type'
    var_6 = module_0.Message(text=var_4, code=var_5)
    var_7 = 'First error'
    var_8 = module_0.Message(text=var_7, code=var_5)
    var_9 = [var_6, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = len(e.messages())
    assert var_11 == 2



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/10 statements.
# Partially parsed test_validate_with_positions_validation_error_with_lookup. Retrieved 17/34 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 15/31 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 23/41 statements.


def test_case_0():
    var_0 = 'some_value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error text'
    var_1 = 'type'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 123
    var_8 = 10
    var_9 = 15
    var_10 = 0
    var_11 = raised_error.messages()[var_10]
    var_12 = var_11.text
    assert var_12 == 'error text'
    var_13 = raised_error.messages()[var_10]
    var_14 = var_13.start_position
    var_15 = raised_error.messages()[var_10]
    var_16 = var_15.end_position

import typesystem.base as module_0

def test_case_0():
    var_0 = 'original text'
    var_1 = 'required'
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = 0
    var_9 = 5
    var_10 = raised_error.messages()[var_8]
    var_11 = var_10.text
    assert var_11 == "The field 'child' is required."
    var_12 = raised_error.messages()[var_8]
    var_13 = var_12.start_position
    var_14 = [var_2]

import typesystem.base as module_0

def test_case_0():
    var_0 = 'first'
    var_1 = 'type'
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'second'
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_8, var_4]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = 50
    var_12 = 60
    var_13 = [var_2, var_6]
    var_14 = lambda : var_13
    var_15 = 10
    var_16 = 20
    var_17 = 0
    var_18 = raised_error.messages()[var_17]
    var_19 = var_18.text
    assert var_19 == 'second'
    var_20 = 1
    var_21 = raised_error.messages()[var_20]
    var_22 = var_21.text
    assert var_22 == 'first'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 3



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 9/18 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 11/26 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 4
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = error.messages()[var_1]
    var_6 = var_5.text
    assert var_6 == 'Wrong type'
    var_7 = error.messages()[var_1]
    var_8 = var_7.start_position.char_index
    assert var_8 == 0

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'age'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = error.messages()[var_1]
    var_10 = var_9.text
    assert var_10 == "The field 'age' is required."



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 12/27 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 7/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 4
    var_5 = "{'a': 1}"
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 6
    var_3 = "{'b': 1}"
    var_4 = 'b'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = 7
    var_8 = module_0.Field()
    var_9 = {var_4: var_8}
    var_10 = 'b'
    var_11 = 'ValidationError was not raised'
    var_12 = AssertionError(var_11)

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '[0]'
    var_3 = [var_1]
    var_4 = 2
    var_5 = 'ValidationError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_error_type_error. Retrieved 1/14 statements.
# Partially parsed test_validate_with_positions_error_required. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 2/27 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'

def test_case_0():
    var_0 = 0
    var_1 = 'user'
    var_2 = 'name'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 1/32 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/25 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 4/25 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 5/29 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 1
    var_3 = '10'

def test_case_0():
    var_0 = 'not an int'
    var_1 = 0
    var_2 = 9
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0]
    var_5 = var_4.code
    assert var_5 == 'type'
    var_6 = var_4.text
    assert var_6 == 'Not an int'
    var_7 = var_4.start_position.char_index
    assert var_7 == 0
    var_8 = var_4.end_position.char_index
    assert var_8 == 9

def test_case_0():
    var_0 = 'username'
    var_1 = {}
    var_2 = 0
    var_3 = '{}'
    var_4 = error.messages()[0]
    var_5 = var_4.code
    assert var_5 == 'required'
    var_6 = var_4.text
    assert var_6 == "The field 'username' is required."
    var_7 = var_4.index
    var_8 = bool(var_4.index == ['username'])
    assert var_8 is True

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/15 statements.
# Failed to parse test_validate_with_positions_raises_validation_error.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/6 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 8/17 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'ok'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'error'
    var_1 = 0
    var_2 = 4
    var_3 = 'sub'
    var_4 = 2
    var_5 = 'key'
    var_6 = error.messages()[0]
    var_7 = var_6.code
    assert var_7 == 'type'
    var_8 = var_6.text
    assert var_8 == 'err'
    var_9 = var_6.start_position.char_index
    assert var_9 == 0
    var_10 = var_6.end_position.char_index
    assert var_10 == 2

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = ''
    var_4 = 1
    var_5 = 'missing'
    var_6 = 'required_field'
    var_7 = error.messages()[0]
    var_8 = var_7.code
    assert var_8 == 'required'
    var_9 = 'required_field'
    var_10 = bool('required_field' in var_7.text)
    assert var_10 is True
    var_11 = var_7.start_position.char_index
    assert var_11 == 1
    var_12 = var_7.end_position.char_index
    assert var_12 == 1

def test_case_0():
    var_0 = 'error'
    var_1 = 0
    var_2 = 4
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_required_error_mapping. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_type_error_mapping. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 11/27 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Missing'
    var_1 = 'required'
    var_2 = 'username'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid type'
    var_1 = 'type'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Err 1'
    var_1 = 'type'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Err 2'
    var_6 = 0
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.ValidationError(messages=var_9)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_catches_validation_error. Retrieved 3/27 statements.


def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 3
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].code
    assert var_4 == 'type'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 8/24 statements.
# Partially parsed test_validate_with_positions_sorting_by_position. Retrieved 7/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 4
    var_5 = '{"a":1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"name": "John"}'
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 13
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'age'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 4
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 3
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].text
    assert var_4 == 'Error 1'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_validation_error_with_lookup. Retrieved 14/28 statements.
# Partially parsed test_validate_with_positions_validation_error_with_index_lookup. Retrieved 19/34 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 19/42 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Required'
    var_1 = 'required'
    var_2 = 'fields'
    var_3 = 'name'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = 1
    var_9 = 0
    var_10 = 5
    var_11 = 4
    var_12 = 'ValidationError was not raised'
    var_13 = AssertionError(var_12)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid type'
    var_1 = 'type'
    var_2 = 'user'
    var_3 = 'id'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = 'not-int'
    var_9 = {var_3: var_8}
    var_10 = 1
    var_11 = 10
    var_12 = 9
    var_13 = 18
    var_14 = 17
    var_15 = 6
    var_16 = 5
    var_17 = 'ValidationError was not raised'
    var_18 = AssertionError(var_17)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 2'
    var_1 = 'type'
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 1'
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_8, var_4]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 5
    var_14 = 4
    var_15 = 6
    var_16 = 0
    var_17 = 'ValidationError was not raised'
    var_18 = AssertionError(var_17)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_exception_handling. Retrieved 9/45 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 5
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

def test_case_0():
    var_0 = 'Position'
    var_1 = 'line_no'
    var_2 = 'column_no'
    var_3 = 'char_index'
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = 0
    var_7 = 4
    var_8 = '{"a": 1}'
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].text
    assert var_10 == 'Error'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 3/19 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 3/19 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 3/19 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = '{}'

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = 'not a dict'

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = '{}'

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = "{'a':1,'b':2}"



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 9/31 statements.
# Partially parsed test_validate_with_positions_error_with_index_lookup. Retrieved 16/49 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 7/34 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "John"}'
    var_6 = {}
    var_7 = {}
    var_8 = 14

def test_case_0():
    var_0 = 'age'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"age": 30}'
    var_6 = {}
    var_7 = {}
    var_8 = 14
    var_9 = ''
    var_10 = {}
    var_11 = None
    var_12 = {var_0: var_11}
    var_13 = 1
    var_14 = 4
    var_15 = error.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'type'
    var_17 = var_15.text
    assert var_17 == 'Too old'
    var_18 = var_15.start_position.char_index
    assert var_18 == 1
    var_19 = var_15.end_position.char_index
    assert var_19 == 4

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = {}
    var_4 = {}
    var_5 = 1
    var_6 = error.messages()[0]
    var_7 = var_6.code
    assert var_7 == 'required'
    var_8 = 'name'
    var_9 = bool('name' in var_6.text)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_catches_validation_error. Retrieved 6/34 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 5
    var_5 = '{"a": 1}'
    var_6 = len(error.messages())
    assert var_6 == 1
    var_7 = error.messages()[0].code
    assert var_7 == 'type'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/12 statements.
# Partially parsed test_validate_with_positions_validation_error. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'data'

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].code
    assert var_4 == 'type'

def test_case_0():
    var_0 = {}
    var_1 = 'field_name'
    var_2 = bool('field_name' in error.messages()[0].text)
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/14 statements.
# Partially parsed test_validate_with_positions_error_with_lookup. Retrieved 4/33 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 4/32 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = e.messages()[0]
    var_4 = var_3.text
    assert var_4 == 'error'
    var_5 = var_3.code
    assert var_5 == 'type'

def test_case_0():
    var_0 = 'other'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = e.messages()[0]
    var_4 = var_3.text
    assert var_4 == "The field 'username' is required."
    var_5 = var_3.code
    assert var_5 == 'required'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 9/19 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 11/21 statements.
# Partially parsed test_validate_with_positions_sorting_of_errors. Retrieved 18/31 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Must be an object.'
    var_1 = 'type'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.ValidationError(messages=var_3)
    var_5 = 1
    var_6 = 0
    var_7 = 10
    var_8 = 9

import typesystem.base as module_0

def test_case_0():
    var_0 = 'This field is required.'
    var_1 = 'required'
    var_2 = 'missing_key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = 20
    var_10 = 19

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'Err 2'
    var_5 = 'type'
    var_6 = [var_1]
    var_7 = module_0.Message(text=var_4, code=var_5, index=var_6)
    var_8 = 'Err 1'
    var_9 = [var_0]
    var_10 = module_0.Message(text=var_8, code=var_5, index=var_9)
    var_11 = [var_7, var_10]
    var_12 = module_0.ValidationError(messages=var_11)
    var_13 = 0
    var_14 = 5
    var_15 = 4
    var_16 = 6
    var_17 = [var_0]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 9/19 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 11/21 statements.
# Partially parsed test_validate_with_positions_sorting_of_messages. Retrieved 16/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Must be an object.'
    var_1 = 'type'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.ValidationError(messages=var_3)
    var_5 = 1
    var_6 = 0
    var_7 = 10
    var_8 = 9

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Required'
    var_1 = 'required'
    var_2 = 'missing_field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = 5
    var_10 = 4

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 2'
    var_1 = 'type'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = 'Error 1'
    var_4 = module_0.Message(text=var_3, code=var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = 5
    var_10 = 4
    var_11 = 10
    var_12 = 9
    var_13 = 15
    var_14 = 14
    var_15 = module_0.Position(var_7, var_13, var_14)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/25 statements.
# Partially parsed test_validate_with_positions_validation_error_with_lookup. Retrieved 4/39 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 4/39 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'input'
    var_1 = 0
    var_2 = 4
    var_3 = e.messages()[0]
    var_4 = var_3.text
    assert var_4 == 'Bad value'
    var_5 = var_3.code
    assert var_5 == 'type'
    var_6 = var_3.start_position.char_index
    assert var_6 == 10
    var_7 = var_3.end_position.char_index
    assert var_7 == 20

def test_case_0():
    var_0 = 'input'
    var_1 = 0
    var_2 = 4
    var_3 = e.messages()[0]
    var_4 = var_3.text
    assert var_4 == "The field 'username' is required."
    var_5 = var_3.code
    assert var_5 == 'required'
    var_6 = var_3.start_position.char_index
    assert var_6 == 5



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_validation_error_triggering_try_except. Retrieved 3/24 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 4
    var_5 = '{"a":1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '{}'
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].text
    assert var_4 == 'error'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 6/24 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 7/36 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = 'hello'
    var_3 = 4
    var_4 = 6
    var_5 = 5

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = None
    var_5 = 5
    var_6 = 'name'

def test_case_0():
    var_0 = 'not an int'
    var_1 = 0
    var_2 = 8



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/12 statements.
# Partially parsed test_validate_with_positions_error_handling. Retrieved 1/27 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = len(error.messages())
    assert var_1 == 1
    var_2 = error.messages()[0].text
    assert var_2 == 'error'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/18 statements.
# Partially parsed test_validate_with_tokens_error_type. Retrieved 2/60 statements.
# Partially parsed test_validate_with_positions_required_logic. Retrieved 2/23 statements.
# Partially parsed test_validate_with_positions_type_error_logic. Retrieved 2/22 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'typesystem.tokenize.positional_validation'

def test_case_0():
    var_0 = 'some_value'
    var_1 = e.messages()[0]
    var_2 = var_1.text
    assert var_2 == "The field 'key' is required."
    var_3 = var_1.code
    assert var_3 == 'required'
    var_4 = var_1.index
    var_5 = bool(var_1.index == ['key'])
    assert var_5 is True

def test_case_0():
    var_0 = 'some_value'
    var_1 = e.messages()[0]
    var_2 = var_1.text
    assert var_2 == 'wrong type'
    var_3 = var_1.code
    assert var_3 == 'type'
    var_4 = var_1.index
    var_5 = bool(var_1.index == ['sub'])
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_positions_handles_validation_error. Retrieved 9/22 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'Error text'
    var_3 = 'type'
    var_4 = 0
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_2, code=var_3, index=var_5)
    var_7 = [var_6]
    var_8 = module_0.ValidationError(messages=var_7)
    var_9 = len(caught_error.messages())
    assert var_9 == 1
    var_10 = caught_error.messages()[0].text
    assert var_10 == 'Error text'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 6/22 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = error.messages()[0]
    var_4 = var_3.code
    assert var_4 == 'required'
    var_5 = "The field 'name' is required."
    var_6 = bool("The field 'name' is required." in var_3.text)
    assert var_6 is True
    var_7 = var_3.start_position.line
    assert var_7 == 1
    var_8 = 'ValidationError not raised'
    var_9 = AssertionError(var_8)

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = error.messages()[0]
    var_5 = var_4.code
    assert var_5 == 'type'
    var_6 = var_4.text
    assert var_6 == 'Not a string'
    var_7 = var_4.start_position.char_index
    assert var_7 == 0
    var_8 = var_4.end_position.char_index
    assert var_8 == 2
    var_9 = 'ValidationError not raised'
    var_10 = AssertionError(var_9)



