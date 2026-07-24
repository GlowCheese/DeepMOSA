####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 5/22 statements.
# Partially parsed test_validate_with_positions_required_error_with_lookup. Retrieved 6/20 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 8
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].text
    assert var_4 == 'Not an int'
    var_5 = error.messages()[0].start_position.char_index
    assert var_5 == 0
    var_6 = error.messages()[0].end_position.char_index
    assert var_6 == 8
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0]
    var_5 = var_4.text
    assert var_5 == "The field 'username' is required."
    var_6 = var_4.code
    assert var_6 == 'required'
    var_7 = var_4.index
    var_8 = bool(var_4.index == ['username'])
    assert var_8 is True
    var_9 = 'ValidationError not raised'
    var_10 = AssertionError(var_9)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 'ab'
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/19 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 9/27 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 14/33 statements.


def test_case_0():
    var_0 = 'valid_value'
    var_1 = 0
    var_2 = 10

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Wrong type'
    var_1 = 'type'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = 'invalid_value'
    var_5 = 0
    var_6 = 12
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'This field is required.'
    var_1 = 'required'
    var_2 = 'missing_key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = 'other'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = 10
    var_11 = '{"other": 1}'
    var_12 = 'ValidationError not raised'
    var_13 = AssertionError(var_12)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 3/44 statements.


def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = 'val'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 6/23 statements.
# Partially parsed test_validate_with_positions_required_error_handling. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = len(error.messages())
    assert var_4 == 1
    var_5 = error.messages()[0].text
    assert var_5 == 'Wrong type'
    var_6 = error.messages()[0].start_position
    var_7 = bool(error.messages()[0].start_position == Position(1, 1, 0))
    assert var_7 is True
    var_8 = error.messages()[0].end_position
    var_9 = bool(error.messages()[0].end_position == Position(1, 1, 2))
    assert var_9 is True
    var_10 = 'ValidationError not raised'
    var_11 = AssertionError(var_10)

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = len(error.messages())
    assert var_4 == 1
    var_5 = error.messages()[0].text
    assert var_5 == "The field 'user' is required."
    var_6 = error.messages()[0].code
    assert var_6 == 'required'
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/20 statements.
# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 6/23 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = len(error.messages())
    assert var_4 == 1
    var_5 = error.messages()[0].code
    assert var_5 == 'type'
    var_6 = error.messages()[0].start_position
    var_7 = bool(error.messages()[0].start_position == Position(1, 1, 0))
    assert var_7 is True
    var_8 = error.messages()[0].end_position
    var_9 = bool(error.messages()[0].end_position == Position(1, 1, 2))
    assert var_9 is True
    var_10 = 'ValidationError not raised'
    var_11 = AssertionError(var_10)

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = error.messages()[0].text
    assert var_4 == "The field 'username' is required."
    var_5 = error.messages()[0].code
    assert var_5 == 'required'
    var_6 = error.messages()[0].index
    var_7 = bool(error.messages()[0].index == ['username'])
    assert var_7 is True
    var_8 = 'ValidationError not raised'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error_on_failure. Retrieved 5/33 statements.


def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 6
    var_3 = 'ValidationError was not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 5/22 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 6/25 statements.
# Partially parsed test_validate_with_positions_nested_lookup. Retrieved 10/32 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 8
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].code
    assert var_4 == 'type'
    var_5 = error.messages()[0].text
    assert var_5 == 'Not an int'
    var_6 = error.messages()[0].start_position.char_index
    assert var_6 == 0
    var_7 = error.messages()[0].end_position.char_index
    assert var_7 == 8
    var_8 = 'ValidationError not raised'
    var_9 = AssertionError(var_8)

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = 'name'
    var_4 = len(error.messages())
    assert var_4 == 1
    var_5 = error.messages()[0].code
    assert var_5 == 'required'
    var_6 = error.messages()[0].text
    assert var_6 == "The field 'name' is required."
    var_7 = error.messages()[0].index
    var_8 = bool(error.messages()[0].index == ['name'])
    assert var_8 is True
    var_9 = 'ValidationError not raised'
    var_10 = AssertionError(var_9)

def test_case_0():
    var_0 = 'val'
    var_1 = 5
    var_2 = 7
    var_3 = '{}val'
    var_4 = {}
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 0
    var_8 = error.messages()[0].start_position.char_index
    assert var_8 == 5
    var_9 = error.messages()[0].end_position.char_index
    assert var_9 == 7
    var_10 = 'ValidationError not raised'
    var_11 = AssertionError(var_10)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 2/7 statements.
# Failed to parse test_validate_with_tokens_success_value.
# Partially parsed test_validate_with_positions_validation_error_with_index_lookup. Retrieved 6/23 statements.
# Partially parsed test_validate_with_positions_validation_error_with_direct_index_lookup. Retrieved 6/23 statements.
# Partially parsed test_validate_with_positions_sorting_of_messages. Retrieved 9/35 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = 'ValidationError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 0
    var_3 = 2
    var_4 = 'ValidationError not raised'
    var_5 = AssertionError(var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 5
    var_5 = module_0.Position(var_2, var_4, var_4)
    var_6 = module_0.Position(var_2, var_2, var_2)
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_raises_validation_error_and_reconstructs_messages. Retrieved 4/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '{}'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = len(error.messages())
    assert var_4 == 1
    var_5 = error.messages()[0].text
    assert var_5 == 'Error 1'
    var_6 = error.messages()[0].code
    assert var_6 == 'type'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_validation_error_with_index. Retrieved 9/16 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 7/14 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 15/33 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_0)
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = 'ValidationError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = '{"a": 1, "b": 2}'
    var_8 = None
    var_9 = 3
    var_10 = 11
    var_11 = {var_0: var_2, var_1: var_3}
    var_12 = 15
    var_13 = 'ValidationError not raised'
    var_14 = AssertionError(var_13)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 18/28 statements.
# Partially parsed test_validate_with_positions_type_error_mapping. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_nested_index_lookup. Retrieved 18/30 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'a': 1})
    assert var_12 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = '{"a": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'b'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = {}
    var_13 = '{}'
    var_14 = module_1.Field()
    var_15 = {var_7: var_14}
    var_16 = {}
    var_17 = module_2.Schema(var_15, **var_16)
    var_18 = module_3.validate_with_positions(token=var_6, validator=var_17)
    var_19 = len(error.messages())
    assert var_19 == 1
    var_20 = error.messages()[0]
    var_21 = var_20.code
    assert var_21 == 'required'
    var_22 = 'is required'
    var_23 = bool('is required' in var_20.text)
    assert var_23 is True
    var_24 = var_20.index
    var_25 = bool(var_20.index == ['b'])
    assert var_25 is True
    var_26 = var_20.start_position.char_index
    assert var_26 == 0
    var_27 = var_20.end_position.char_index
    assert var_27 == 0

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = {}
    var_5 = {}
    var_6 = module_0.Schema(var_4, **var_5)
    var_7 = error.messages()[0]
    var_8 = var_7.code
    assert var_8 == 'type'
    var_9 = var_7.text
    assert var_9 == 'Must be an object.'
    var_10 = var_7.start_position.char_index
    assert var_10 == 0
    var_11 = var_7.end_position.char_index
    assert var_11 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = '{"a": 1}'
    var_6 = 'b'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = 'not_b'
    var_15 = {var_14: var_1}
    var_16 = {var_0: var_15}
    var_17 = 15
    var_18 = '{"a": {"not_b": 1}}'
    var_19 = error.messages()[0]
    var_20 = var_19.index
    var_21 = bool(var_19.index == ['a', 'b'])
    assert var_21 is True
    var_22 = var_19.code
    assert var_22 == 'required'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error_on_invalid_value. Retrieved 3/31 statements.


def test_case_0():
    var_0 = 'bad'
    var_1 = 0
    var_2 = 2
    var_3 = len(error.messages())
    assert var_3 == 1
    var_4 = error.messages()[0].code
    assert var_4 == 'invalid'
    var_5 = error.messages()[0].text
    assert var_5 == 'Invalid'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 9/11 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 12/29 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 11/23 statements.
# Partially parsed test_validate_with_positions_sorting_order. Retrieved 14/28 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = '{"name": "John"}'
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 15
    var_6 = {}
    var_7 = module_0.String(**var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = 'name'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = {}
    var_11 = 4
    var_12 = None
    var_13 = 6
    var_14 = 'name'

def test_case_0():
    var_0 = '{"age": "not_an_int"}'
    var_1 = 'age'
    var_2 = 'not_an_int'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 18
    var_6 = {var_1: var_2}
    var_7 = 1
    var_8 = 4
    var_9 = 7
    var_10 = 17

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 0
    var_7 = 14
    var_8 = 'array'
    var_9 = {var_1: var_3, var_8: var_4}
    var_10 = 5
    var_11 = 12
    var_12 = 10
    var_13 = 11



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/19 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 6/22 statements.
# Partially parsed test_validate_with_positions_type_error_with_position. Retrieved 3/19 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 4/21 statements.
# Failed to parse test_validate_with_matches_positions_success.


def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = 'field'
    var_4 = module_0.Field()
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'not_a_dict'
    var_1 = 0
    var_2 = 9

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = {}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/7 statements.
# Partially parsed test_validate_with_positions_validation_error_with_lookup. Retrieved 15/30 statements.
# Partially parsed test_validate_with_positions_required_field_formatting. Retrieved 7/17 statements.
# Partially parsed test_validate_with_positions_sorting_by_index. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'valid_value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'Error 1'
    var_3 = 'type'
    var_4 = [var_0]
    var_5 = module_0.Message(text=var_2, code=var_3, index=var_4)
    var_6 = 'Error 2'
    var_7 = 'required'
    var_8 = 'b'
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_6, code=var_7, index=var_9)
    var_11 = [var_5, var_10]
    var_12 = module_0.ValidationError(messages=var_11)
    var_13 = []
    var_14 = lambda : var_13

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
    var_0 = 'Late'
    var_1 = 'type'
    var_2 = 'z'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Early'
    var_6 = 'a'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.ValidationError(messages=var_9)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/14 statements.
# Partially parsed test_validate_with_positions_error_handling. Retrieved 3/27 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'content'
    var_1 = 0
    var_2 = 7



