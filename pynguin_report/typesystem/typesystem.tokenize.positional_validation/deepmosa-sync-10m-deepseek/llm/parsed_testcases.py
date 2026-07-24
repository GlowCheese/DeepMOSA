####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 9/26 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_valid. Retrieved 8/20 statements.
# Partially parsed test_validate_with_positions_null_not_allowed. Retrieved 8/14 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 10/16 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 9/32 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 10
    var_9 = '{"age": 25}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 12
    var_8 = '{"name": 123}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'inner_name'
    var_4 = 'outer'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 12
    var_9 = '{"outer": {}}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'title'
    var_5 = 123
    var_6 = 456
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 24
    var_10 = '{"name": 123, "title": 456}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 14
    var_8 = '{"name": "test"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = 0
    var_7 = 3
    var_8 = 'null'
    var_9 = module_1.Token(var_5, var_6, var_7, var_8)
    var_10 = module_2.validate_with_positions(token=var_9, validator=var_4)
    assert var_10 is None

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = 3
    var_7 = 'null'
    var_8 = module_1.Token(var_5, var_1, var_6, var_7)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_4)

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 12
    var_8 = '{123: "value"}'
    var_9 = module_1.Token(var_5, var_6, var_7, var_8)
    var_10 = module_2.validate_with_positions(token=var_9, validator=var_2)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = 3.14
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = 14
    var_11 = '{"data": 3.14}'
    var_12 = module_0.Token(var_8, var_9, var_10, var_11)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 12/30 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 18/39 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 12/30 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 16/36 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 7/27 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 8/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 10
    var_8 = '{"field": null}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = {var_3: var_4}
    var_11 = 8
    var_12 = 11
    var_13 = module_0.Token(var_4, var_11, var_12, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = None
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 20
    var_10 = '{"outer": {"inner": null}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = {var_3: var_5}
    var_13 = {var_4: var_12}
    var_14 = {var_3: var_5}
    var_15 = 10
    var_16 = module_0.Token(var_14, var_15, var_9, var_10)
    var_17 = 18
    var_18 = 21
    var_19 = module_0.Token(var_5, var_17, var_18, var_10)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 20
    var_8 = '{"field": "invalid"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = {var_3: var_4}
    var_11 = 10
    var_12 = 18
    var_13 = module_0.Token(var_4, var_11, var_12, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'invalid1'
    var_6 = 'invalid2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 40
    var_10 = '{"field1": "invalid1", "field2": "invalid2"}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = {var_3: var_5, var_4: var_6}
    var_13 = 12
    var_14 = 34
    var_15 = 20
    var_16 = 42
    var_17 = 1

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = 'invalid'
    var_7 = 0
    var_8 = 6
    var_9 = '"invalid"'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 18
    var_5 = '{"field": "valid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {var_0: var_1}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 7/21 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 13/29 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 8/32 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 9/35 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 7/21 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = ''
    var_4 = 'name'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'required'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True
    var_14 = var_10.text
    assert var_14 == "The field 'name' is required."

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 15
    var_3 = ''
    var_4 = {}
    var_5 = {}
    var_6 = module_0.Schema(var_4, **var_5)
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0]
    var_9 = var_8.code
    assert var_9 == 'type'
    var_10 = var_8.index
    var_11 = bool(var_8.index == [])
    assert var_11 is True
    var_12 = var_8.text
    assert var_12 == 'Must be an object.'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = 'name'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'required'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['user', 'name'])
    assert var_18 is True
    var_19 = var_15.text
    assert var_19 == "The field 'name' is required."

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = {}
    var_7 = {}
    var_8 = module_0.Schema(var_6, **var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'invalid_key'
    var_12 = var_10.index
    var_13 = bool(var_10.index == [123])
    assert var_13 is True
    var_14 = var_10.text
    assert var_14 == 'All object keys must be strings.'

def test_case_0():
    var_0 = 'max_length'
    var_1 = 'Must have at most {max_length} characters.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'too long'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 30
    var_8 = ''
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'max_length'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True
    var_14 = var_10.text
    assert var_14 == 'Must have at most 5 characters.'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = ''
    var_8 = 'always'
    var_9 = 'Always fails.'
    var_10 = {var_8: var_9}

import typesystem.schemas as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = ''
    var_4 = {}
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Schema(var_4, **var_7)

def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 14/31 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 13/33 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 13/36 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 15/40 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 14/41 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 10/23 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 14/34 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = '{"age": 25}'
    var_5 = 'age'
    var_6 = 25
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = len(var_4)
    var_10 = 1
    var_11 = var_9 - var_10
    var_12 = module_0.Token(var_7, var_8, var_11, var_4)
    var_13 = error.messages()[var_8]
    var_14 = var_13.code
    assert var_14 == 'required'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['name'])
    assert var_16 is True
    var_17 = var_13.text
    assert var_17 == "The field 'name' is required."
    var_18 = var_13.start_position.char_index
    assert var_18 == 0
    var_19 = len(var_4)
    var_20 = var_19 - var_10
    var_21 = var_13.end_position.char_index
    var_22 = bool(var_13.end_position.char_index == var_20)
    assert var_22 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = '{"name": 123}'
    var_5 = 123
    var_6 = {var_3: var_5}
    var_7 = 0
    var_8 = len(var_4)
    var_9 = 1
    var_10 = var_8 - var_9
    var_11 = module_0.Token(var_6, var_7, var_10, var_4)
    var_12 = error.messages()[var_7]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.index
    var_15 = bool(var_12.index == ['name'])
    assert var_15 is True
    var_16 = var_12.text
    assert var_16 == 'Must be a string.'
    var_17 = '123'
    var_18 = var_12.start_position.char_index
    var_19 = 2
    var_20 = var_12.end_position.char_index

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'city'
    var_4 = 'address'
    var_5 = '{"address": {}}'
    var_6 = {}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = len(var_5)
    var_10 = 1
    var_11 = var_9 - var_10
    var_12 = module_0.Token(var_7, var_8, var_11, var_5)
    var_13 = error.messages()[var_8]
    var_14 = var_13.code
    assert var_14 == 'required'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['address', 'city'])
    assert var_16 is True
    var_17 = var_13.text
    assert var_17 == "The field 'city' is required."
    var_18 = '{}'
    var_19 = var_13.start_position.char_index
    var_20 = var_13.end_position.char_index

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'email'
    var_5 = '{"name": 123, "email": 456}'
    var_6 = 123
    var_7 = 456
    var_8 = {var_3: var_6, var_4: var_7}
    var_9 = 0
    var_10 = len(var_5)
    var_11 = 1
    var_12 = var_10 - var_11
    var_13 = module_0.Token(var_8, var_9, var_12, var_5)
    var_14 = '123'
    var_15 = 2
    var_16 = '456'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = 'value'
    var_7 = '{"value": null}'
    var_8 = None
    var_9 = {var_6: var_8}
    var_10 = 0
    var_11 = len(var_7)
    var_12 = 1
    var_13 = var_11 - var_12
    var_14 = module_0.Token(var_9, var_10, var_13, var_7)
    var_15 = error.messages()[var_10]
    var_16 = var_15.code
    assert var_16 == 'null'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['value'])
    assert var_18 is True
    var_19 = 'null'
    var_20 = var_15.start_position.char_index
    var_21 = 3
    var_22 = var_15.end_position.char_index

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = '{"name": "John"}'
    var_5 = 'John'
    var_6 = {var_3: var_5}
    var_7 = 0
    var_8 = len(var_4)
    var_9 = 1
    var_10 = var_8 - var_9
    var_11 = module_0.Token(var_6, var_7, var_10, var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = '{123: "value"}'
    var_5 = 123
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = len(var_4)
    var_10 = 1
    var_11 = var_9 - var_10
    var_12 = module_0.Token(var_7, var_8, var_11, var_4)
    var_13 = error.messages()[var_8]
    var_14 = var_13.code
    assert var_14 == 'invalid_key'
    var_15 = var_13.index
    var_16 = bool(var_13.index == [123])
    assert var_16 is True
    var_17 = '123'
    var_18 = var_13.start_position.char_index
    var_19 = 2
    var_20 = var_13.end_position.char_index



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 6/18 statements.
# Partially parsed test_validate_with_positions_nested_required_error. Retrieved 8/22 statements.
# Partially parsed test_validate_with_positions_general_error. Retrieved 6/17 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/24 statements.
# Partially parsed test_validate_with_positions_invalid_key_error. Retrieved 8/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required_field'
    var_1 = {}
    var_2 = 0
    var_3 = 2
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner_required'
    var_1 = 'outer_key'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 20
    var_6 = '{"outer_key": {}}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'bad'
    var_4 = 0
    var_5 = 5
    var_6 = '"bad"'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'x'
    var_6 = 'y'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 30
    var_10 = '{"field1": "x", "field2": "y"}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 123
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 20
    var_6 = '{123: "invalid"}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_validation_error_without_index. Retrieved 5/15 statements.
# Partially parsed test_validate_with_positions_validation_error_with_index. Retrieved 8/20 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 9/23 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value'
    var_2 = {var_0: var_1}
    var_3 = 'bad'
    var_4 = 0
    var_5 = 2
    var_6 = module_0.Token(var_3, var_4, var_5, var_3)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value'
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = 'bad'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 10
    var_8 = '{"key":"bad"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required_field'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'bad'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 0
    var_8 = 30
    var_9 = '{"field1":"bad","field2":"bad"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 9/26 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_valid. Retrieved 8/20 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 8/20 statements.
# Partially parsed test_validate_with_positions_null_not_allowed. Retrieved 7/22 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 9/24 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 10
    var_9 = '{"age": 25}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 12
    var_8 = '{"name": 123}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'city'
    var_4 = 'address'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 15
    var_9 = '{"address": {}}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 123
    var_6 = 456
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 25
    var_10 = '{"name": 123, "email": 456}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'John'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"name": "John"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = True
    var_5 = None
    var_6 = 0
    var_7 = 3
    var_8 = 'null'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = False
    var_5 = None
    var_6 = 3
    var_7 = 'null'
    var_8 = module_0.Token(var_5, var_4, var_6, var_7)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = 'John'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 12
    var_9 = '{123: "John"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_handles_required_error. Retrieved 8/20 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 10
    var_8 = '{"field": null}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/29 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 16/34 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 5/14 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 16/33 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 10
    var_8 = '{"field": null}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = {var_3: var_4}
    var_11 = 8
    var_12 = 11
    var_13 = module_0.Token(var_4, var_11, var_12, var_8)
    var_14 = error.messages()[var_6]
    var_15 = var_14.text
    assert var_15 == "The field 'field' is required."
    var_16 = var_14.code
    assert var_16 == 'required'
    var_17 = var_14.index
    var_18 = bool(var_14.index == ['field'])
    assert var_18 is True
    var_19 = var_14.start_position.char_index
    assert var_19 == 8
    var_20 = var_14.end_position.char_index
    assert var_20 == 11

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = 5
    var_5 = 11
    var_6 = 'value: "invalid"'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = module_0.Token(var_3, var_4, var_5, var_6)
    var_9 = 0
    var_10 = error.messages()[var_9]
    var_11 = var_10.text
    assert var_11 == 'Custom error.'
    var_12 = var_10.code
    assert var_12 == 'custom'
    var_13 = var_10.index
    var_14 = bool(var_10.index == [])
    assert var_14 is True
    var_15 = var_10.start_position.char_index
    assert var_15 == 5
    var_16 = var_10.end_position.char_index
    assert var_16 == 11

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'nested'
    var_4 = 'outer'
    var_5 = 123
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 30
    var_10 = '{"outer": {"nested": 123}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = {var_3: var_5}
    var_13 = {var_4: var_12}
    var_14 = 20
    var_15 = 22
    var_16 = module_0.Token(var_5, var_14, var_15, var_10)
    var_17 = error.messages()[var_8]
    var_18 = var_17.text
    assert var_18 == 'Must be a string.'
    var_19 = var_17.code
    assert var_19 == 'type'
    var_20 = var_17.index
    var_21 = bool(var_17.index == ['outer', 'nested'])
    assert var_21 is True
    var_22 = var_17.start_position.char_index
    assert var_22 == 20
    var_23 = var_17.end_position.char_index
    assert var_23 == 22

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4
    var_3 = '"hello"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 20
    var_10 = '{"a": 1, "b": 2}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = {var_3: var_5, var_4: var_6}
    var_13 = -1
    var_14 = -1
    var_15 = 7
    var_16 = 16
    var_17 = -1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_validation_error_without_required. Retrieved 4/15 statements.
# Partially parsed test_validate_with_positions_validation_error_with_required. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 10/25 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 6
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '{"field2": "val", "field1": "val"}'
    var_3 = 'val'
    var_4 = {var_1: var_3, var_0: var_3}
    var_5 = 0
    var_6 = len(var_2)
    var_7 = 1
    var_8 = var_6 - var_7
    var_9 = module_0.Token(var_4, var_5, var_8, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 5/13 statements.
# Partially parsed test_validate_with_positions_validation_error_without_index. Retrieved 5/15 statements.
# Partially parsed test_validate_with_positions_validation_error_with_index. Retrieved 5/28 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 8/27 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 6/38 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 13
    var_3 = '{"field": 42}'
    var_4 = 'field'
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = 'field'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 14
    var_3 = '{"a":1,"b":2}'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_simple_validation_error. Retrieved 6/17 statements.
# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_nested_schema_error. Retrieved 8/35 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 7/32 statements.
# Partially parsed test_validate_with_positions_successful_validation. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_invalid_key_error. Retrieved 9/25 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = 0
    var_5 = 3
    var_6 = module_0.Token(var_3, var_4, var_5, var_3)
    var_7 = error.messages()[var_4]
    var_8 = var_7.text
    assert var_8 == 'Invalid value.'
    var_9 = var_7.code
    assert var_9 == 'custom'
    var_10 = var_7.start_position.line_no
    assert var_10 == 1
    var_11 = var_7.start_position.column_no
    assert var_11 == 1
    var_12 = var_7.start_position.char_index
    assert var_12 == 0
    var_13 = var_7.end_position.line_no
    assert var_13 == 1
    var_14 = var_7.end_position.column_no
    assert var_14 == 4
    var_15 = var_7.end_position.char_index
    assert var_15 == 3

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 20
    var_3 = '{"existing": "value"}'
    var_4 = 'missing'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = error.messages()[var_1]
    var_10 = var_9.text
    assert var_10 == "The field 'missing' is required."
    var_11 = var_9.code
    assert var_11 == 'required'
    var_12 = var_9.index
    var_13 = bool(var_9.index == ['missing'])
    assert var_13 is True
    var_14 = var_9.start_position.char_index
    assert var_14 == 10
    var_15 = var_9.end_position.char_index
    assert var_15 == 15

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be an integer.'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 0
    var_5 = 35
    var_6 = '{"nested": {"field": "invalid"}}'
    var_7 = 'nested'
    var_8 = 'field'
    var_9 = error.messages()[var_4]
    var_10 = var_9.text
    assert var_10 == 'Must be an integer.'
    var_11 = var_9.code
    assert var_11 == 'type'
    var_12 = var_9.index
    var_13 = bool(var_9.index == ['nested', 'field'])
    assert var_13 is True
    var_14 = var_9.start_position.char_index
    assert var_14 == 15
    var_15 = var_9.end_position.char_index
    assert var_15 == 21

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid.'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 0
    var_5 = 30
    var_6 = '{"a": "invalid1", "b": "invalid2"}'
    var_7 = 'a'
    var_8 = 'b'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = True
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_1.Schema(var_5, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    assert var_10 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 12
    var_3 = '{123: "value"}'
    var_4 = 'valid'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = error.messages()[var_1]
    var_10 = var_9.code
    assert var_10 == 'invalid_key'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [123])
    assert var_12 is True
    var_13 = var_9.start_position.char_index
    assert var_13 == 5
    var_14 = var_9.end_position.char_index
    assert var_14 == 9



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = {}
    var_2 = 0
    var_3 = ''



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 5/25 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 7/27 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/18 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 10
    var_8 = '{"name": 123}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'inner_name'
    var_1 = 'outer'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 15
    var_6 = '{"outer": {}}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 15
    var_10 = '{"a": 1, "b": 2}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = 3.14
    var_7 = 0
    var_8 = 3
    var_9 = '3.14'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 17/38 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 23/46 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 30
    var_3 = ''
    var_4 = 'name'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = error.messages()[var_1]
    var_10 = var_9.code
    assert var_10 == 'required'
    var_11 = var_9.index
    var_12 = bool(var_9.index == ['name'])
    assert var_12 is True
    var_13 = var_9.start_position.line_no
    assert var_13 == 1
    var_14 = var_9.start_position.column_no
    assert var_14 == 1

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 15
    var_3 = ''
    var_4 = 'name'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = error.messages()[var_1]
    var_10 = var_9.code
    assert var_10 == 'type'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [])
    assert var_12 is True
    var_13 = var_9.start_position.line_no
    assert var_13 == 1
    var_14 = var_9.start_position.column_no
    assert var_14 == 1

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 10
    var_2 = 20
    var_3 = ''
    var_4 = 'inner'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 30
    var_9 = module_0.Field()
    var_10 = 'required_field'
    var_11 = module_0.Field()
    var_12 = {var_10: var_11}
    var_13 = {}
    var_14 = module_1.Schema(var_12, **var_13)
    var_15 = {var_4: var_14}
    var_16 = {}
    var_17 = module_1.Schema(var_15, **var_16)
    var_18 = error.messages()[var_7]
    var_19 = var_18.code
    assert var_19 == 'required'
    var_20 = var_18.index
    var_21 = bool(var_18.index == ['inner', 'required_field'])
    assert var_21 is True
    var_22 = var_18.start_position.line_no
    assert var_22 == 1
    var_23 = var_18.start_position.column_no
    assert var_23 == 11

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 30
    var_5 = ''
    var_6 = module_0.Field()
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = ''
    var_4 = module_0.Field()
    var_5 = module_0.Field()
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = error.messages()[var_1]
    var_10 = var_9.code
    assert var_10 == 'union'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [])
    assert var_12 is True
    var_13 = var_9.start_position.line_no
    assert var_13 == 1
    var_14 = var_9.start_position.column_no
    assert var_14 == 1

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 20
    var_2 = 30
    var_3 = ''
    var_4 = {}
    var_5 = 0
    var_6 = 10
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 40
    var_13 = 'req1'
    var_14 = module_0.Field()
    var_15 = {var_13: var_14}
    var_16 = {}
    var_17 = module_1.Schema(var_15, **var_16)
    var_18 = 'req2'
    var_19 = module_0.Field()
    var_20 = {var_18: var_19}
    var_21 = {}
    var_22 = module_1.Schema(var_20, **var_21)
    var_23 = {var_7: var_17, var_8: var_22}
    var_24 = {}
    var_25 = module_1.Schema(var_23, **var_24)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 6/24 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 10/40 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 7/33 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 4
    var_3 = ''
    var_4 = 'name'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'required'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True
    var_14 = var_10.start_position
    var_15 = bool(var_10.start_position is not None)
    assert var_15 is True
    var_16 = var_10.end_position
    var_17 = bool(var_10.end_position is not None)
    assert var_17 is True

def test_case_0():
    var_0 = 'not an integer'
    var_1 = 0
    var_2 = 15
    var_3 = ''
    var_4 = 'type'
    var_5 = 'Must be an integer.'
    var_6 = {var_4: var_5}
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0]
    var_9 = var_8.code
    assert var_9 == 'type'
    var_10 = var_8.index
    var_11 = bool(var_8.index == [])
    assert var_11 is True
    var_12 = var_8.start_position
    var_13 = bool(var_8.start_position is not None)
    assert var_13 is True
    var_14 = var_8.end_position
    var_15 = bool(var_8.end_position is not None)
    assert var_15 is True

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = 'not an int'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 35
    var_7 = ''
    var_8 = 'type'
    var_9 = 'Must be an integer.'
    var_10 = {var_8: var_9}
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.index
    var_15 = bool(var_12.index == ['user', 'age'])
    assert var_15 is True
    var_16 = var_12.start_position
    var_17 = bool(var_12.start_position is not None)
    assert var_17 is True
    var_18 = var_12.end_position
    var_19 = bool(var_12.end_position is not None)
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = module_0.Field()
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = ''
    var_4 = 'type'
    var_5 = 'Must be a string.'
    var_6 = {var_4: var_5}
    var_7 = 'type'
    var_8 = 'Must be an integer.'
    var_9 = {var_7: var_8}
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0]
    var_12 = var_11.code
    assert var_12 == 'union'
    var_13 = var_11.index
    var_14 = bool(var_11.index == [])
    assert var_14 is True
    var_15 = var_11.start_position
    var_16 = bool(var_11.start_position is not None)
    assert var_16 is True
    var_17 = var_11.end_position
    var_18 = bool(var_11.end_position is not None)
    assert var_18 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 5/25 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 7/27 statements.
# Partially parsed test_validate_with_positions_schema_field_validation_error. Retrieved 7/27 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 4/27 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 6/25 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 5
    var_4 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 5
    var_6 = ''

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'bad'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 5
    var_8 = ''

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 5
    var_3 = ''

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 5
    var_5 = ''

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Error.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 5
    var_10 = ''



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 11/29 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/29 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 10/33 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 7/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 10
    var_9 = '{"age": 25}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = error.messages()[var_7]
    var_12 = var_11.code
    assert var_12 == 'required'
    var_13 = var_11.index
    var_14 = bool(var_11.index == ['name'])
    assert var_14 is True
    var_15 = var_11.text
    assert var_15 == "The field 'name' is required."

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 12
    var_8 = '{"name": 123}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = error.messages()[var_6]
    var_11 = var_10.code
    assert var_11 == 'type'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True
    var_14 = var_10.text
    assert var_14 == 'Must be a string.'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'max_length'
    var_1 = 'Must have no more than 5 characters.'
    var_2 = {var_0: var_1}
    var_3 = 'title'
    var_4 = 'item'
    var_5 = 'too long'
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 30
    var_10 = '{"item": {"title": "too long"}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = error.messages()[var_8]
    var_13 = var_12.code
    assert var_13 == 'max_length'
    var_14 = var_12.index
    var_15 = bool(var_12.index == ['item', 'title'])
    assert var_15 is True
    var_16 = var_12.text
    assert var_16 == 'Must have no more than 5 characters.'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'max_length'
    var_2 = 'Must be a string.'
    var_3 = 'Too long.'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 123
    var_8 = 'longvalue'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 0
    var_11 = 30
    var_12 = '{"a": 123, "b": "longvalue"}'
    var_13 = module_0.Token(var_9, var_10, var_11, var_12)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be an integer.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be a string.'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = 3.14
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = 15
    var_11 = '{"data": 3.14}'
    var_12 = module_0.Token(var_8, var_9, var_10, var_11)
    var_13 = error.messages()[var_9]
    var_14 = var_13.code
    assert var_14 == 'union'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['data'])
    assert var_16 is True
    var_17 = var_13.text
    assert var_17 == 'Did not match any valid type.'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": null}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 13/29 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 8/28 statements.
# Partially parsed test_validate_with_positions_invalid_key_error. Retrieved 11/25 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 11/32 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 7/21 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = 'field'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'required'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['field'])
    assert var_13 is True
    var_14 = var_10.start_position
    var_15 = bool(var_10.start_position is not None)
    assert var_15 is True
    var_16 = var_10.end_position
    var_17 = bool(var_10.end_position is not None)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'outer'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = 'content'
    var_6 = 'inner'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'required'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['outer', 'inner'])
    assert var_18 is True
    var_19 = var_15.start_position
    var_20 = bool(var_15.start_position is not None)
    assert var_20 is True
    var_21 = var_15.end_position
    var_22 = bool(var_15.end_position is not None)
    assert var_22 is True

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 20
    var_8 = 'content'
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'invalid'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['field'])
    assert var_13 is True
    var_14 = var_10.start_position
    var_15 = bool(var_10.start_position is not None)
    assert var_15 is True
    var_16 = var_10.end_position
    var_17 = bool(var_10.end_position is not None)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 123
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = 'content'
    var_6 = 'field'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'invalid_key'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [123])
    assert var_15 is True
    var_16 = var_12.start_position
    var_17 = bool(var_12.start_position is not None)
    assert var_17 is True
    var_18 = var_12.end_position
    var_19 = bool(var_12.end_position is not None)
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 0
    var_5 = 20
    var_6 = 'content'
    var_7 = module_0.Field()
    var_8 = module_0.Field()
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = 'content'
    var_6 = module_0.Field()
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'content'
    var_4 = {}
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Schema(var_4, **var_7)

def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 3/35 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 4/36 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 6/42 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 3/32 statements.


def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = {}

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'value'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'inner_field'
    var_4 = 'outer_field'
    var_5 = 'value'
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_validation_error_without_positions. Retrieved 5/15 statements.
# Partially parsed test_validate_with_positions_required_field_error. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 9/26 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "valid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 6
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'field'
    var_6 = bool(False)
    assert var_6 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 'bad'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"outer": {"inner": "bad"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 13
    var_7 = '{"a":1,"b":2}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/26 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 12/32 statements.
# Partially parsed test_validate_with_positions_non_required_error. Retrieved 7/33 statements.
# Partially parsed test_validate_with_positions_sorts_messages_by_position. Retrieved 9/46 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 6/37 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = ''
    var_4 = module_0.Field()
    var_5 = 'field'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'outer'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = module_0.Field()
    var_7 = 'inner'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)

def test_case_0():
    var_0 = 'field'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = 'type'
    var_7 = 'Must be an integer.'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'invalid1'
    var_3 = 'invalid2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = ''
    var_8 = 'type'
    var_9 = 'Must be an integer.'
    var_10 = {var_8: var_9}

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = ''
    var_4 = 'type'
    var_5 = 'Must be an integer.'
    var_6 = {var_4: var_5}
    var_7 = 'max_length'
    var_8 = 'Must be at most 5 characters.'
    var_9 = {var_7: var_8}



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 9/26 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 13/32 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 8/22 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/22 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 5
    var_3 = '{"key":'
    var_4 = module_0.Field()
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = '{"key": 123}'
    var_6 = module_0.Field()
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'nested'
    var_1 = 'inner'
    var_2 = 456
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 32
    var_7 = '{"nested": {"inner": 456}}'
    var_8 = module_0.Field()
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_1.Schema(var_12, **var_13)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = ''
    var_5 = module_0.Field()
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Field()
    var_5 = module_0.Field()
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = {}
    var_5 = 0
    var_6 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 10/28 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 10/30 statements.
# Partially parsed test_validate_with_positions_field_error. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 9/27 statements.
# Partially parsed test_validate_with_positions_no_error. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 7/23 statements.
# Partially parsed test_validate_with_positions_schema_type_error. Retrieved 9/17 statements.
# Partially parsed test_validate_with_positions_schema_null_error. Retrieved 9/17 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key_error. Retrieved 11/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'other'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 10
    var_9 = '{"other": "value"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = error.messages()[var_7]
    var_12 = var_11.text
    assert var_12 == "The field 'field' is required."
    var_13 = var_11.code
    assert var_13 == 'required'
    var_14 = var_11.index
    var_15 = bool(var_11.index == ['field'])
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 20
    var_9 = '{"outer": {}}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = error.messages()[var_7]
    var_12 = var_11.text
    assert var_12 == "The field 'inner' is required."
    var_13 = var_11.code
    assert var_13 == 'required'
    var_14 = var_11.index
    var_15 = bool(var_11.index == ['outer', 'inner'])
    assert var_15 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'bad'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"field": "bad"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = error.messages()[var_6]
    var_11 = var_10.text
    assert var_11 == 'Invalid value.'
    var_12 = var_10.code
    assert var_12 == 'invalid'
    var_13 = var_10.index
    var_14 = bool(var_10.index == ['field'])
    assert var_14 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'bad'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 0
    var_8 = 30
    var_9 = '{"field1": "bad", "field2": "bad"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = 'good'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
    var_5 = '{"field": "good"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'bad'
    var_4 = 0
    var_5 = 3
    var_6 = '"bad"'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.text
    assert var_9 == 'Invalid value.'
    var_10 = var_8.code
    assert var_10 == 'invalid'
    var_11 = var_8.index
    var_12 = bool(var_8.index == [])
    assert var_12 is True

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not an object'
    var_4 = 0
    var_5 = 13
    var_6 = '"not an object"'
    var_7 = module_1.Token(var_3, var_4, var_5, var_6)
    var_8 = module_2.validate_with_positions(token=var_7, validator=var_2)
    var_9 = error.messages()[var_4]
    var_10 = var_9.text
    assert var_10 == 'Must be an object.'
    var_11 = var_9.code
    assert var_11 == 'type'
    var_12 = var_9.index
    var_13 = bool(var_9.index == [])
    assert var_13 is True

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = 4
    var_7 = 'null'
    var_8 = module_1.Token(var_5, var_1, var_6, var_7)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_4)
    var_10 = error.messages()[var_1]
    var_11 = var_10.text
    assert var_11 == 'May not be null.'
    var_12 = var_10.code
    assert var_12 == 'null'
    var_13 = var_10.index
    var_14 = bool(var_10.index == [])
    assert var_14 is True

import typesystem.schemas as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 12
    var_8 = '{1: "value"}'
    var_9 = module_1.Token(var_5, var_6, var_7, var_8)
    var_10 = module_2.validate_with_positions(token=var_9, validator=var_2)
    var_11 = error.messages()[var_6]
    var_12 = var_11.text
    assert var_12 == 'All object keys must be strings.'
    var_13 = var_11.code
    assert var_13 == 'invalid_key'
    var_14 = var_11.index
    var_15 = bool(var_11.index == [1])
    assert var_15 is True



