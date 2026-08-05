####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/20 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 4/20 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 4/19 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 6/26 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '10'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = error.messages()[0]

def test_case_0():
    var_0 = 'abc'
    var_1 = 0
    var_2 = 2
    var_3 = error.messages()[0]

def test_case_0():
    var_0 = 'user'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = '{}'
    var_5 = error.messages()[0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 10/35 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 7/32 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 12/41 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = '{"a": 1}'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = {}
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = '{"a": 1}'
    var_8 = 'missing'
    var_9 = 5

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = {}
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = '{"a": 1}'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = {}
    var_4 = 10
    var_5 = {}
    var_6 = 5
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Failed to parse test_validate_with_positions_error_catch.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'mock_value'
    var_1 = 0
    var_2 = 8



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error_on_failure. Retrieved 1/37 statements.
# Partially parsed test_validate_with_positions_triggers_except_block. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'val'

def test_case_0():
    var_0 = '{}'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'req'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 12/33 statements.
# Partially parsed test_validate_with_positions_type_error_mapping. Retrieved 6/21 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 8/24 statements.
# Failed to parse test_validate_with_tokens_success_helper.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'other'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 8
    var_5 = '{"other": 1}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'username'
    var_8 = {}
    var_9 = 0
    var_10 = '{}'
    var_11 = error.messages()[0]

def test_case_0():
    var_0 = 'age'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = '{"age": 10}'
    var_5 = error.messages()[0]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 14
    var_7 = '{"a": 1, "b": 2}'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/9 statements.
# Partially parsed test_validate_with_positions_validation_error_simple. Retrieved 6/13 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 6/24 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'bad'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = 'ValidationError not raised'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'other'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 6
    var_5 = "{'other': 1}"
    var_6 = {}
    var_7 = 'ValidationError not raised'
    var_8 = AssertionError(var_7)

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 6
    var_3 = 'content'
    var_4 = 'ValidationError not raised'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/14 statements.
# Partially parsed test_validate_with_positions_error_with_index. Retrieved 6/17 statements.
# Partially parsed test_validate_with_positions_error_required. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key1": 1}'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_error_standard. Retrieved 7/17 statements.
# Partially parsed test_validate_with_positions_error_required. Retrieved 8/18 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid type'
    var_1 = 'type'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Missing'
    var_1 = 'required'
    var_2 = 'user'
    var_3 = 'name'
    var_4 = [var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Err2'
    var_1 = 'type'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Err1'
    var_6 = 'a'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.ValidationError(messages=var_9)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 10/43 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 5
    var_5 = '{"a": 1}'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = None
    var_3 = 0
    var_4 = ''
    var_5 = {}
    var_6 = 'a'
    var_7 = {var_6: var_2}
    var_8 = 10
    var_9 = '{"a": 1}'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 1/83 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_required_error_mapping. Retrieved 5/23 statements.
# Partially parsed test_validate_with_positions_type_error_mapping. Retrieved 4/20 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 6/43 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = 'name'
    var_4 = {var_3: var_1}

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1
    var_3 = 'name'

def test_case_0():
    var_0 = 'a b'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/18 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 5/24 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 4/21 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = '{}'
    var_4 = error.messages()[0]

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 9
    var_3 = error.messages()[0]

def test_case_0():
    var_0 = 'child'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"child": "val"}'
    var_6 = error.messages()[0]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/14 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 123

def test_case_0():
    pass



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 3/21 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 3/20 statements.
# Partially parsed test_validate_with_positions_sorting_order. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = 1

def test_case_0():
    var_0 = 'abcde'
    var_1 = 0
    var_2 = 4



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_error_handling. Retrieved 12/23 statements.
# Partially parsed test_validate_with_positions_error_handling_other_code. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 5
    var_3 = 6
    var_4 = 'Required'
    var_5 = 'required'
    var_6 = [var_0]
    var_7 = module_0.Message(text=var_4, code=var_5, index=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = 'ValidationError was not raised'
    var_11 = AssertionError(var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 5
    var_3 = 6
    var_4 = 'Invalid'
    var_5 = 'type'
    var_6 = [var_0]
    var_7 = module_0.Message(text=var_4, code=var_5, index=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = 'ValidationError was not raised'
    var_11 = AssertionError(var_10)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/9 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 11/22 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 10/21 statements.
# Partially parsed test_validate_with_positions_sorting_messages. Retrieved 15/33 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Not a dict'
    var_1 = 'type'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 5
    var_8 = 10
    var_9 = 'ValidationError not raised'
    var_10 = AssertionError(var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'required'
    var_2 = 'field_name'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 0
    var_8 = 'ValidationError not raised'
    var_9 = AssertionError(var_8)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'First'
    var_1 = 'type'
    var_2 = 10
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Second'
    var_6 = 5
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = 6
    var_12 = 11
    var_13 = 'ValidationError not raised'
    var_14 = AssertionError(var_13)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error_on_failure. Retrieved 3/24 statements.


def test_case_0():
    var_0 = 'val'
    var_1 = 0
    var_2 = 3



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 5/46 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 0
    var_2 = 4
    var_3 = 'val'
    var_4 = 3



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_validation_error_with_positional_mapping. Retrieved 14/24 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 4/18 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorting. Retrieved 14/30 statements.
# Failed to parse test_validate_with_tokens_setup.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 4
    var_5 = "{'a': 1}"
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)

import typesystem.base as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = '{"other": 1}'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_1, var_1, var_2)
    var_4 = None
    var_5 = 9
    var_6 = 10
    var_7 = 'other'
    var_8 = {var_7: var_1}
    var_9 = 11
    var_10 = 'a'
    var_11 = module_1.Field()
    var_12 = {var_10: var_11}
    var_13 = module_2.Schema(var_12)

def test_case_0():
    var_0 = '"not_an_int"'
    var_1 = 'not_an_int'
    var_2 = 0
    var_3 = 11

import typesystem.base as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_1, var_1, var_2)
    var_4 = 8
    var_5 = 7
    var_6 = module_0.Position(var_1, var_4, var_5)
    var_7 = {}
    var_8 = 2
    var_9 = {}
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_1, var_11: var_8}
    var_13 = 13



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/10 statements.
# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 9/35 statements.


def test_case_0():
    var_0 = 'some_value'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error text'
    var_1 = 'type'
    var_2 = module_0.Message(text=var_0, code=var_1)
    var_3 = [var_2]
    var_4 = 'val'
    var_5 = 0
    var_6 = 3
    var_7 = 'ValidationError was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/12 statements.
# Partially parsed test_validate_with_positions_validation_error_type. Retrieved 1/31 statements.


def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'data'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/12 statements.
# Partially parsed test_validate_with_positions_validation_error_type_error. Retrieved 12/26 statements.
# Partially parsed test_validate_with_positions_validation_error_required. Retrieved 11/25 statements.
# Partially parsed test_validate_with_positions_sorting. Retrieved 24/36 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'Invalid type'
    var_3 = 'type'
    var_4 = [var_0]
    var_5 = module_0.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = 2
    var_9 = 3
    var_10 = 'ValidationError not raised'
    var_11 = AssertionError(var_10)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Required'
    var_1 = 'required'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 1
    var_8 = 0
    var_9 = 'ValidationError not raised'
    var_10 = AssertionError(var_9)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error 2'
    var_1 = 'type'
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 'Error 1'
    var_6 = 'a'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_5, code=var_1, index=var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    var_11 = 1
    var_12 = 0
    var_13 = module_0.Position(var_11, var_11, var_12)
    var_14 = 2
    var_15 = module_0.Position(var_11, var_14, var_12)
    var_16 = 5
    var_17 = 4
    var_18 = module_0.Position(var_11, var_16, var_17)
    var_19 = 6
    var_20 = module_0.Position(var_11, var_19, var_17)
    var_21 = [var_6]
    var_22 = 'ValidationError not raised'
    var_23 = AssertionError(var_22)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/24 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 3/27 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 3/27 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '{}'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 'val'

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/16 statements.
# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 6/57 statements.
# Partially parsed test_validate_with_positions_required_field_logic. Retrieved 7/47 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = ''
    var_2 = 0
    var_3 = 10
    var_4 = '{"a": "not_int"}'
    var_5 = error.messages()[0]

def test_case_0():
    var_0 = 'missing'
    var_1 = ''
    var_2 = 0
    var_3 = 15
    var_4 = '{"missing": 1}'
    var_5 = '"missing"'
    var_6 = error.messages()[0]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 1/12 statements.
# Partially parsed test_validate_with_tokens_positional_mapping. Retrieved 9/43 statements.
# Partially parsed test_validate_with_positions_required_logic. Retrieved 10/38 statements.


def test_case_0():
    var_0 = 'hello'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'type'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 0
    var_6 = 5
    var_7 = '{}'
    var_8 = 'val'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'required'
    var_2 = 'missing_key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = 0
    var_6 = 10
    var_7 = '{}'
    var_8 = ''
    var_9 = e.messages()[0]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 7/21 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error text'
    var_1 = 'type_error'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 3/23 statements.
# Partially parsed test_validate_with_positions_validation_error_mapping. Retrieved 3/34 statements.
# Partially parsed test_validate_with_positions_required_logic. Retrieved 3/34 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3

def test_case_0():
    var_0 = 'input'
    var_1 = 0
    var_2 = 4

def test_case_0():
    var_0 = 'input'
    var_1 = 0
    var_2 = 4



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    pass



