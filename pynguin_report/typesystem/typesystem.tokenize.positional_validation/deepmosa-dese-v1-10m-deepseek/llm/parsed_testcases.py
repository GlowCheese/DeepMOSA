####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_valid_value. Retrieved 7/12 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 5/11 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 7/13 statements.


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
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"key": "invalid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_handles_required_error. Retrieved 6/20 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 'required_field'
    var_3 = module_0.Field()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 4/10 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 13/14 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = module_2.validate_with_positions(token=var_6, validator=var_7)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    assert var_7 is None

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 1
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{1: "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'twenty'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"age": "twenty"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_with_required_field_error. Retrieved 10/22 statements.
# Partially parsed test_validate_with_positions_with_validation_error. Retrieved 8/20 statements.
# Partially parsed test_validate_with_positions_with_valid_value. Retrieved 8/17 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = None
    var_7 = 0
    var_8 = 50
    var_9 = 'some content'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0
    var_6 = 50
    var_7 = 'some content'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0
    var_6 = 50
    var_7 = 'some content'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_valid_value. Retrieved 4/11 statements.
# Partially parsed test_validate_with_positions_invalid_value. Retrieved 7/14 statements.
# Partially parsed test_validate_with_positions_required_field. Retrieved 8/13 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 10/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 123
    var_4 = 0
    var_5 = 2
    var_6 = '123'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[0]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 0
    var_5 = 1
    var_6 = '{}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = 'field'
    var_9 = error.messages()[0]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'nested'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 13
    var_8 = '{"nested": {}}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = 'field'
    var_11 = error.messages()[0]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_validation_error. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = ''
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 12
    var_3 = ''
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 18
    var_7 = ''
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_null_value. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{123: "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'field'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = var_4.value
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/18 statements.
# Partially parsed test_validate_with_positions_invalid_field. Retrieved 8/17 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'existing_field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 'required_field'
    var_5 = module_0.Field()
    var_6 = {var_4: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'invalid_field'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = module_0.Field()
    var_5 = {var_0: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = error.messages()[0]



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = '{}'
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = module_1.Field()
    var_5 = var_3.value
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #14
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"field": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'missing_field'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = e.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not_a_dict'
    var_1 = 0
    var_2 = 10
    var_3 = '"not_a_dict"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'field'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = e.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"field": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'field'
    var_6 = True
    var_7 = module_1.Field(allow_null=var_6)
    var_8 = {var_5: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    assert var_10 is None



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = False
    var_4 = module_1.Field(allow_null=var_3)
    var_5 = var_2.value
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_validation. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"user": {"name": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"name": "test", "age": 25}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 8/15 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 8/15 statements.
# Partially parsed test_validate_with_positions_null_error. Retrieved 8/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = 10
    var_7 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = 10
    var_7 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = 10
    var_7 = error.messages()[0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 7/15 statements.
# Partially parsed test_validate_with_positions_generic_error. Retrieved 7/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 5
    var_3 = 'field_name'
    var_4 = module_0.Field()
    var_5 = {var_3: var_4}
    var_6 = module_1.Schema(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 20
    var_3 = 'field_name'
    var_4 = module_0.Field()
    var_5 = {var_3: var_4}
    var_6 = module_1.Schema(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_missing_required_field. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_field_error. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 3
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"user": {"name": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 10/24 statements.
# Partially parsed test_validate_with_positions_invalid_field. Retrieved 7/21 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/20 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_0.String()
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = 0
    var_9 = 10

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0
    var_6 = 10

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0
    var_6 = 10



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 0
    var_3 = module_1.Token(var_1, var_2, var_2)
    var_4 = module_2.validate_with_positions(token=var_3, validator=var_0)



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = "{'name': 'John'}"
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 1
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 9
    var_5 = "{1: 'John'}"
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'John'
    var_1 = 0
    var_2 = 4
    var_3 = "'John'"
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = False
    var_9 = module_2.Schema(var_7)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = error.messages()[0]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 3/23 statements.
# Partially parsed test_validate_with_positions_non_required_field. Retrieved 3/23 statements.


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10

def test_case_0():
    var_0 = {}
    var_1 = 5
    var_2 = 15



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_valid_value. Retrieved 1/12 statements.
# Partially parsed test_validate_with_positions_validation_error. Retrieved 1/19 statements.
# Partially parsed test_validate_with_positions_required_field_error. Retrieved 1/19 statements.


def test_case_0():
    var_0 = 'valid'

def test_case_0():
    var_0 = 'invalid'

def test_case_0():
    var_0 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 6/18 statements.
# Partially parsed test_validate_with_positions_invalid_key_error. Retrieved 6/18 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 6/17 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key2'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 6/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 3/11 statements.
# Partially parsed test_validate_with_positions_null_value. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_valid_value. Retrieved 3/10 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 'field'
    var_3 = module_0.Field()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = module_0.Field()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 7/8 statements.
# Partially parsed test_validate_with_positions_nested_validation_error. Retrieved 15/16 statements.


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
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_2.validate_with_positions(token=var_4, validator=var_5)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"name": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

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
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    assert var_7 is None



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = 'field'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"field": null}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = False
    var_8 = module_1.Field(allow_null=var_7)
    var_9 = {var_0: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = var_6.value
    var_12 = var_10.validate(var_11)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 8/9 statements.
# Partially parsed test_validate_with_positions_nested_validation_error. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 12
    var_3 = '"not an object"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = 'not a number'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"age": "not a number"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"name": "John", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_invalid_type_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_invalid_key_error. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'twenty'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"age": "twenty"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 1
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{1: "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_positions_raises_validation_error. Retrieved 4/10 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_validation_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_null_value_not_allowed. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 12
    var_3 = '"not an object"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = 'not a number'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"age": "not a number"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"name": "John", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = True
    var_9 = module_2.Schema(var_7)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    assert var_10 is None

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)



