####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 14
    var_5 = '{"key": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 11
    var_5 = '{"key": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 22
    var_7 = '{"nested": {"key": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 24
    var_7 = '{"key1": 123, "key2": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.String()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.String()
    var_6 = module_1.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_1.Union(var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_nested_schema. Retrieved 17/18 statements.
# Partially parsed test_validate_with_positions_union_field. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_null_value_not_allowed. Retrieved 8/9 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_positional_info. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'user'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"username": "user", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'user'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "user"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.String()
    var_9 = module_1.Integer()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'user'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"username": "user", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'username'
    var_2 = 'age'
    var_3 = 'invalid'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = 30
    var_8 = '{"user": {"username": "user", "age": "invalid"}}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = module_1.String()
    var_11 = module_1.Integer()
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = {var_0: var_13}
    var_15 = module_2.Schema(var_14)
    var_16 = module_3.validate_with_positions(token=var_9, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.String()
    var_6 = module_1.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_1.Union(var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)

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
    var_6 = module_1.String()
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    assert var_7 is None

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = False
    var_6 = module_1.String()
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 'user'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{123: "user"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'username'
    var_8 = module_1.String()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'user'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"username": "user", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_null_value. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_union_field. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 13/14 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
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
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 11
    var_3 = '"not a dict"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
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
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"user": {}}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = {var_0: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = module_1.Union(var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": 123}'
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
    var_0 = 'username'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": null}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'user123'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
    var_5 = '{"username": "user123"}'
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
    var_2 = 1
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
    var_0 = 'not_a_dict'
    var_1 = 0
    var_2 = 10
    var_3 = '"not_a_dict"'
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
    var_0 = 'user'
    var_1 = 'username'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"username": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = None
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": null, "email": 123}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = False
    var_10 = module_1.Field(allow_null=var_9)
    var_11 = module_1.Field()
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"age": "invalid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Integer()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)
    var_11 = error.messages()[0]

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
    var_9 = module_1.String()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"name": 123, "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.String()
    var_9 = module_1.String()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 11
    var_3 = '{"username": "test"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

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
    var_9 = module_1.String()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)
    var_15 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": "test", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'email'
    var_8 = module_1.String()
    var_9 = module_1.String()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 10
    var_3 = '"not a dict"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.String()
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
    var_9 = module_1.String()
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
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_5 = 'username'
    var_6 = module_1.String()
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
    var_0 = 'username'
    var_1 = 'invalid_key'
    var_2 = 123
    var_3 = 456
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "invalid_key": 456}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'email'
    var_10 = module_1.String()
    var_11 = module_1.String()
    var_12 = {var_0: var_10, var_9: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 17/18 statements.
# Partially parsed test_validate_with_positions_union_type. Retrieved 10/11 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.String()
    var_9 = module_1.Integer()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'not_a_number'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 22
    var_5 = '{"age": "not_a_number"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Integer()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'not_a_number'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 35
    var_7 = '{"name": 123, "age": "not_a_number"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"name": "John"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'age'
    var_10 = module_1.String()
    var_11 = module_1.Integer()
    var_12 = {var_1: var_10, var_9: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = {var_0: var_13}
    var_15 = module_2.Schema(var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)

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
    var_6 = 22
    var_7 = '{"name": "John", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 12
    var_3 = '"not_an_int"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Integer()
    var_6 = module_1.String()
    var_7 = [var_5, var_6]
    var_8 = module_1.Union(var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"key": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 11
    var_5 = '{"key": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 22
    var_7 = '{"nested": {"key": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = {var_1: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = {var_0: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 10
    var_3 = '"not_an_int"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Integer()
    var_6 = module_1.String()
    var_7 = [var_5, var_6]
    var_8 = module_1.Union(var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"key1": 123, "key2": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.String()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key2'
    var_1 = 'key1'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"key2": 123, "key1": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.String()
    var_11 = {var_1: var_9, var_0: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_custom_error. Retrieved 12/15 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 12/13 statements.
# Partially parsed test_validate_with_positions_sorted_by_position. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"key": "value"}'
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
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key'
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
    var_0 = 'outer'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = '{"outer": {}}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'inner'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = {var_0: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.base as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 6
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = ()
    var_7 = 'Custom error'
    var_8 = 'custom'
    var_9 = module_2.ValidationError(text=var_7, code=var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_5)
    var_11 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'b'
    var_6 = 'a'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 15/16 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"key": "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'key'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 11
    var_5 = '{"key": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
    var_8 = {var_0: var_7}
    var_9 = module_2.Schema(var_8)
    var_10 = module_3.validate_with_positions(token=var_6, validator=var_9)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 22
    var_7 = '{"nested": {"key": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
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
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"key1": 123, "key2": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = False
    var_11 = module_1.String()
    var_12 = {var_0: var_9, var_1: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_schema. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.String()
    var_9 = module_1.String()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 10
    var_3 = '{"value": "not a dict"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = module_1.Integer()
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
    var_9 = module_1.String()
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
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 10
    var_7 = '{"a": 1, "b": 2}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'c'
    var_10 = module_1.String()
    var_11 = module_1.String()
    var_12 = module_1.String()
    var_13 = {var_0: var_10, var_1: var_11, var_9: var_12}
    var_14 = module_2.Schema(var_13)
    var_15 = module_3.validate_with_positions(token=var_8, validator=var_14)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'password'
    var_2 = 'test'
    var_3 = 'pass'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": "test", "password": "pass"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.String()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_2.Schema(var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 10
    var_3 = '"not a dict"'
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
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
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
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "john"}'
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
    var_0 = 'user'
    var_1 = 'username'
    var_2 = 'john'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"user": {"username": "john"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'password'
    var_10 = module_1.Field()
    var_11 = module_1.Field()
    var_12 = {var_1: var_10, var_9: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = {var_0: var_13}
    var_15 = module_2.Schema(var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'john'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 40
    var_7 = '{"username": "john", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'password'
    var_10 = module_1.Field()
    var_11 = module_1.Field()
    var_12 = module_1.Field()
    var_13 = {var_0: var_10, var_9: var_11, var_1: var_12}
    var_14 = module_2.Schema(var_13)
    var_15 = module_3.validate_with_positions(token=var_8, validator=var_14)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not a number'
    var_1 = 0
    var_2 = 12
    var_3 = '"not a number"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = var_5 | var_6
    var_8 = module_2.validate_with_positions(token=var_4, validator=var_7)
    var_9 = error.messages()[0]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 9/10 statements.
# Partially parsed test_validate_with_positions_null_error. Retrieved 8/9 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{"key": "value"}'
    var_7 = module_1.Token(var_3, var_4, var_5, var_6)
    var_8 = module_2.validate_with_positions(token=var_7, validator=var_0)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = None
    var_3 = 4
    var_4 = 'null'
    var_5 = module_1.Token(var_2, var_0, var_3, var_4)
    var_6 = module_2.validate_with_positions(token=var_5, validator=var_1)
    var_7 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = 0
    var_6 = 1
    var_7 = '{}'
    var_8 = module_2.Token(var_4, var_5, var_6, var_7)
    var_9 = module_3.validate_with_positions(token=var_8, validator=var_3)
    var_10 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = 0
    var_3 = var_1 / var_2
    var_4 = 'not_an_int'
    var_5 = 10
    var_6 = '"not_an_int"'
    var_7 = module_1.Token(var_4, var_2, var_5, var_6)
    var_8 = module_2.validate_with_positions(token=var_7, validator=var_0)
    var_9 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {var_0: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = {}
    var_8 = {var_0: var_7}
    var_9 = 0
    var_10 = 15
    var_11 = '{"user": {}}'
    var_12 = module_2.Token(var_8, var_9, var_10, var_11)
    var_13 = module_3.validate_with_positions(token=var_12, validator=var_6)
    var_14 = error.messages()[0]

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {}
    var_7 = 0
    var_8 = 1
    var_9 = '{}'
    var_10 = module_2.Token(var_6, var_7, var_8, var_9)
    var_11 = module_3.validate_with_positions(token=var_10, validator=var_5)
    var_12 = sorted(error.messages(), key=lambda m: m.start_position.char_index)

import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'invalid'
    var_5 = 0
    var_6 = 7
    var_7 = '"invalid"'
    var_8 = module_1.Token(var_4, var_5, var_6, var_7)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_3)
    var_10 = error.messages()[0]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = module_0.Token(var_2, var_3, var_4)
    var_6 = module_1.Field()
    var_7 = {var_0: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_5, validator=var_8)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)
    var_7 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Token(var_0, var_1, var_2)
    var_4 = 'required_field'
    var_5 = module_1.Field()
    var_6 = {var_4: var_5}
    var_7 = module_2.Schema(var_6)
    var_8 = module_3.validate_with_positions(token=var_3, validator=var_7)
    var_9 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = module_0.Token(var_4, var_5, var_6)
    var_8 = module_1.Field()
    var_9 = {var_1: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = {var_0: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_7, validator=var_12)
    var_14 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = module_0.Token(var_4, var_5, var_6)
    var_8 = False
    var_9 = module_1.Field(allow_null=var_8)
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_7, validator=var_12)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"invalid": "data"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'valid'
    var_8 = module_1.String()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_schema. Retrieved 17/18 statements.
# Partially parsed test_validate_with_positions_union_field. Retrieved 9/10 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
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
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 11
    var_3 = '{"value": "not a dict"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'value'
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
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 22
    var_7 = '{"user": {"name": "test"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'age'
    var_10 = module_1.Field()
    var_11 = module_1.Field()
    var_12 = {var_1: var_10, var_9: var_11}
    var_13 = module_2.Schema(var_12)
    var_14 = {var_0: var_13}
    var_15 = module_2.Schema(var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 3
    var_3 = '{"value": 123}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = var_5 | var_6
    var_8 = module_2.validate_with_positions(token=var_4, validator=var_7)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'password'
    var_2 = 'test'
    var_3 = 'pass'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 35
    var_7 = '{"username": "test", "password": "pass"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.String()
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
    var_5 = 'name'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 1
    var_3 = '{}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.String()
    var_7 = {var_5: var_6}
    var_8 = module_2.Schema(var_7)
    var_9 = module_3.validate_with_positions(token=var_4, validator=var_8)
    var_10 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 13
    var_5 = '{123: "value"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.String()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = '{"user": {}}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.String()
    var_9 = {var_7: var_8}
    var_10 = module_2.Schema(var_9)
    var_11 = {var_0: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = error.messages()[0]

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'not_a_number'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"name": 123, "age": "not_a_number"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.String()
    var_10 = module_1.Integer()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #14
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
    var_4 = 17
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
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
    var_0 = 'not_a_dict'
    var_1 = 0
    var_2 = 10
    var_3 = '{"username": "test"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
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
    var_0 = 'username'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
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
    var_0 = 'username'
    var_1 = 'password'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "password": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = False
    var_5 = module_1.Field(allow_null=var_4)
    var_6 = module_2.validate_with_positions(token=var_3, validator=var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_with_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_with_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_with_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_with_multiple_errors. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
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
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 10
    var_3 = '{"username": "john"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
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
    var_1 = 'username'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"user": {"username": 123}}'
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
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'john'
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": "john", "age": "invalid"}'
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
    var_0 = 'username'
    var_1 = 'password'
    var_2 = 'john'
    var_3 = 'secret'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 35
    var_7 = '{"username": "john", "password": "secret"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = module_2.Schema(var_11)
    var_13 = module_3.validate_with_positions(token=var_8, validator=var_12)



