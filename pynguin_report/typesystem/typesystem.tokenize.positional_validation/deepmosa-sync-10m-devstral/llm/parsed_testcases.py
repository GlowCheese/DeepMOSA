####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'user1'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"username": "user1", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = bool(var_16 == {'username': 'user1', 'age': 30})
    assert var_17 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"age": 30}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'username'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.Integer(**var_10)
    var_12 = {var_7: var_9, var_0: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)
    var_16 = len(error.messages())
    assert var_16 == 1
    var_17 = error.messages()[0]
    var_18 = var_17.code
    assert var_18 == 'required'
    var_19 = var_17.index
    var_20 = bool(var_17.index == ['username'])
    assert var_20 is True
    var_21 = var_17.text
    assert var_21 == "The field 'username' is required."
    var_22 = var_17.start_position.char_index
    assert var_22 == 0
    var_23 = var_17.end_position.char_index
    assert var_23 == 0

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 123
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"username": 123, "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 1
    var_18 = error.messages()[0]
    var_19 = var_18.code
    assert var_19 == 'type'
    var_20 = var_18.index
    var_21 = bool(var_18.index == ['username'])
    assert var_21 is True
    var_22 = var_18.text
    assert var_22 == 'Must be a string.'
    var_23 = var_18.start_position.char_index
    assert var_23 == 13
    var_24 = var_18.end_position.char_index
    assert var_24 == 15

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
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)
    var_18 = len(error.messages())
    assert var_18 == 1
    var_19 = error.messages()[0]
    var_20 = var_19.code
    assert var_20 == 'type'
    var_21 = var_19.index
    var_22 = bool(var_19.index == ['user', 'username'])
    assert var_22 is True
    var_23 = var_19.text
    assert var_23 == 'Must be a string.'
    var_24 = var_19.start_position.char_index
    assert var_24 == 18
    var_25 = var_19.end_position.char_index
    assert var_25 == 20

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'thirty'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "age": "thirty"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 2
    var_18 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_19 = var_18[0].code
    assert var_19 == 'type'
    var_20 = var_18[0].index
    var_21 = bool(var_18[0].index == ['username'])
    assert var_21 is True
    var_22 = var_18[0].text
    assert var_22 == 'Must be a string.'
    var_23 = var_18[1].code
    assert var_23 == 'type'
    var_24 = var_18[1].index
    var_25 = bool(var_18[1].index == ['age'])
    assert var_25 is True
    var_26 = var_18[1].text
    assert var_26 == 'Must be an integer.'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_a_number'
    var_1 = 0
    var_2 = 12
    var_3 = '"not_a_number"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.Integer(**var_5)
    var_7 = 5
    var_8 = {}
    var_9 = module_1.String(max_length=var_7, **var_8)
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_1.Union(var_10, **var_11)
    var_13 = module_2.validate_with_positions(token=var_4, validator=var_12)
    var_14 = len(error.messages())
    assert var_14 == 1
    var_15 = error.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'union'
    var_17 = var_15.index
    var_18 = bool(var_15.index == [])
    assert var_18 is True
    var_19 = var_15.text
    assert var_19 == 'Did not match any valid type.'
    var_20 = var_15.start_position.char_index
    assert var_20 == 0
    var_21 = var_15.end_position.char_index
    assert var_21 == 12

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.String(**var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    var_8 = len(error.messages())
    assert var_8 == 1
    var_9 = error.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'null'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [])
    assert var_12 is True
    var_13 = var_9.text
    assert var_13 == 'May not be null.'
    var_14 = var_9.start_position.char_index
    assert var_14 == 0
    var_15 = var_9.end_position.char_index
    assert var_15 == 3



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = module_0.Token(var_0, var_1, var_1)
    var_3 = False
    var_4 = module_1.Field(allow_null=var_3)
    var_5 = module_2.validate_with_positions(token=var_2, validator=var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_required_error. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_nested_type_error. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 10/11 statements.
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
    var_4 = 14
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = bool(var_12 == {'name': 'John'})
    assert var_13 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_6, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 13
    var_5 = '{"user": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_6, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.String(**var_5)
    var_7 = {}
    var_8 = module_1.Integer(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_1.Union(var_9, **var_10)
    var_12 = module_2.validate_with_positions(token=var_4, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'abc'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 24
    var_7 = '{"name": 123, "age": "abc"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)



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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = len(error.messages())
    assert var_14 == 1
    var_15 = error.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'required'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['password'])
    assert var_18 is True
    var_19 = var_15.text
    assert var_19 == "The field 'password' is required."

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.text
    assert var_14 == 'Must be an object.'

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
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)
    var_16 = len(error.messages())
    assert var_16 == 1
    var_17 = error.messages()[0]
    var_18 = var_17.code
    assert var_18 == 'required'
    var_19 = var_17.index
    var_20 = bool(var_17.index == ['user', 'name'])
    assert var_20 is True
    var_21 = var_17.text
    assert var_21 == "The field 'name' is required."

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'test'})
    assert var_12 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 3
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.Union(var_7, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    assert var_10 == 123

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.Union(var_7, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.text
    assert var_14 == 'Did not match any valid type.'



# Parsed testcases at query #6
#--------------------------




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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'john'})
    assert var_12 is True

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'required'
    var_14 = var_12.text
    assert var_14 == "The field 'username' is required."
    var_15 = var_12.index
    var_16 = bool(var_12.index == ['username'])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(line_no=1, column_no=2, char_index=1))
    assert var_20 is True

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.text
    assert var_14 == 'Must be an object.'
    var_15 = var_12.index
    var_16 = bool(var_12.index == [])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(line_no=1, column_no=11, char_index=10))
    assert var_20 is True

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 1
    var_18 = error.messages()[0]
    var_19 = var_18.code
    assert var_19 == 'type'
    var_20 = var_18.text
    assert var_20 == 'Must be an object.'
    var_21 = var_18.index
    var_22 = bool(var_18.index == ['user', 'name'])
    assert var_22 is True
    var_23 = var_18.start_position
    var_24 = bool(var_18.start_position == Position(line_no=1, column_no=13, char_index=12))
    assert var_24 is True
    var_25 = var_18.end_position
    var_26 = bool(var_18.end_position == Position(line_no=1, column_no=16, char_index=15))
    assert var_26 is True

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
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_8, validator=var_14)
    var_16 = len(error.messages())
    assert var_16 == 2
    var_17 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_18 = var_17[0].code
    assert var_18 == 'null'
    var_19 = var_17[0].text
    assert var_19 == 'May not be null.'
    var_20 = var_17[0].index
    var_21 = bool(var_17[0].index == ['username'])
    assert var_21 is True
    var_22 = var_17[1].code
    assert var_22 == 'type'
    var_23 = var_17[1].text
    assert var_23 == 'Must be an object.'
    var_24 = var_17[1].index
    var_25 = bool(var_17[1].index == ['email'])
    assert var_25 is True

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
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.Union(var_7, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.text
    assert var_14 == 'Did not match any valid type.'
    var_15 = var_12.index
    var_16 = bool(var_12.index == [])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(line_no=1, column_no=8, char_index=7))
    assert var_20 is True



# Parsed testcases at query #7
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
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.String(**var_10)
    var_12 = {var_0: var_9, var_7: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)

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
    var_6 = {}
    var_7 = module_1.Integer(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

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
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = '{"username": "test", "age": 25}'
    var_7 = module_0.Token(var_4, var_5, var_3, var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.Integer(**var_10)
    var_12 = {var_0: var_9, var_1: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_7, validator=var_14)
    var_16 = bool(var_15 == {'username': 'test', 'age': 25})
    assert var_16 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = '123'
    var_1 = 0
    var_2 = 3
    var_3 = '"123"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.String(**var_5)
    var_7 = {}
    var_8 = module_1.Integer(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_1.Union(var_9, **var_10)
    var_12 = module_2.validate_with_positions(token=var_4, validator=var_11)
    assert var_12 == '123'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 7
    var_6 = '[1, 2, 3]'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.Integer(**var_10)
    var_12 = [var_9, var_11]
    var_13 = {}
    var_14 = module_1.Union(var_12, **var_13)
    var_15 = module_2.validate_with_positions(token=var_7, validator=var_14)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


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
    var_7 = 'email'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.String(**var_10)
    var_12 = {var_0: var_9, var_7: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)

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
    var_7 = {}
    var_8 = module_1.Integer(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_6, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'john'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"username": "john", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = bool(var_16 == {'username': 'john', 'age': 30})
    assert var_17 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "email": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.String(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)



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
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

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
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)

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
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)

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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'name': 'John'})
    assert var_12 is True

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
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)

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
    var_6 = 'age'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_4, validator=var_11)
    var_13 = 'name'
    var_14 = 'age'
    var_15 = {var_13, var_14}

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'value'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = '{"value": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = var_7 | var_8
    var_10 = {var_0: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = bool(var_13 == {'value': 123})
    assert var_14 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'value'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
    var_5 = '{"value": "invalid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = var_7 | var_8
    var_10 = {var_0: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.
# Partially parsed test_validate_with_positions_null_input_without_allow_null. Retrieved 10/11 statements.


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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.Integer(**var_10)
    var_12 = {var_0: var_9, var_7: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)

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
    var_5 = 'name'
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

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
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)

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
    var_6 = 25
    var_7 = '{"name": "John", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = bool(var_16 == {'name': 'John', 'age': 30})
    assert var_17 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'thirty'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"name": 123, "age": "thirty"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.Integer(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = True
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_2.Schema(var_8, **var_11)
    var_13 = module_3.validate_with_positions(token=var_4, validator=var_12)
    assert var_13 is None

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)



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
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_required_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'user1'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"username": "user1"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = bool(var_12 == {'username': 'user1'})
    assert var_13 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

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
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)

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
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": 123, "email": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.String(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)



# Parsed testcases at query #14
#--------------------------




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
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = bool(var_12 == {'username': 'john'})
    assert var_13 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)
    var_12 = len(error.messages())
    assert var_12 == 1
    var_13 = error.messages()[0]
    var_14 = var_13.code
    assert var_14 == 'required'
    var_15 = var_13.text
    assert var_15 == "The field 'username' is required."
    var_16 = var_13.index
    var_17 = bool(var_13.index == ['username'])
    assert var_17 is True
    var_18 = var_13.start_position
    var_19 = bool(var_13.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_19 is True
    var_20 = var_13.end_position
    var_21 = bool(var_13.end_position == Position(line_no=1, column_no=2, char_index=1))
    assert var_21 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 13
    var_5 = '{"username": 123}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = len(error.messages())
    assert var_13 == 1
    var_14 = error.messages()[0]
    var_15 = var_14.code
    assert var_15 == 'type'
    var_16 = var_14.text
    assert var_16 == 'Must be a string.'
    var_17 = var_14.index
    var_18 = bool(var_14.index == ['username'])
    assert var_18 is True
    var_19 = var_14.start_position
    var_20 = bool(var_14.start_position == Position(line_no=1, column_no=12, char_index=11))
    assert var_20 is True
    var_21 = var_14.end_position
    var_22 = bool(var_14.end_position == Position(line_no=1, column_no=15, char_index=13))
    assert var_22 is True

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
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)
    var_18 = len(error.messages())
    assert var_18 == 1
    var_19 = error.messages()[0]
    var_20 = var_19.code
    assert var_20 == 'type'
    var_21 = var_19.text
    assert var_21 == 'Must be a string.'
    var_22 = var_19.index
    var_23 = bool(var_19.index == ['user', 'name'])
    assert var_23 is True
    var_24 = var_19.start_position
    var_25 = bool(var_19.start_position == Position(line_no=1, column_no=17, char_index=16))
    assert var_25 is True
    var_26 = var_19.end_position
    var_27 = bool(var_19.end_position == Position(line_no=1, column_no=20, char_index=19))
    assert var_27 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = 123
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"username": 123, "email": null}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {}
    var_12 = module_1.String(**var_11)
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 2
    var_18 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_19 = var_18[0].code
    assert var_19 == 'type'
    var_20 = var_18[0].text
    assert var_20 == 'Must be a string.'
    var_21 = var_18[0].index
    var_22 = bool(var_18[0].index == ['username'])
    assert var_22 is True
    var_23 = var_18[0].start_position
    var_24 = bool(var_18[0].start_position == Position(line_no=1, column_no=12, char_index=11))
    assert var_24 is True
    var_25 = var_18[0].end_position
    var_26 = bool(var_18[0].end_position == Position(line_no=1, column_no=15, char_index=13))
    assert var_26 is True
    var_27 = var_18[1].code
    assert var_27 == 'type'
    var_28 = var_18[1].text
    assert var_28 == 'Must be a string.'
    var_29 = var_18[1].index
    var_30 = bool(var_18[1].index == ['email'])
    assert var_30 is True
    var_31 = var_18[1].start_position
    var_32 = bool(var_18[1].start_position == Position(line_no=1, column_no=24, char_index=23))
    assert var_32 is True
    var_33 = var_18[1].end_position
    var_34 = bool(var_18[1].end_position == Position(line_no=1, column_no=28, char_index=25))
    assert var_34 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = {}
    var_6 = module_1.String(**var_5)
    var_7 = {}
    var_8 = module_1.Integer(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_1.Union(var_9, **var_10)
    var_12 = module_2.validate_with_positions(token=var_4, validator=var_11)
    var_13 = len(error.messages())
    assert var_13 == 1
    var_14 = error.messages()[0]
    var_15 = var_14.code
    assert var_15 == 'union'
    var_16 = var_14.text
    assert var_16 == 'Did not match any valid type.'
    var_17 = var_14.index
    var_18 = bool(var_14.index == [])
    assert var_18 is True
    var_19 = var_14.start_position
    var_20 = bool(var_14.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_20 is True
    var_21 = var_14.end_position
    var_22 = bool(var_14.end_position == Position(line_no=1, column_no=4, char_index=2))
    assert var_22 is True



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
    var_4 = module_1.Field()
    var_5 = module_2.validate_with_positions(token=var_3, validator=var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 15/16 statements.


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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'name': 'John'})
    assert var_12 is True

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
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.text
    assert var_13 == "The field 'name' is required."
    var_14 = var_12.code
    assert var_14 == 'required'
    var_15 = var_12.index
    var_16 = bool(var_12.index == ['name'])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(1, 1, 0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(1, 1, 0))
    assert var_20 is True

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
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.text
    assert var_13 == 'Must be an object.'
    var_14 = var_12.code
    assert var_14 == 'type'
    var_15 = var_12.index
    var_16 = bool(var_12.index == [])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(1, 1, 0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(1, 11, 10))
    assert var_20 is True

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
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.text
    assert var_13 == 'May not be null.'
    var_14 = var_12.code
    assert var_14 == 'null'
    var_15 = var_12.index
    var_16 = bool(var_12.index == [])
    assert var_16 is True
    var_17 = var_12.start_position
    var_18 = bool(var_12.start_position == Position(1, 1, 0))
    assert var_18 is True
    var_19 = var_12.end_position
    var_20 = bool(var_12.end_position == Position(1, 4, 3))
    assert var_20 is True

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 1
    var_18 = error.messages()[0]
    var_19 = var_18.index
    var_20 = bool(var_18.index == ['user', 'name'])
    assert var_20 is True
    var_21 = var_18.start_position
    var_22 = bool(var_18.start_position == Position(1, 13, 12))
    assert var_22 is True
    var_23 = var_18.end_position
    var_24 = bool(var_18.end_position == Position(1, 16, 15))
    assert var_24 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = None
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"name": null, "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = False
    var_10 = module_1.Field(allow_null=var_9)
    var_11 = module_1.Field()
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_8, validator=var_14)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/19 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 4/14 statements.
# Partially parsed test_validate_with_positions_valid_input. Retrieved 9/18 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '{"password": "test"}'
    var_3 = 'username'
    var_4 = 'password'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '123'
    var_3 = {}
    var_4 = module_0.String(**var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = '{"username": "test", "password": "test"}'
    var_3 = 'username'
    var_4 = 'password'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 15/16 statements.


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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'name': 'John'})
    assert var_12 is True

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
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 4
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = False
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    var_8 = len(error.messages())
    assert var_8 == 1
    var_9 = error.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'null'
    var_11 = var_9.text
    assert var_11 == 'May not be null.'
    var_12 = var_9.start_position.char_index
    assert var_12 == 0
    var_13 = var_9.end_position.char_index
    assert var_13 == 4

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'required'
    var_14 = var_12.text
    assert var_14 == "The field 'name' is required."
    var_15 = var_12.index
    var_16 = bool(var_12.index == ['name'])
    assert var_16 is True
    var_17 = var_12.start_position.char_index
    assert var_17 == 0
    var_18 = var_12.end_position.char_index
    assert var_18 == 2

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 12
    var_3 = '"not a dict"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.text
    assert var_14 == 'Must be an object.'
    var_15 = var_12.start_position.char_index
    assert var_15 == 0
    var_16 = var_12.end_position.char_index
    assert var_16 == 12

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"user": {}}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'name'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)
    var_16 = len(error.messages())
    assert var_16 == 1
    var_17 = error.messages()[0]
    var_18 = var_17.code
    assert var_18 == 'required'
    var_19 = var_17.text
    assert var_19 == "The field 'name' is required."
    var_20 = var_17.index
    var_21 = bool(var_17.index == ['user', 'name'])
    assert var_21 is True
    var_22 = var_17.start_position.char_index
    assert var_22 == 7
    var_23 = var_17.end_position.char_index
    assert var_23 == 9

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = None
    var_3 = 'invalid'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"name": null, "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = False
    var_10 = module_1.Field(allow_null=var_9)
    var_11 = module_1.Field()
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_8, validator=var_14)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 8
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.Union(var_7, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.text
    assert var_14 == 'Did not match any valid type.'
    var_15 = var_12.start_position.char_index
    assert var_15 == 0
    var_16 = var_12.end_position.char_index
    assert var_16 == 8

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 8
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Union(var_6, **var_7)
    var_9 = module_2.validate_with_positions(token=var_4, validator=var_8)
    var_10 = len(error.messages())
    assert var_10 == 1
    var_11 = error.messages()[0]
    var_12 = var_11.code
    assert var_12 == 'type'
    var_13 = var_11.text
    assert var_13 == 'Must be an object.'
    var_14 = var_11.start_position.char_index
    assert var_14 == 0
    var_15 = var_11.end_position.char_index
    assert var_15 == 8



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_null_value. Retrieved 11/12 statements.
# Partially parsed test_validate_with_positions_nested_schema. Retrieved 17/18 statements.
# Partially parsed test_validate_with_positions_union_field. Retrieved 9/10 statements.


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
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'age'
    var_1 = 'not_a_number'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"age": "not_a_number"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
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
    var_6 = False
    var_7 = module_1.Field(allow_null=var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)

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
    var_6 = 25
    var_7 = '{"name": "John", "age": 30}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)
    var_15 = bool(var_14 == {'name': 'John', 'age': 30})
    assert var_15 is True

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
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)

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
    var_10 = module_1.Field()
    var_11 = module_1.Field()
    var_12 = {var_1: var_10, var_9: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = {var_0: var_14}
    var_16 = {}
    var_17 = module_2.Schema(var_15, **var_16)
    var_18 = module_3.validate_with_positions(token=var_8, validator=var_17)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = 12
    var_3 = '"not_an_int"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = var_5 | var_6
    var_8 = module_2.validate_with_positions(token=var_4, validator=var_7)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'age'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

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
    var_5 = 'name'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)

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
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)

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
    var_6 = 'age'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_4, validator=var_11)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'name': 'test'})
    assert var_12 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/25 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 5/22 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 12/29 statements.
# Partially parsed test_validate_with_positions_valid_input. Retrieved 8/24 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 9/26 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'name'
    var_4 = False
    var_5 = module_0.Field(allow_null=var_4)
    var_6 = {var_3: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = len(error.messages())
    assert var_9 == 1
    var_10 = error.messages()[0].code
    assert var_10 == 'required'
    var_11 = error.messages()[0].text
    assert var_11 == "The field 'name' is required."
    var_12 = error.messages()[0].start_position.char_index
    assert var_12 == 0
    var_13 = error.messages()[0].end_position.char_index
    assert var_13 == 0

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = ''
    var_3 = {}
    var_4 = {}
    var_5 = module_0.Schema(var_3, **var_4)
    var_6 = len(error.messages())
    assert var_6 == 1
    var_7 = error.messages()[0].code
    assert var_7 == 'type'
    var_8 = error.messages()[0].text
    assert var_8 == 'Must be an object.'
    var_9 = error.messages()[0].start_position.char_index
    assert var_9 == 0
    var_10 = error.messages()[0].end_position.char_index
    assert var_10 == 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = module_0.Field()
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = len(error.messages())
    assert var_14 == 1
    var_15 = error.messages()[0].code
    assert var_15 == 'type'
    var_16 = error.messages()[0].text
    assert var_16 == 'Must be an object.'
    var_17 = error.messages()[0].start_position.char_index
    assert var_17 == 0
    var_18 = error.messages()[0].end_position.char_index
    assert var_18 == 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = ''
    var_5 = module_0.Field()
    var_6 = {var_0: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'name'
    var_4 = 'age'
    var_5 = module_0.Field()
    var_6 = module_0.Field()
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = len(error.messages())
    assert var_10 == 2
    var_11 = error.messages()[0].code
    assert var_11 == 'required'
    var_12 = error.messages()[0].text
    assert var_12 == "The field 'name' is required."
    var_13 = error.messages()[1].code
    assert var_13 == 'required'
    var_14 = error.messages()[1].text
    assert var_14 == "The field 'age' is required."



# Parsed testcases at query #6
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
    var_4 = 15
    var_5 = '{"username": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = len(error.messages())
    assert var_14 == 1
    var_15 = error.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'required'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['password'])
    assert var_18 is True
    var_19 = var_15.start_position
    var_20 = bool(var_15.start_position == var_6.lookup(['password']).start)
    assert var_20 is True
    var_21 = var_15.end_position
    var_22 = bool(var_15.end_position == var_6.lookup(['password']).end)
    assert var_22 is True

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'type'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [])
    assert var_15 is True
    var_16 = var_12.start_position
    var_17 = bool(var_12.start_position == var_4.start)
    assert var_17 is True
    var_18 = var_12.end_position
    var_19 = bool(var_12.end_position == var_4.end)
    assert var_19 is True

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 1
    var_18 = error.messages()[0]
    var_19 = var_18.code
    assert var_19 == 'type'
    var_20 = var_18.index
    var_21 = bool(var_18.index == ['user', 'name'])
    assert var_21 is True
    var_22 = var_18.start_position
    var_23 = bool(var_18.start_position == var_8.lookup(['user', 'name']).start)
    assert var_23 is True
    var_24 = var_18.end_position
    var_25 = bool(var_18.end_position == var_8.lookup(['user', 'name']).end)
    assert var_25 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 3
    var_3 = '123'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = module_1.Field()
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_1.Union(var_7, **var_8)
    var_10 = module_2.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [])
    assert var_15 is True
    var_16 = var_12.start_position
    var_17 = bool(var_12.start_position == var_4.start)
    assert var_17 is True
    var_18 = var_12.end_position
    var_19 = bool(var_12.end_position == var_4.end)
    assert var_19 is True

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'test'})
    assert var_12 is True

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
    var_8 = True
    var_9 = 'allow_null'
    var_10 = {var_9: var_8}
    var_11 = module_2.Schema(var_7, **var_10)
    var_12 = module_3.validate_with_positions(token=var_4, validator=var_11)
    assert var_12 is None

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = len(error.messages())
    assert var_11 == 1
    var_12 = error.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'null'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [])
    assert var_15 is True
    var_16 = var_12.start_position
    var_17 = bool(var_12.start_position == var_4.start)
    assert var_17 is True
    var_18 = var_12.end_position
    var_19 = bool(var_12.end_position == var_4.end)
    assert var_19 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{123: "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'username'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = len(error.messages())
    assert var_13 == 1
    var_14 = error.messages()[0]
    var_15 = var_14.code
    assert var_15 == 'invalid_key'
    var_16 = var_14.index
    var_17 = bool(var_14.index == [123])
    assert var_17 is True
    var_18 = var_14.start_position
    var_19 = bool(var_14.start_position == var_6.lookup([123]).start)
    assert var_19 is True
    var_20 = var_14.end_position
    var_21 = bool(var_14.end_position == var_6.lookup([123]).end)
    assert var_21 is True



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_field. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 14/15 statements.


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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'not a dict'
    var_1 = 0
    var_2 = 10
    var_3 = '{"username": "test"}'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = 'username'
    var_6 = module_1.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'test'})
    assert var_12 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 'not a number'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"username": "test", "age": "not a number"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_custom_error_message. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_positional_ordering. Retrieved 14/15 statements.


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
    var_7 = 'email'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)
    var_14 = len(error.messages())
    assert var_14 == 1
    var_15 = error.messages()[0]
    var_16 = var_15.code
    assert var_16 == 'required'
    var_17 = var_15.index
    var_18 = bool(var_15.index == ['email'])
    assert var_18 is True
    var_19 = var_15.start_position
    var_20 = bool(var_15.start_position == var_6.start)
    assert var_20 is True
    var_21 = var_15.end_position
    var_22 = bool(var_15.end_position == var_6.end)
    assert var_22 is True
    var_23 = var_15.text
    assert var_23 == "The field 'email' is required."

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 123
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{123: "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'username'
    var_8 = module_1.Field()
    var_9 = {var_7: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = len(error.messages())
    assert var_13 == 1
    var_14 = error.messages()[0]
    var_15 = var_14.code
    assert var_15 == 'invalid_key'
    var_16 = var_14.index
    var_17 = bool(var_14.index == [123])
    assert var_17 is True
    var_18 = var_14.start_position
    var_19 = bool(var_14.start_position == var_6.start)
    assert var_19 is True
    var_20 = var_14.end_position
    var_21 = bool(var_14.end_position == var_6.end)
    assert var_21 is True
    var_22 = var_14.text
    assert var_22 == 'All object keys must be strings.'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 'john'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 25
    var_7 = '{"user": {"name": "john"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'email'
    var_10 = module_1.Field()
    var_11 = {var_9: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)
    var_18 = len(error.messages())
    assert var_18 == 1
    var_19 = error.messages()[0]
    var_20 = var_19.code
    assert var_20 == 'required'
    var_21 = var_19.index
    var_22 = bool(var_19.index == ['user', 'email'])
    assert var_22 is True
    var_23 = var_19.start_position
    var_24 = bool(var_19.start_position == var_8.lookup(['user']).start)
    assert var_24 is True
    var_25 = var_19.end_position
    var_26 = bool(var_19.end_position == var_8.lookup(['user']).end)
    assert var_26 is True
    var_27 = var_19.text
    assert var_27 == "The field 'email' is required."

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_1.Field()
    var_11 = [var_9, var_10]
    var_12 = {}
    var_13 = module_1.Union(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_4, validator=var_13)
    var_15 = len(error.messages())
    assert var_15 == 1
    var_16 = error.messages()[0]
    var_17 = var_16.code
    assert var_17 == 'union'
    var_18 = var_16.index
    var_19 = bool(var_16.index == [])
    assert var_19 is True
    var_20 = var_16.start_position
    var_21 = bool(var_16.start_position == var_4.start)
    assert var_21 is True
    var_22 = var_16.end_position
    var_23 = bool(var_16.end_position == var_4.end)
    assert var_23 is True
    var_24 = var_16.text
    assert var_24 == 'Did not match any valid type.'

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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'john'})
    assert var_12 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = True
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    assert var_7 is None

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = False
    var_6 = module_1.Field(allow_null=var_5)
    var_7 = module_2.validate_with_positions(token=var_4, validator=var_6)
    var_8 = len(error.messages())
    assert var_8 == 1
    var_9 = error.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'null'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [])
    assert var_12 is True
    var_13 = var_9.start_position
    var_14 = bool(var_9.start_position == var_4.start)
    assert var_14 is True
    var_15 = var_9.end_position
    var_16 = bool(var_9.end_position == var_4.end)
    assert var_16 is True
    var_17 = var_9.text
    assert var_17 == 'May not be null.'

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
    var_6 = 'email'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_4, validator=var_11)
    var_13 = len(error.messages())
    assert var_13 == 2
    var_14 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_15 = var_14[0].code
    assert var_15 == 'required'
    var_16 = var_14[0].index
    var_17 = bool(var_14[0].index == ['email'])
    assert var_17 is True
    var_18 = var_14[1].code
    assert var_18 == 'required'
    var_19 = var_14[1].index
    var_20 = bool(var_14[1].index == ['username'])
    assert var_20 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 6
    var_3 = '"invalid"'
    var_4 = module_0.Token(var_0, var_1, var_2, var_3)
    var_5 = module_1.Field()
    var_6 = 'type'
    var_7 = 'Custom error message.'
    var_8 = module_2.validate_with_positions(token=var_4, validator=var_5)
    var_9 = len(error.messages())
    assert var_9 == 1
    var_10 = error.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'type'
    var_12 = var_10.index
    var_13 = bool(var_10.index == [])
    assert var_13 is True
    var_14 = var_10.start_position
    var_15 = bool(var_10.start_position == var_4.start)
    assert var_15 is True
    var_16 = var_10.end_position
    var_17 = bool(var_10.end_position == var_4.end)
    assert var_17 is True
    var_18 = var_10.text
    assert var_18 == 'Custom error message.'

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'email'
    var_1 = 'username'
    var_2 = 'john'
    var_3 = 'doe'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"email": "john", "username": "doe"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = module_1.Field()
    var_10 = module_1.Field()
    var_11 = {var_1: var_9, var_0: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_8, validator=var_13)



# Parsed testcases at query #12
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
    var_4 = 15
    var_5 = '{"invalid": "data"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'valid'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)



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
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {}
    var_11 = module_1.String(**var_10)
    var_12 = {var_0: var_9, var_7: var_11}
    var_13 = {}
    var_14 = module_2.Schema(var_12, **var_13)
    var_15 = module_3.validate_with_positions(token=var_6, validator=var_14)
    var_16 = len(error.messages())
    assert var_16 == 1
    var_17 = error.messages()[0]
    var_18 = var_17.text
    assert var_18 == "The field 'password' is required."
    var_19 = var_17.code
    assert var_19 == 'required'
    var_20 = var_17.index
    var_21 = bool(var_17.index == ['password'])
    assert var_21 is True
    var_22 = var_17.start_position
    var_23 = bool(var_17.start_position == Position(line_no=1, column_no=18, char_index=17))
    assert var_23 is True
    var_24 = var_17.end_position
    var_25 = bool(var_17.end_position == Position(line_no=1, column_no=26, char_index=25))
    assert var_25 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)
    var_12 = len(error.messages())
    assert var_12 == 1
    var_13 = error.messages()[0]
    var_14 = var_13.text
    assert var_14 == 'Must be an object.'
    var_15 = var_13.code
    assert var_15 == 'type'
    var_16 = var_13.index
    var_17 = bool(var_13.index == [])
    assert var_17 is True
    var_18 = var_13.start_position
    var_19 = bool(var_13.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_19 is True
    var_20 = var_13.end_position
    var_21 = bool(var_13.end_position == Position(line_no=1, column_no=11, char_index=10))
    assert var_21 is True

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
    var_6 = 24
    var_7 = '{"user": {"username": 123}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_1: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = {var_0: var_13}
    var_15 = {}
    var_16 = module_2.Schema(var_14, **var_15)
    var_17 = module_3.validate_with_positions(token=var_8, validator=var_16)
    var_18 = len(error.messages())
    assert var_18 == 1
    var_19 = error.messages()[0]
    var_20 = var_19.text
    assert var_20 == 'Must be a string.'
    var_21 = var_19.code
    assert var_21 == 'type'
    var_22 = var_19.index
    var_23 = bool(var_19.index == ['user', 'username'])
    assert var_23 is True
    var_24 = var_19.start_position
    var_25 = bool(var_19.start_position == Position(line_no=1, column_no=18, char_index=17))
    assert var_25 is True
    var_26 = var_19.end_position
    var_27 = bool(var_19.end_position == Position(line_no=1, column_no=21, char_index=20))
    assert var_27 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'john'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 17
    var_5 = '{"username": "john"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = bool(var_12 == {'username': 'john'})
    assert var_13 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = True
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_2.Schema(var_8, **var_11)
    var_13 = module_3.validate_with_positions(token=var_4, validator=var_12)
    assert var_13 is None

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)
    var_12 = len(error.messages())
    assert var_12 == 1
    var_13 = error.messages()[0]
    var_14 = var_13.text
    assert var_14 == 'May not be null.'
    var_15 = var_13.code
    assert var_15 == 'null'
    var_16 = var_13.index
    var_17 = bool(var_13.index == [])
    assert var_17 is True
    var_18 = var_13.start_position
    var_19 = bool(var_13.start_position == Position(line_no=1, column_no=1, char_index=0))
    assert var_19 is True
    var_20 = var_13.end_position
    var_21 = bool(var_13.end_position == Position(line_no=1, column_no=4, char_index=3))
    assert var_21 is True



# Parsed testcases at query #15
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
    var_4 = 12
    var_5 = '{"name": "John"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_6, validator=var_11)
    var_13 = bool(var_12 == {'name': 'John'})
    assert var_13 is True

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
    var_6 = {}
    var_7 = module_1.String(**var_6)
    var_8 = {var_5: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_4, validator=var_10)
    var_12 = len(error.messages())
    assert var_12 == 1
    var_13 = error.messages()[0]
    var_14 = var_13.text
    assert var_14 == "The field 'name' is required."
    var_15 = var_13.code
    assert var_15 == 'required'
    var_16 = var_13.index
    var_17 = bool(var_13.index == ['name'])
    assert var_17 is True
    var_18 = var_13.start_position
    var_19 = bool(var_13.start_position == Position(1, 1, 0))
    assert var_19 is True
    var_20 = var_13.end_position
    var_21 = bool(var_13.end_position == Position(1, 1, 0))
    assert var_21 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 8
    var_6 = '[1, 2, 3]'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = 'name'
    var_9 = {}
    var_10 = module_1.String(**var_9)
    var_11 = {var_8: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_7, validator=var_13)
    var_15 = len(error.messages())
    assert var_15 == 1
    var_16 = error.messages()[0]
    var_17 = var_16.text
    assert var_17 == 'Must be an object.'
    var_18 = var_16.code
    assert var_18 == 'type'
    var_19 = var_16.index
    var_20 = bool(var_16.index == [])
    assert var_20 is True
    var_21 = var_16.start_position
    var_22 = bool(var_16.start_position == Position(1, 1, 0))
    assert var_22 is True
    var_23 = var_16.end_position
    var_24 = bool(var_16.end_position == Position(1, 8, 8))
    assert var_24 is True

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
    var_8 = {}
    var_9 = module_1.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_6, validator=var_15)
    var_17 = len(error.messages())
    assert var_17 == 1
    var_18 = error.messages()[0]
    var_19 = var_18.text
    assert var_19 == "The field 'name' is required."
    var_20 = var_18.code
    assert var_20 == 'required'
    var_21 = var_18.index
    var_22 = bool(var_18.index == ['user', 'name'])
    assert var_22 is True
    var_23 = var_18.start_position
    var_24 = bool(var_18.start_position == Position(1, 9, 8))
    assert var_24 is True
    var_25 = var_18.end_position
    var_26 = bool(var_18.end_position == Position(1, 9, 8))
    assert var_26 is True

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
    var_6 = 'age'
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = {}
    var_10 = module_1.Integer(**var_9)
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = module_3.validate_with_positions(token=var_4, validator=var_13)
    var_15 = len(error.messages())
    assert var_15 == 2
    var_16 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_17 = var_16[0].text
    assert var_17 == "The field 'age' is required."
    var_18 = var_16[0].code
    assert var_18 == 'required'
    var_19 = var_16[0].index
    var_20 = bool(var_16[0].index == ['age'])
    assert var_20 is True
    var_21 = var_16[1].text
    assert var_21 == "The field 'name' is required."
    var_22 = var_16[1].code
    assert var_22 == 'required'
    var_23 = var_16[1].index
    var_24 = bool(var_16[1].index == ['name'])
    assert var_24 is True



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_type_error. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 13/14 statements.


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
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'user'})
    assert var_12 is True

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)
    var_17 = bool(False)
    assert var_17 is True

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'email'
    var_2 = None
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 0
    var_5 = 30
    var_6 = '{"username": null, "email": null}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_7, validator=var_12)
    var_14 = bool(False)
    assert var_14 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_positions_required_message. Retrieved 12/13 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = '{"a": 1}'
    var_5 = module_0.Token(var_2, var_3, var_3, var_4)
    var_6 = 'b'
    var_7 = module_1.Field()
    var_8 = module_1.Field()
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = module_3.validate_with_positions(token=var_5, validator=var_11)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 13/14 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 10/11 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 15/16 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 16/17 statements.


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'user'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "user"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = 'password'
    var_8 = module_1.Field()
    var_9 = module_1.Field()
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = module_3.validate_with_positions(token=var_6, validator=var_12)

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
    var_8 = {}
    var_9 = module_2.Schema(var_7, **var_8)
    var_10 = module_3.validate_with_positions(token=var_4, validator=var_9)

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
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)

import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.schemas as module_2
import typesystem.tokenize.positional_validation as module_3

def test_case_0():
    var_0 = 'username'
    var_1 = 'user'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"username": "user"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = module_1.Field()
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = module_3.validate_with_positions(token=var_6, validator=var_10)
    var_12 = bool(var_11 == {'username': 'user'})
    assert var_12 is True

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
    var_6 = 35
    var_7 = '{"username": "user", "age": "invalid"}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = 'email'
    var_10 = module_1.Field()
    var_11 = module_1.Field()
    var_12 = module_1.Field()
    var_13 = {var_0: var_10, var_1: var_11, var_9: var_12}
    var_14 = {}
    var_15 = module_2.Schema(var_13, **var_14)
    var_16 = module_3.validate_with_positions(token=var_8, validator=var_15)



# Parsed testcases at query #20
#--------------------------




import typesystem.base as module_0

def test_case_0():
    var_0 = 'Error'
    var_1 = 'type'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    var_5 = bool(not var_4.code == 'required')
    assert var_5 is True



