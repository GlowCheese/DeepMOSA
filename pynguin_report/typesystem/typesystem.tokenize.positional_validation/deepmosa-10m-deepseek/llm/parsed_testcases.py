####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 5/23 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 6/24 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 10/30 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 10/28 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/30 statements.
# Partially parsed test_validate_with_positions_allow_null. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 10/28 statements.


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = '{}'
    var_4 = module_0.Token(var_1, var_2, var_2, var_3)


def test_case_0():
    var_0 = 'name'
    var_1 = 'not an object'
    var_2 = 0
    var_3 = 12
    var_4 = '"not an object"'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)


def test_case_0():
    var_0 = 'inner'
    var_1 = 'outer'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 15
    var_6 = '{"outer": {}}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = [var_1]
    var_9 = var_7.lookup(var_8)


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'bad'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"name": "bad"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = [var_3]
    var_11 = var_9.lookup(var_10)


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'bad'
    var_6 = {var_4: var_5, var_3: var_5}
    var_7 = 0
    var_8 = 23
    var_9 = '{"b": "bad", "a": "bad"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = 1


def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = 'null'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 15
    var_6 = '{123: "value"}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = [var_1]
    var_9 = var_7.lookup_key(var_8)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 12/46 statements.


def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = 10
    var_7 = 20
    var_8 = ' '
    var_9 = 30
    var_10 = var_8 * var_9
    var_11 = [var_3]
    var_12 = 5
    var_13 = var_8 * var_9



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/12 statements.
# Partially parsed test_validate_with_positions_validation_error_without_index. Retrieved 5/14 statements.
# Partially parsed test_validate_with_positions_validation_error_with_index. Retrieved 7/20 statements.
# Partially parsed test_validate_with_positions_required_error. Retrieved 5/19 statements.
# Partially parsed test_validate_with_positions_multiple_messages_sorted. Retrieved 8/26 statements.



def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)


def test_case_0():
    var_0 = 'bad'
    var_1 = 0
    var_2 = 2
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0]
    var_7 = var_6.text
    assert var_7 == 'Invalid value'
    var_8 = var_6.code
    assert var_8 == 'invalid'
    var_9 = var_6.index
    var_10 = bool(var_6.index == [])
    assert var_10 is True
    var_11 = var_6.start_position
    var_12 = bool(var_6.start_position == var_3.start)
    assert var_12 is True
    var_13 = var_6.end_position
    var_14 = bool(var_6.end_position == var_3.end)
    assert var_14 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"key": "val"}'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0]
    var_9 = var_8.text
    assert var_9 == 'Field error'
    var_10 = var_8.code
    assert var_10 == 'error'
    var_11 = var_8.index
    var_12 = bool(var_8.index == ['key'])
    assert var_12 is True
    var_13 = var_8.start_position.char_index
    assert var_13 == 5
    var_14 = var_8.end_position.char_index
    assert var_14 == 9

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 2
    var_3 = '{}'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0]
    var_7 = var_6.text
    assert var_7 == "The field 'field' is required."
    var_8 = var_6.code
    assert var_8 == 'required'
    var_9 = var_6.index
    var_10 = bool(var_6.index == ['field'])
    assert var_10 is True
    var_11 = var_6.start_position.char_index
    assert var_11 == 1
    var_12 = var_6.end_position.char_index
    assert var_12 == 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"a":1,"b":2}'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 12/29 statements.
# Partially parsed test_validate_with_positions_non_required_error. Retrieved 7/28 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 8/37 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 6/34 statements.


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
    var_6 = 'custom'
    var_7 = 'Custom error'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'invalid'
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 0
    var_5 = 25
    var_6 = ''
    var_7 = 'custom'
    var_8 = 'Custom error'
    var_9 = {var_7: var_8}

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 9/27 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_field_type_error. Retrieved 6/21 statements.
# Partially parsed test_validate_with_positions_schema_nested_field_error. Retrieved 10/30 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 7/30 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 8/21 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 7/17 statements.
# Partially parsed test_validate_with_positions_null_not_allowed. Retrieved 6/21 statements.


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 12
    var_9 = '{123: "value"}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 123
    var_4 = 0
    var_5 = 2
    var_6 = '123'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = 123
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 24
    var_10 = '{"outer": {"inner": 123}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"name": "test"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)


def test_case_0():
    var_0 = 'null'
    var_1 = 'May not be null.'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = 0
    var_6 = 3
    var_7 = 'null'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)


def test_case_0():
    var_0 = 'null'
    var_1 = 'May not be null.'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = None
    var_5 = 3
    var_6 = 'null'
    var_7 = module_0.Token(var_4, var_3, var_5, var_6)



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = {}
    var_2 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 11/26 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 14/42 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 18/42 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 10/24 statements.


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = 'required_field'
    var_7 = False
    var_8 = module_0.Field(allow_null=var_7)
    var_9 = {var_6: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)


def test_case_0():
    var_0 = 'not an object'
    var_1 = 0
    var_2 = 15
    var_3 = ''
    var_4 = 'some_field'
    var_5 = False
    var_6 = module_0.Field(allow_null=var_5)
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)


def test_case_0():
    var_0 = 'nested'
    var_1 = 'field'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 40
    var_7 = ''
    var_8 = False
    var_9 = module_0.Field(allow_null=var_8)
    var_10 = {var_1: var_9}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = {var_0: var_12}
    var_14 = {}
    var_15 = module_1.Schema(var_13, **var_14)


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = ''
    var_8 = 5
    var_9 = 6
    var_10 = 12
    var_11 = 13
    var_12 = False
    var_13 = module_0.Field(allow_null=var_12)
    var_14 = False
    var_15 = module_0.Field(allow_null=var_14)
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = {}
    var_18 = module_1.Schema(var_16, **var_17)


def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = ''
    var_4 = False
    var_5 = module_0.Field(allow_null=var_4)
    var_6 = False
    var_7 = module_0.Field(allow_null=var_6)
    var_8 = [var_5, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 10/28 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 9/24 statements.
# Partially parsed test_validate_with_positions_field_type_error. Retrieved 7/22 statements.
# Partially parsed test_validate_with_positions_schema_nested_field_error. Retrieved 11/31 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/31 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/18 statements.


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
    var_11 = error.messages()[var_7]
    var_12 = var_11.text
    assert var_12 == "The field 'name' is required."
    var_13 = var_11.code
    assert var_13 == 'required'
    var_14 = var_11.index
    var_15 = bool(var_11.index == ['name'])
    assert var_15 is True
    var_16 = var_11.start_position.line_no
    assert var_16 == 1
    var_17 = var_11.start_position.column_no
    assert var_17 == 1
    var_18 = var_11.start_position.char_index
    assert var_18 == 0
    var_19 = var_11.end_position.line_no
    assert var_19 == 1
    var_20 = var_11.end_position.column_no
    assert var_20 == 12
    var_21 = var_11.end_position.char_index
    assert var_21 == 11


def test_case_0():
    var_0 = 'name'
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{1: "value"}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.text
    assert var_9 == 'All object keys must be strings.'
    var_10 = var_8.code
    assert var_10 == 'invalid_key'
    var_11 = var_8.index
    var_12 = bool(var_8.index == [1])
    assert var_12 is True
    var_13 = var_8.start_position.line_no
    assert var_13 == 1
    var_14 = var_8.start_position.column_no
    assert var_14 == 1
    var_15 = var_8.start_position.char_index
    assert var_15 == 0
    var_16 = var_8.end_position.line_no
    assert var_16 == 1
    var_17 = var_8.end_position.column_no
    assert var_17 == 12
    var_18 = var_8.end_position.char_index
    assert var_18 == 11


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 123
    var_4 = 0
    var_5 = 2
    var_6 = '123'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.text
    assert var_9 == 'Must be a string.'
    var_10 = var_8.code
    assert var_10 == 'type'
    var_11 = var_8.index
    var_12 = bool(var_8.index == [])
    assert var_12 is True
    var_13 = var_8.start_position.line_no
    assert var_13 == 1
    var_14 = var_8.start_position.column_no
    assert var_14 == 1
    var_15 = var_8.start_position.char_index
    assert var_15 == 0
    var_16 = var_8.end_position.line_no
    assert var_16 == 1
    var_17 = var_8.end_position.column_no
    assert var_17 == 3
    var_18 = var_8.end_position.char_index
    assert var_18 == 2


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'nested'
    var_5 = 456
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 25
    var_10 = '{"nested": {"inner": 456}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = error.messages()[var_8]
    var_13 = var_12.text
    assert var_13 == 'Must be a string.'
    var_14 = var_12.code
    assert var_14 == 'type'
    var_15 = var_12.index
    var_16 = bool(var_12.index == ['nested', 'inner'])
    assert var_16 is True
    var_17 = var_12.start_position.line_no
    assert var_17 == 1
    var_18 = var_12.start_position.column_no
    assert var_18 == 13
    var_19 = var_12.start_position.char_index
    assert var_19 == 12
    var_20 = var_12.end_position.line_no
    assert var_20 == 1
    var_21 = var_12.end_position.column_no
    assert var_21 == 25
    var_22 = var_12.end_position.char_index
    assert var_22 == 24


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
    var_11 = error.messages()[var_7]
    var_12 = var_11.text
    assert var_12 == 'Did not match any valid type.'
    var_13 = var_11.code
    assert var_13 == 'union'
    var_14 = var_11.index
    var_15 = bool(var_11.index == [])
    assert var_15 is True
    var_16 = var_11.start_position.line_no
    assert var_16 == 1
    var_17 = var_11.start_position.column_no
    assert var_17 == 1
    var_18 = var_11.start_position.char_index
    assert var_18 == 0
    var_19 = var_11.end_position.line_no
    assert var_19 == 1
    var_20 = var_11.end_position.column_no
    assert var_20 == 4
    var_21 = var_11.end_position.char_index
    assert var_21 == 3


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 4/33 statements.


def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_schema_field_validation_error. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/29 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 11/27 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 10/25 statements.



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 25
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{"age": 25}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'required'
    var_10 = var_8.index
    var_11 = bool(var_8.index == ['name'])
    assert var_11 is True
    var_12 = var_8.text
    assert var_12 == "The field 'name' is required."


def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 12
    var_6 = '{123: "value"}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'invalid_key'
    var_10 = var_8.index
    var_11 = bool(var_8.index == [123])
    assert var_11 is True


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'bad'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"name": "bad"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = error.messages()[var_6]
    var_11 = var_10.code
    assert var_11 == 'invalid'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True
    var_14 = var_10.text
    assert var_14 == 'Invalid value.'


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
    var_11 = error.messages()[var_7]
    var_12 = var_11.code
    assert var_12 == 'union'


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = 'bad'
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 30
    var_10 = '{"outer": {"inner": "bad"}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = error.messages()[var_8]
    var_13 = var_12.code
    assert var_13 == 'invalid'
    var_14 = var_12.index
    var_15 = bool(var_12.index == ['outer', 'inner'])
    assert var_15 is True
    var_16 = var_12.text
    assert var_16 == 'Invalid.'


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
    var_9 = 15
    var_10 = '{"a": 1, "b": 2}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/22 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 5/18 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 13/32 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/21 statements.


import typesystem.fields as module_0


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


def test_case_0():
    var_0 = 'not an integer'
    var_1 = 0
    var_2 = 15
    var_3 = ''
    var_4 = module_0.Field()


def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 40
    var_7 = ''
    var_8 = module_0.Field()
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = {var_0: var_11}
    var_13 = {}
    var_14 = module_1.Schema(var_12, **var_13)


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
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
    var_4 = module_0.Field()
    var_5 = module_0.Field()
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 7/21 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_schema_nested_validation_error. Retrieved 12/33 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/29 statements.
# Partially parsed test_validate_with_positions_successful_validation. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_sorted_messages. Retrieved 11/31 statements.


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)
    var_6 = error.messages()[var_2]
    var_7 = var_6.code
    assert var_7 == 'required'
    var_8 = var_6.index
    var_9 = bool(var_6.index == ['name'])
    assert var_9 is True
    var_10 = var_6.start_position.line_no
    assert var_10 == 1
    var_11 = var_6.start_position.column_no
    assert var_11 == 1
    var_12 = var_6.start_position.char_index
    assert var_12 == 0
    var_13 = var_6.end_position.line_no
    assert var_13 == 1
    var_14 = var_6.end_position.column_no
    assert var_14 == 2
    var_15 = var_6.end_position.char_index
    assert var_15 == 1


def test_case_0():
    var_0 = 'name'
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = "{1: 'value'}"
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = error.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'invalid_key'
    var_10 = var_8.index
    var_11 = bool(var_8.index == [1])
    assert var_11 is True
    var_12 = var_8.start_position.line_no
    assert var_12 == 1
    var_13 = var_8.start_position.column_no
    assert var_13 == 1
    var_14 = var_8.start_position.char_index
    assert var_14 == 0
    var_15 = var_8.end_position.line_no
    assert var_15 == 1
    var_16 = var_8.end_position.column_no
    assert var_16 == 11
    var_17 = var_8.end_position.char_index
    assert var_17 == 10


def test_case_0():
    var_0 = 'max_length'
    var_1 = 'Must have at most {max_length} characters.'
    var_2 = {var_0: var_1}
    var_3 = 'user'
    var_4 = 'name'
    var_5 = 5
    var_6 = 'longname'
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = 0
    var_10 = 25
    var_11 = "{'user': {'name': 'longname'}}"
    var_12 = module_0.Token(var_8, var_9, var_10, var_11)
    var_13 = error.messages()[var_9]
    var_14 = var_13.code
    assert var_14 == 'max_length'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['user', 'name'])
    assert var_16 is True
    var_17 = var_13.start_position.line_no
    assert var_17 == 1
    var_18 = var_13.start_position.column_no
    assert var_18 == 1
    var_19 = var_13.start_position.char_index
    assert var_19 == 0
    var_20 = var_13.end_position.line_no
    assert var_20 == 1
    var_21 = var_13.end_position.column_no
    assert var_21 == 26
    var_22 = var_13.end_position.char_index
    assert var_22 == 25


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
    var_11 = error.messages()[var_7]
    var_12 = var_11.code
    assert var_12 == 'union'
    var_13 = var_11.index
    var_14 = bool(var_11.index == [])
    assert var_14 is True
    var_15 = var_11.start_position.line_no
    assert var_15 == 1
    var_16 = var_11.start_position.column_no
    assert var_16 == 1
    var_17 = var_11.start_position.char_index
    assert var_17 == 0
    var_18 = var_11.end_position.line_no
    assert var_18 == 1
    var_19 = var_11.end_position.column_no
    assert var_19 == 4
    var_20 = var_11.end_position.char_index
    assert var_20 == 3


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = "{'name': 'test'}"
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'min_length'
    var_1 = 'Must have at least {min_length} characters.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 10
    var_6 = 'short'
    var_7 = 'also'
    var_8 = {var_3: var_6, var_4: var_7}
    var_9 = 0
    var_10 = 25
    var_11 = "{'a': 'short', 'b': 'also'}"
    var_12 = module_0.Token(var_8, var_9, var_10, var_11)



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/23 statements.
# Partially parsed test_validate_with_positions_invalid_key. Retrieved 8/21 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 8/25 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 8/21 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_null_not_allowed. Retrieved 6/19 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/24 statements.



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 25
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{"age": 25}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)


def test_case_0():
    var_0 = 'name'
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{1: "value"}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)


def test_case_0():
    var_0 = 'inner'
    var_1 = 'outer'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 15
    var_6 = '{"outer": {}}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 16
    var_8 = '{"field": "value"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = None
    var_3 = 0
    var_4 = 4
    var_5 = 'null'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = False
    var_2 = None
    var_3 = 4
    var_4 = 'null'
    var_5 = module_0.Token(var_2, var_1, var_3, var_4)


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 12
    var_10 = '{"a":1,"b":2}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)

def test_case_0():
    pass



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/26 statements.
# Partially parsed test_validate_with_positions_nested_error. Retrieved 9/29 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 11/32 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 7/33 statements.



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 25
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = '{"age": 25}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)


def test_case_0():
    var_0 = 'count'
    var_1 = 'inner'
    var_2 = 'invalid'
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}
    var_5 = 0
    var_6 = 30
    var_7 = '{"inner": {"count": "invalid"}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'error_a'
    var_3 = 'error_b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 0
    var_8 = 20
    var_9 = '{"a": 1, "b": 2}'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = 'error_a'
    var_12 = 'error_b'


def test_case_0():
    var_0 = 'data'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"data": null}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 12/29 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 13/30 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 8/31 statements.
# Partially parsed test_validate_with_positions_successful_validation. Retrieved 9/22 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 13/33 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 8/22 statements.


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = '{"existing": "value"}'
    var_6 = 'missing'
    var_7 = module_0.Field()
    var_8 = module_0.Field()
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = len(e.messages())
    assert var_13 == 1
    var_14 = e.messages()[0]
    var_15 = var_14.text
    assert var_15 == "The field 'missing' is required."
    var_16 = var_14.code
    assert var_16 == 'required'
    var_17 = var_14.index
    var_18 = bool(var_14.index == ['missing'])
    assert var_18 is True
    var_19 = var_14.start_position.line_no
    assert var_19 == 1
    var_20 = var_14.start_position.column_no
    assert var_20 == 1
    var_21 = var_14.start_position.char_index
    assert var_21 == 0
    var_22 = var_14.end_position.line_no
    assert var_22 == 1
    var_23 = var_14.end_position.column_no
    assert var_23 == 21
    var_24 = var_14.end_position.char_index
    assert var_24 == 20


def test_case_0():
    var_0 = 'outer'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"outer": {}}'
    var_6 = 'inner'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    assert var_15 == 1
    var_16 = e.messages()[0]
    var_17 = var_16.text
    assert var_17 == "The field 'inner' is required."
    var_18 = var_16.code
    assert var_18 == 'required'
    var_19 = var_16.index
    var_20 = bool(var_16.index == ['outer', 'inner'])
    assert var_20 is True
    var_21 = var_16.start_position.line_no
    assert var_21 == 1
    var_22 = var_16.start_position.column_no
    assert var_22 == 12
    var_23 = var_16.start_position.char_index
    assert var_23 == 11
    var_24 = var_16.end_position.line_no
    assert var_24 == 1
    var_25 = var_16.end_position.column_no
    assert var_25 == 12
    var_26 = var_16.end_position.char_index
    assert var_26 == 11

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
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0]
    var_12 = var_11.text
    assert var_12 == 'Custom error.'
    var_13 = var_11.code
    assert var_13 == 'custom'
    var_14 = var_11.index
    var_15 = bool(var_11.index == ['field'])
    assert var_15 is True
    var_16 = var_11.start_position.line_no
    assert var_16 == 1
    var_17 = var_11.start_position.column_no
    assert var_17 == 11
    var_18 = var_11.start_position.char_index
    assert var_18 == 10
    var_19 = var_11.end_position.line_no
    assert var_19 == 1
    var_20 = var_11.end_position.column_no
    assert var_20 == 18
    var_21 = var_11.end_position.char_index
    assert var_21 == 17


def test_case_0():
    var_0 = 'field'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 18
    var_5 = '{"field": "valid"}'
    var_6 = module_0.Field()
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)


def test_case_0():
    var_0 = 'b'
    var_1 = 'a'
    var_2 = 'value'
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 0
    var_5 = 30
    var_6 = '{"a": "value", "b": "value"}'
    var_7 = 'c'
    var_8 = module_0.Field()
    var_9 = module_0.Field()
    var_10 = module_0.Field()
    var_11 = {var_1: var_8, var_0: var_9, var_7: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = bool(False)
    assert var_14 is True


def test_case_0():
    var_0 = 'invalid'
    var_1 = 0
    var_2 = 7
    var_3 = '"invalid"'
    var_4 = module_0.Field()
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.text
    assert var_11 == 'Did not match any valid type.'
    var_12 = var_10.code
    assert var_12 == 'union'
    var_13 = var_10.index
    var_14 = bool(var_10.index == [])
    assert var_14 is True
    var_15 = var_10.start_position.line_no
    assert var_15 == 1
    var_16 = var_10.start_position.column_no
    assert var_16 == 1
    var_17 = var_10.start_position.char_index
    assert var_17 == 0
    var_18 = var_10.end_position.line_no
    assert var_18 == 1
    var_19 = var_10.end_position.column_no
    assert var_19 == 8
    var_20 = var_10.end_position.char_index
    assert var_20 == 7



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/27 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 8/26 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 9/29 statements.
# Partially parsed test_validate_with_positions_nested_invalid_type. Retrieved 10/30 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 11/31 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 9/34 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 2/14 statements.


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'city'
    var_4 = 'address'
    var_5 = 456
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 24
    var_10 = '{"address": {"city": 456}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)


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
    var_12 = 1


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = 14
    var_11 = '{"data": null}'
    var_12 = module_0.Token(var_8, var_9, var_10, var_11)

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_positions_schema_required_field. Retrieved 7/19 statements.
# Partially parsed test_validate_with_positions_schema_invalid_key. Retrieved 9/21 statements.
# Partially parsed test_validate_with_positions_schema_nested_error. Retrieved 11/25 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/27 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/19 statements.



def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0]
    var_8 = var_7.code
    assert var_8 == 'required'
    var_9 = var_7.index
    var_10 = bool(var_7.index == ['name'])
    assert var_10 is True
    var_11 = var_7.start_position.line_no
    assert var_11 == 1
    var_12 = var_7.start_position.column_no
    assert var_12 == 1
    var_13 = var_7.start_position.char_index
    assert var_13 == 0
    var_14 = var_7.end_position.line_no
    assert var_14 == 1
    var_15 = var_7.end_position.column_no
    assert var_15 == 2
    var_16 = var_7.end_position.char_index
    assert var_16 == 1


def test_case_0():
    var_0 = 'name'
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 10
    var_6 = "{1: 'value'}"
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'invalid_key'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [1])
    assert var_12 is True
    var_13 = var_9.start_position.line_no
    assert var_13 == 1
    var_14 = var_9.start_position.column_no
    assert var_14 == 1
    var_15 = var_9.start_position.char_index
    assert var_15 == 0
    var_16 = var_9.end_position.line_no
    assert var_16 == 1
    var_17 = var_9.end_position.column_no
    assert var_17 == 11
    var_18 = var_9.end_position.char_index
    assert var_18 == 10


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = 'bad'
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 25
    var_10 = '{"outer": {"inner": "bad"}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0]
    var_14 = var_13.code
    assert var_14 == 'custom'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['outer', 'inner'])
    assert var_16 is True
    var_17 = var_13.start_position.line_no
    assert var_17 == 1
    var_18 = var_13.start_position.column_no
    assert var_18 == 1
    var_19 = var_13.start_position.char_index
    assert var_19 == 0
    var_20 = var_13.end_position.line_no
    assert var_20 == 1
    var_21 = var_13.end_position.column_no
    assert var_21 == 26
    var_22 = var_13.end_position.char_index
    assert var_22 == 25


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 123
    var_4 = 5
    var_5 = 7
    var_6 = '  "123"'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'type'
    var_11 = var_9.index
    var_12 = bool(var_9.index == [])
    assert var_12 is True
    var_13 = var_9.start_position.line_no
    assert var_13 == 1
    var_14 = var_9.start_position.column_no
    assert var_14 == 3
    var_15 = var_9.start_position.char_index
    assert var_15 == 2
    var_16 = var_9.end_position.line_no
    assert var_16 == 1
    var_17 = var_9.end_position.column_no
    assert var_17 == 8
    var_18 = var_9.end_position.char_index
    assert var_18 == 7


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
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [])
    assert var_15 is True
    var_16 = var_12.start_position.line_no
    assert var_16 == 1
    var_17 = var_12.start_position.column_no
    assert var_17 == 1
    var_18 = var_12.start_position.char_index
    assert var_18 == 0
    var_19 = var_12.end_position.line_no
    assert var_19 == 1
    var_20 = var_12.end_position.column_no
    assert var_20 == 4
    var_21 = var_12.end_position.char_index
    assert var_21 == 3


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"name": "test"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/22 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_custom_error. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_multiple_errors. Retrieved 12/37 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 7/18 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/29 statements.
# Partially parsed test_validate_with_positions_sorted_by_position. Retrieved 8/23 statements.



def test_case_0():
    var_0 = 'field'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 10
    var_5 = '{"field": null}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)
    var_7 = error.messages()[var_3]
    var_8 = var_7.text
    assert var_8 == "The field 'field' is required."
    var_9 = var_7.code
    assert var_9 == 'required'
    var_10 = var_7.index
    var_11 = bool(var_7.index == ['field'])
    assert var_11 is True
    var_12 = var_7.start_position
    var_13 = bool(var_7.start_position is not None)
    assert var_13 is True
    var_14 = var_7.end_position
    var_15 = bool(var_7.end_position is not None)
    assert var_15 is True


def test_case_0():
    var_0 = 'inner'
    var_1 = 'outer'
    var_2 = None
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = '{"outer": {"inner": null}}'
    var_8 = module_0.Token(var_4, var_5, var_6, var_7)
    var_9 = error.messages()[var_5]
    var_10 = var_9.text
    assert var_10 == "The field 'inner' is required."
    var_11 = var_9.code
    assert var_11 == 'required'
    var_12 = var_9.index
    var_13 = bool(var_9.index == ['outer', 'inner'])
    assert var_13 is True
    var_14 = var_9.start_position
    var_15 = bool(var_9.start_position is not None)
    assert var_15 is True
    var_16 = var_9.end_position
    var_17 = bool(var_9.end_position is not None)
    assert var_17 is True


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = '{"field": "invalid"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = error.messages()[var_6]
    var_11 = var_10.text
    assert var_11 == 'Custom error'
    var_12 = var_10.code
    assert var_12 == 'custom'
    var_13 = var_10.index
    var_14 = bool(var_10.index == ['field'])
    assert var_14 is True
    var_15 = var_10.start_position
    var_16 = bool(var_10.start_position is not None)
    assert var_16 is True
    var_17 = var_10.end_position
    var_18 = bool(var_10.end_position is not None)
    assert var_18 is True


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'Invalid value'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = None
    var_6 = 'bad'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 25
    var_10 = '{"field1": null, "field2": "bad"}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = 'required'
    var_13 = 'invalid'


def test_case_0():
    var_0 = 'field'
    var_1 = 'valid'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 15
    var_5 = '{"field": "valid"}'
    var_6 = module_0.Token(var_2, var_3, var_4, var_5)


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be integer'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be string'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 0
    var_8 = 4
    var_9 = 'true'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = error.messages()[var_7]
    var_12 = var_11.code
    assert var_12 == 'union'
    var_13 = var_11.start_position
    var_14 = bool(var_11.start_position is not None)
    assert var_14 is True
    var_15 = var_11.end_position
    var_16 = bool(var_11.end_position is not None)
    assert var_16 is True


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = {var_1: var_2, var_0: var_2}
    var_4 = 0
    var_5 = 30
    var_6 = '{"field2": null, "field1": null}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 6/30 statements.
# Partially parsed test_validate_with_positions_nested_required_field. Retrieved 11/42 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 9/38 statements.
# Partially parsed test_validate_with_positions_successful_validation. Retrieved 7/28 statements.
# Partially parsed test_validate_with_positions_messages_sorted_by_position. Retrieved 11/42 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 2/21 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = {}
    var_5 = 0
    var_6 = 10
    var_7 = '{}'

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'inner_name'
    var_4 = 'outer'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = 20
    var_9 = '{"outer": {}}'
    var_10 = {}
    var_11 = 9
    var_12 = 11

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 20
    var_8 = '{"name": 123}'
    var_9 = 10
    var_10 = 12

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'valid'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 20
    var_8 = '{"name": "valid"}'

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
    var_9 = 30
    var_10 = '{"a": 1, "b": 2}'
    var_11 = 10
    var_12 = 20

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 8/26 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 7/35 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 12/32 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 9/41 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 6/37 statements.


import typesystem.fields as module_0


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 10
    var_3 = ''
    var_4 = module_0.Field()
    var_5 = 'field1'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)

def test_case_0():
    var_0 = 'field1'
    var_1 = 'not an int'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 20
    var_5 = ''
    var_6 = 'type'
    var_7 = 'Must be an integer.'
    var_8 = {var_6: var_7}


def test_case_0():
    var_0 = 'outer'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = 12
    var_5 = ''
    var_6 = 'inner'
    var_7 = module_0.Field()
    var_8 = {var_6: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)

def test_case_0():
    var_0 = 'field2'
    var_1 = 'field1'
    var_2 = 'invalid'
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 0
    var_5 = 20
    var_6 = ''
    var_7 = 'type'
    var_8 = 'Must be an integer.'
    var_9 = {var_7: var_8}
    var_10 = 1

def test_case_0():
    var_0 = 'not an int or string'
    var_1 = 0
    var_2 = 20
    var_3 = ''
    var_4 = 'type'
    var_5 = 'Must be an integer.'
    var_6 = {var_4: var_5}
    var_7 = 'type'
    var_8 = 'Must be a string.'
    var_9 = {var_7: var_8}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_positions_success. Retrieved 4/13 statements.
# Partially parsed test_validate_with_positions_field_validation_error. Retrieved 5/16 statements.
# Partially parsed test_validate_with_positions_schema_required_error. Retrieved 6/20 statements.
# Partially parsed test_validate_with_positions_schema_nested_required_error. Retrieved 8/24 statements.
# Partially parsed test_validate_with_positions_schema_multiple_errors_sorted. Retrieved 10/25 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 7/28 statements.


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'valid'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Invalid value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = 0
    var_5 = 6
    var_6 = module_0.Token(var_3, var_4, var_5, var_3)
    var_7 = bool(False)
    assert var_7 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 0
    var_3 = 1
    var_4 = '{}'
    var_5 = module_0.Token(var_1, var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = 'inner'
    var_1 = 'outer'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = 12
    var_6 = '{"outer": {}}'
    var_7 = module_0.Token(var_3, var_4, var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Field error'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 0
    var_9 = 11
    var_10 = '{"a":1,"b":2}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = bool(False)
    assert var_12 is True


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be string'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be integer'
    var_5 = {var_3: var_4}
    var_6 = 3.14
    var_7 = 0
    var_8 = 3
    var_9 = '3.14'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'required'
    var_1 = 'This field is required.'
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = {}
    var_5 = 0
    var_6 = ''



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_positions_required_field. Retrieved 9/23 statements.
# Partially parsed test_validate_with_positions_invalid_type. Retrieved 8/22 statements.
# Partially parsed test_validate_with_positions_nested_required. Retrieved 9/25 statements.
# Partially parsed test_validate_with_positions_nested_invalid_type. Retrieved 10/26 statements.
# Partially parsed test_validate_with_positions_multiple_errors_sorted. Retrieved 10/25 statements.
# Partially parsed test_validate_with_positions_union_field_error. Retrieved 7/26 statements.
# Partially parsed test_validate_with_positions_success. Retrieved 8/21 statements.
# Partially parsed test_validate_with_positions_null_allowed. Retrieved 8/21 statements.



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


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'city'
    var_4 = 'address'
    var_5 = 456
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 24
    var_10 = '{"address": {"city": 456}}'
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be an integer.'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 0
    var_8 = 4
    var_9 = 'true'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'John'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 14
    var_8 = '{"name": "John"}'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)


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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_positions_required_field_error. Retrieved 4/26 statements.


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 30
    var_3 = ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_positions_schema_field_validation_error. Retrieved 9/20 statements.
# Partially parsed test_validate_with_positions_union_error. Retrieved 8/26 statements.
# Partially parsed test_validate_with_positions_schema_nested_error. Retrieved 11/24 statements.
# Partially parsed test_validate_with_positions_messages_sorted_by_position. Retrieved 10/23 statements.


import typesystem.fields as module_0
import typesystem.tokenize.positional_validation as module_3
import typesystem.tokenize.tokens as module_2


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = 0
    var_7 = 1
    var_8 = '{}'
    var_9 = module_2.Token(var_5, var_6, var_7, var_8)
    var_10 = module_3.validate_with_positions(token=var_9, validator=var_4)
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'required'
    var_14 = var_12.index
    var_15 = bool(var_12.index == ['name'])
    assert var_15 is True
    var_16 = var_12.start_position.char_index
    assert var_16 == 0
    var_17 = var_12.end_position.char_index
    assert var_17 == 1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 1
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 0
    var_9 = 10
    var_10 = "{1: 'value'}"
    var_11 = module_2.Token(var_7, var_8, var_9, var_10)
    var_12 = module_3.validate_with_positions(token=var_11, validator=var_4)
    var_13 = len(e.messages())
    assert var_13 == 1
    var_14 = e.messages()[0]
    var_15 = var_14.code
    assert var_15 == 'invalid_key'
    var_16 = var_14.index
    var_17 = bool(var_14.index == [1])
    assert var_17 is True
    var_18 = var_14.start_position.char_index
    assert var_18 == 1
    var_19 = var_14.end_position.char_index
    assert var_19 == 1

import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Custom error.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'invalid'
    var_5 = {var_3: var_4}
    var_6 = 0
    var_7 = 15
    var_8 = "{'name': 'invalid'}"
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0]
    var_12 = var_11.code
    assert var_12 == 'custom'
    var_13 = var_11.index
    var_14 = bool(var_11.index == ['name'])
    assert var_14 is True
    var_15 = var_11.start_position.char_index
    assert var_15 == 9
    var_16 = var_11.end_position.char_index
    assert var_16 == 14


def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be integer.'
    var_5 = {var_3: var_4}
    var_6 = 3.14
    var_7 = 0
    var_8 = 3
    var_9 = '3.14'
    var_10 = module_0.Token(var_6, var_7, var_8, var_9)
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0]
    var_13 = var_12.code
    assert var_13 == 'union'
    var_14 = var_12.index
    var_15 = bool(var_12.index == [])
    assert var_15 is True
    var_16 = var_12.start_position.char_index
    assert var_16 == 0
    var_17 = var_12.end_position.char_index
    assert var_17 == 3


def test_case_0():
    var_0 = 'custom'
    var_1 = 'Nested error.'
    var_2 = {var_0: var_1}
    var_3 = 'inner'
    var_4 = 'outer'
    var_5 = 'value'
    var_6 = {var_3: var_5}
    var_7 = {var_4: var_6}
    var_8 = 0
    var_9 = 30
    var_10 = "{'outer': {'inner': 'value'}}"
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0]
    var_14 = var_13.code
    assert var_14 == 'custom'
    var_15 = var_13.index
    var_16 = bool(var_13.index == ['outer', 'inner'])
    assert var_16 is True
    var_17 = var_13.start_position.char_index
    assert var_17 == 19
    var_18 = var_13.end_position.char_index
    assert var_18 == 24

import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'John'
    var_6 = {var_0: var_5}
    var_7 = 0
    var_8 = 15
    var_9 = "{'name': 'John'}"
    var_10 = module_2.Token(var_6, var_7, var_8, var_9)
    var_11 = module_3.validate_with_positions(token=var_10, validator=var_4)
    var_12 = bool(var_11 == {'name': 'John'})
    assert var_12 is True


def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'allow_null'
    var_5 = {var_4: var_1}
    var_6 = module_1.Schema(var_3, **var_5)
    var_7 = None
    var_8 = 0
    var_9 = 3
    var_10 = 'null'
    var_11 = module_2.Token(var_7, var_8, var_9, var_10)
    var_12 = module_3.validate_with_positions(token=var_11, validator=var_6)
    assert var_12 is None


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'not an object'
    var_6 = 0
    var_7 = 12
    var_8 = module_2.Token(var_5, var_6, var_7, var_5)
    var_9 = module_3.validate_with_positions(token=var_8, validator=var_4)
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0]
    var_12 = var_11.code
    assert var_12 == 'type'
    var_13 = var_11.index
    var_14 = bool(var_11.index == [])
    assert var_14 is True
    var_15 = var_11.start_position.char_index
    assert var_15 == 0
    var_16 = var_11.end_position.char_index
    assert var_16 == 12

import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'error'
    var_1 = 'Error.'
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'val1'
    var_6 = 'val2'
    var_7 = {var_4: var_5, var_3: var_6}
    var_8 = 0
    var_9 = 20
    var_10 = "{'b': 'val1', 'a': 'val2'}"
    var_11 = module_0.Token(var_7, var_8, var_9, var_10)



