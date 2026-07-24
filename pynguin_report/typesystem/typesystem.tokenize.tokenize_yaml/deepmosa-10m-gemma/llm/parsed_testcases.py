####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 6/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 8/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['item1', 'item2'])
    assert var_3 is True
    var_4 = 0
    var_5 = var_1.lookup(var_4)
    var_6 = var_1.lookup(var_4)
    var_7 = var_6.value
    assert var_7 == 'item1'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'data'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    user:\n      name: Alice\n      roles:\n        - admin\n        - editor\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['user']['name']
    assert var_2 == 'Alice'
    var_3 = var_1.value['user']['roles']
    var_4 = bool(var_1.value['user']['roles'] == ['admin', 'editor'])
    assert var_4 is True
    var_5 = 'user'
    var_6 = 'roles'
    var_7 = 0
    var_8 = [var_5, var_6, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 'admin'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_yaml_is_not_none. Retrieved 3/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_yaml_is_not_none. Retrieved 8/24 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = None
    var_2 = 'resolver'
    var_3 = 'BaseResolver'
    var_4 = 'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = str(var_4)
    var_7 = "'pyyaml' must be installed."
    var_8 = bool("'pyyaml' must be installed." in var_6)
    assert var_8 is True
    var_9 = 'yaml'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_empty_error. Retrieved 4/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'data'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)
    var_4 = bool(var_0)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_required_error. Retrieved 1/11 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/7 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'age: not_an_int'

def test_case_0():
    var_0 = 'age: 30'

def test_case_0():
    var_0 = 'name: "John'

def test_case_0():
    var_0 = '   '



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 7/10 statements.
# Partially parsed test_tokenize_yaml_empty_content_raises_error. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value[0].value
    assert var_2 == 'item1'
    var_3 = var_1.value[1].value
    assert var_3 == 'item2'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = [var_2]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_0)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'data'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = var_2.line_no
    assert var_3 == 2
    var_4 = var_2.column_no
    assert var_4 == 1
    var_5 = var_2.char_index
    assert var_5 == 6



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1
import builtins as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'content'
    var_2 = module_1.validate_yaml(var_1, var_0)
    var_3 = 'AssertionError was not raised'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_2.Exception(*var_4, **var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 'a'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'data'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/11 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'not_an_int'

def test_case_0():
    var_0 = 'key: : value'

def test_case_0():
    var_0 = '   '



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/11 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_validation_error_required. Retrieved 2/9 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_type_mismatch. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name: : : invalid'
    var_1 = 'parse_error'

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'other_field: value'
    var_2 = "'required_field'"

def test_case_0():
    var_0 = '   '
    var_1 = 'No content'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 'type'
    var_2 = bool('type' in str(e).lower())
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_yaml_is_not_none. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '4    2'
    var_4 = var_1.string
    assert var_4 == '42'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['true']
    assert var_2 is None
    var_3 = 'true'
    var_4 = [var_3]
    var_5 = var_1.lookup_key(var_4)
    var_6 = var_5.value
    assert var_6 is None

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = var_2.line_no
    assert var_3 == 2
    var_4 = var_2.column_no
    assert var_4 == 1
    var_5 = var_2.char_index
    assert var_5 == 6



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 13/17 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1[0].value
    assert var_4 == 1
    var_5 = var_1[1].value
    assert var_5 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'
    var_8 = [var_4]
    var_9 = var_1.lookup_key(var_8)
    var_10 = var_9.value
    assert var_10 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = str(e).split('.')[0]
    assert var_2 == 'No content'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    foo:\n      - bar\n      - 123\n    baz: true\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['foo']
    var_3 = bool(var_1.value['foo'] is not None)
    assert var_3 is True
    var_4 = 'foo'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = 0
    var_8 = [var_4]
    var_9 = var_1.lookup(var_8)[var_7]
    var_10 = var_9.value
    assert var_10 == 'bar'
    var_11 = 1
    var_12 = [var_4]
    var_13 = var_1.lookup(var_12)[var_11]
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = var_1.value['baz']
    assert var_15 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 5/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '12.34'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 12.34)
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'key'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'name': 'test'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/9 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/9 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name: John\nage: : 30'
    var_1 = 'ParseError'

def test_case_0():
    var_0 = 'age: not_an_int'



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: : yaml'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_does_not_trigger_problem_is_none_assertion. Retrieved 3/14 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'valid: yaml'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_exists. Retrieved 4/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = None
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 11/16 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup_key(var_7)
    var_9 = var_8.value
    assert var_9 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    foo:\n      - bar\n      - baz\n    num: 42\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['foo']
    var_3 = bool(var_1.value['foo'] == ['bar', 'baz'])
    assert var_3 is True
    var_4 = var_1.value['num']
    assert var_4 == 42
    var_5 = 'foo'
    var_6 = [var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = 0
    var_9 = [var_5, var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = [var_5, var_8]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 'bar'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value[0].value
    assert var_2 == 'item1'
    var_3 = var_1.value[1].value
    assert var_3 == 'item2'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['key']
    assert var_2 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'data'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/9 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_validation_error_positions. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name: : invalid'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'age: not_an_int'

def test_case_0():
    var_0 = ''
    var_1 = bool(False)
    assert var_1 is True
    var_2 = ''



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'content'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'Assertion was not triggered when yaml was None'
    var_5 = AssertionError(var_4)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'content'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'Expected AssertionError was not raised'
    var_5 = AssertionError(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 3/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 7/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "'hello world'"
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello world'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_4]
    var_8 = var_1.lookup_key(var_7)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\t'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/11 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 5/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == 'line1\nline2'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_yaml_assert_yaml_not_none. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'content'
    var_1 = 'The assertion at line 14 did not trigger when yaml is None.'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



