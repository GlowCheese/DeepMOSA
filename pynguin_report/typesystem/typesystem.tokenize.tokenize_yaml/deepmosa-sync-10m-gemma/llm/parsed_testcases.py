####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 13/19 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = bool(var_1.string == "12<'123'>" or var_1.string == '123')
    assert var_3 is True

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

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: python'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'name': 'python'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'data:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'data'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = 'item1'
    var_6 = 'item_2'
    var_7 = '_'
    var_8 = ''
    var_9 = var_4.value
    var_10 = module_0.tokenize_yaml(var_0)
    var_11 = [var_2]
    var_12 = var_10.lookup(var_11)
    var_13 = var_12.value
    var_14 = bool(var_13 == ['item1', 'item2'])
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 6/7 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 6/7 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_dict. Retrieved 5/7 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 3/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'count: 42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['count']
    assert var_2 == 42
    var_3 = 'count'
    var_4 = [var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 42

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "name: 'John Doe'"
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'name'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'John Doe'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- apple\n- banana'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value[0]
    assert var_2 == 'apple'
    var_3 = var_1.value[1]
    assert var_3 == 'banana'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'user:\n  id: 1\n  active: true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'user'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value['id']
    assert var_5 == 1
    var_6 = var_4.value['active']
    assert var_6 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'pi: 3.14\nis_valid: false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'pi'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    var_6 = bool(var_5 == 3.14)
    assert var_6 is True
    var_7 = 'is_valid'
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 is False

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

def test_case_0():
    var_0 = 'data: null'
    var_1 = 'data'
    var_2 = [var_1]



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/12 statements.
# Partially parsed test_validate_yaml_validation_error_with_positions. Retrieved 3/16 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'name: : invalid'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)

def test_case_0():
    var_0 = 'name'
    var_1 = 'age: 30'
    var_2 = 'required'

def test_case_0():
    var_0 = b'name: John'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/6 statements.
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.value[0].value
    assert var_4 == 1

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
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none_fails. Retrieved 3/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = str(var_0)
    var_3 = "'pyyaml' must be installed."
    var_4 = bool("'pyyaml' must be installed." in var_2)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 7/9 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = bool(var_1.string == "12<'123'>" or var_1.string == '123')
    assert var_3 is True

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

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: 10'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['outer']
    var_3 = bool(var_1.value['outer'] == {'inner': 10})
    assert var_3 is True
    var_4 = 'outer'
    var_5 = 'inner'
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 10

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['name']
    assert var_2 == 'test'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_with_valid_content. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'value'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 7/12 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 9/19 statements.
# Partially parsed test_validate_yaml_required_error. Retrieved 9/19 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 5/11 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 5/11 statements.
# Partially parsed test_validate_yaml_null_value. Retrieved 7/17 statements.
# Partially parsed test_validate_yaml_allow_null_success. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name: John\nage: 30'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: not_an_integer'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = 'Should have raised ValidationError'
    var_9 = AssertionError(var_8)
    var_10 = 'type'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: 30'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = 'Should have raised ValidationError'
    var_9 = AssertionError(var_8)
    var_10 = 'required'

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name John'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'Should have raised ParseError'
    var_5 = AssertionError(var_4)
    var_6 = bool(var_2)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = '   '
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'Should have raised ParseError'
    var_5 = AssertionError(var_4)
    var_6 = bool(var_2)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'null'
    var_5 = module_1.validate_yaml(var_4, var_3)
    var_6 = 'Should have raised ValidationError'
    var_7 = AssertionError(var_6)
    var_8 = bool(var_6)
    assert var_8 is True
    var_9 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'null'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/11 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/7 statements.
# Partially parsed test_validate_yaml_validation_error_positions. Retrieved 2/10 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_null_handling. Retrieved 2/7 statements.
# Partially parsed test_validate_yaml_required_field_missing. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name: : invalid'

def test_case_0():
    var_0 = 'age'
    var_1 = 'age: not_an_int'

def test_case_0():
    var_0 = '   '

def test_case_0():
    var_0 = b'Hello World'

def test_case_0():
    var_0 = True
    var_1 = 'null'

def test_case_0():
    var_0 = 'age'
    var_1 = 'name: John'



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: [unclosed bracket'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 3/4 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/7 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 2/3 statements.


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
    var_1 = '123'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = var_2.value
    assert var_3 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

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

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['item1', 'item2'])
    assert var_3 is True
    var_4 = 0
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 'item1'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'parent:\n  child: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'child'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'foo: bar'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'foo': 'bar'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_yaml_asserts_yaml_not_none. Retrieved 10/17 statements.


import typesystem.base as module_0
import typesystem.fields as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_1, var_1, var_2)
    var_4 = 5
    var_5 = 4
    var_6 = module_0.Position(var_1, var_4, var_5)
    var_7 = {}
    var_8 = module_1.String(**var_7)
    var_9 = 'test'
    var_10 = module_2.validate_yaml(var_9, var_8)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_yaml_scanner_error_with_none_problem. Retrieved 2/15 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 4/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = str(var_0)
    var_3 = "'pyyaml' must be installed."
    var_4 = bool("'pyyaml' must be installed." in var_2)
    assert var_4 is True
    var_5 = 'yaml'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 4/5 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/7 statements.
# Partially parsed test_tokenize_yaml_empty_string_raises_error. Retrieved 4/9 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 2/3 statements.


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
    var_4 = var_1.value
    var_5 = len(var_4)
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

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'foo: bar'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'foo': 'bar'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)
    var_4 = bool(var_1)
    assert var_4 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\nlist:\n  - item1\n  - item2\ndict:\n  inner: true\n'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['list'][0]
    assert var_2 == 'item1'
    var_3 = var_1.value['dict']['inner']
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 7/8 statements.


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
    var_4 = var_1.string
    assert var_4 == '- 1\n- 2'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['outer']['inner']
    var_3 = bool(var_1.value['outer']['inner'] == [1, 2])
    assert var_3 is True
    var_4 = 'outer'
    var_5 = 'inner'
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    var_9 = bool(var_8 == [1, 2])
    assert var_9 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 5/21 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = None
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = str(var_2)
    var_5 = "'pyyaml' must be installed."
    var_6 = bool("'pyyaml' must be installed." in var_4)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_no_problem_attribute. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/8 statements.
# Partially parsed test_validate_yaml_validation_error_required. Retrieved 2/9 statements.
# Partially parsed test_validate_yaml_validation_error_type. Retrieved 2/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name: John\nage: 30'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'name: : John'
    var_1 = {}
    var_2 = module_0.validate_yaml(var_0, var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'age: 30'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = "age: 'not_an_int'"

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.validate_yaml(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_yaml_assert_yaml_not_none_evaluates_to_false. Retrieved 12/22 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = module_0.Position(var_1, var_1, var_2)
    var_4 = 5
    var_5 = 4
    var_6 = module_0.Position(var_1, var_4, var_5)
    var_7 = 'test'
    var_8 = []
    var_9 = 'test'
    var_10 = 'The assertion at line 14 did not trigger.'
    var_11 = AssertionError(var_10)



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n\n'
    var_1 = ':'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_fails_when_yaml_is_none. Retrieved 5/15 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = str(var_1)
    var_4 = "'pyyaml' must be installed."
    var_5 = bool("'pyyaml' must be installed." in var_3)
    assert var_5 is True
    var_6 = 'yaml'



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = '{}'
    var_4 = module_0.Field()
    var_5 = module_1.validate_yaml(var_3, var_4)
    var_6 = "'pyyaml' must be installed."

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = 0
    var_3 = 3
    var_4 = module_1.ScalarToken(var_1, var_2, var_3, var_1)
    var_5 = 'test'
    var_6 = module_2.validate_yaml(var_5, var_0)
    var_7 = 'AssertionError was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_exists. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_yaml_valid_content_avoids_exception_block. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'value'



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '  : :'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none_fails. Retrieved 4/15 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = 'yaml'



# Parsed testcases at query #18
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_exists. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 5/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/9 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '[1, 2]'
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
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    a: [1, 2]\n    b: {c: 3}\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value['a']
    var_3 = bool(var_1.value['a'] == [1, 2])
    assert var_3 is True
    var_4 = var_1.value['b']
    var_5 = bool(var_1.value['b'] == {'c': 3})
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: : syntax'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_yaml_pyyaml_installed. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'yaml'
    var_1 = 'content'
    var_2 = 'yaml'



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'content'
    var_3 = module_0.tokenize_yaml(var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_tokenize_yaml_yaml_is_not_none. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_yaml_assert_yaml_not_none. Retrieved 6/18 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'yaml'
    var_2 = 'yaml'
    var_3 = 'content'
    var_4 = None
    var_5 = module_0.validate_yaml(var_3, var_4)



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)



