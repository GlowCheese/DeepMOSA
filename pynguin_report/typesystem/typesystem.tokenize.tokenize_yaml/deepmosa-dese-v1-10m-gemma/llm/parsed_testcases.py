####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: tester'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    user:\n      id: 1\n      tags:\n        - admin\n        - dev\n    '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'some content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_tokenize_yaml_assert_yaml_not_none.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 2/13 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 2/8 statements.
# Partially parsed test_validate_yaml_validation_failure. Retrieved 2/16 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'age: 25'
    var_1 = 'age'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age 25'
    var_1 = module_0.Field()

def test_case_0():
    var_0 = "age: 'twenty-five'"
    var_1 = 'age'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.Field()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_yaml_assert_pyyaml_not_none. Retrieved 6/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = 'test'
    var_5 = 'Failed to trigger AssertionError for missing yaml'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 10/12 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/14 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup(var_5)
    var_7 = [var_2]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1
    var_3 = var_2._value
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = [var_5]
    var_7 = 1
    var_8 = [var_7]

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'integer: 123\nboolean: true\nfloat: 45.6'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'integer'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 123
    var_6 = 'boolean'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 is True
    var_10 = 'float'
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_no_problem_attribute. Retrieved 3/18 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = None
    var_2 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': :'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/11 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 1/12 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/8 statements.


def test_case_0():
    var_0 = '42'

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

def test_case_0():
    var_0 = "'not an int'"

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '   '
    var_2 = module_1.validate_yaml(var_1, var_0)

def test_case_0():
    var_0 = b'123'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_yaml_asserts_yaml_is_not_none. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_str. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 3/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 7/10 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 8/11 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup_key(var_5)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_yaml_assert_yaml_not_none. Retrieved 4/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)
    var_3 = str(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_is_not_none. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'yaml'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/9 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 4/14 statements.
# Partially parsed test_validate_yaml_required_error. Retrieved 2/13 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/7 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'name: Alice\nage: 30'

def test_case_0():
    var_0 = 'age: not_an_int'
    var_1 = 0
    var_2 = error.messages()[var_1]
    var_3 = var_2.code
    assert var_3 == 'type'

def test_case_0():
    var_0 = 'age: 30'
    var_1 = 'required'

def test_case_0():
    var_0 = 'name:\tAlice'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 8/13 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    a:\n      - 1\n      - 2\n    b: true\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = 0
    var_6 = [var_2, var_5]
    var_7 = var_1.lookup(var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "'hello world'"
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: 10'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'outer'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = 'inner'
    var_6 = [var_5]
    var_7 = var_4.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 10

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 3/6 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 6/7 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'list:\n  - item1\n  - item2\ndict:\n  a: 1'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: test'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'foo: bar'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'some content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_tokenize_yaml_scanner_error_with_no_problem.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: : yaml'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 7/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup_key(var_5)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = var_1.lookup(var_2)
    var_4 = var_3.value
    assert var_4 == 'item1'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'i: 1\nf: 1.5\nb: true\nn: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'i'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 1
    var_6 = 'f'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    var_10 = 'b'
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 is True
    var_14 = 'n'
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 is None

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: 123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'outer'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = 'inner'
    var_6 = [var_5]
    var_7 = var_4.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 123

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'list: [1, 2]\ndict: {a: b}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'list'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    var_6 = 'dict'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_tokenize_yaml_exception_predicate_is_false. Retrieved 2/14 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'valid: yaml'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_empty_string_raises_error. Retrieved 3/7 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "'hello world'"
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'foo: bar'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_exists. Retrieved 3/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 3/8 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 3/8 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/7 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/5 statements.
# Partially parsed test_validate_yaml_list_input. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name: : invalid'

def test_case_0():
    var_0 = 'name: John\nage: not_an_int'

def test_case_0():
    var_0 = '   '

def test_case_0():
    var_0 = b'Hello World'

def test_case_0():
    var_0 = '- item1\n- item2'



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 8/13 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 5/9 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'name: John\nage: 30'
    var_7 = 0

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'age: not_an_integer'

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.String()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = '   '
    var_1 = module_0.String()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = b'name: John'
    var_1 = module_0.String()
    var_2 = '_get_value'



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = module_1.Field()
    var_5 = '1'
    var_6 = module_2.validate_yaml(var_5, var_4)



# Parsed testcases at query #9
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_avoids_assertion_error_on_empty_problem. Retrieved 2/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'some content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 9/13 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = [var_2]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Should have raised ParseError'
    var_3 = AssertionError(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    a:\n      - 1\n      - 2\n    b: true\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = 0
    var_6 = [var_2, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: : mapping'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    list:\n      - item1\n      - item2\n    dict:\n      a: 1\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a: [1, 2]'
    var_3 = module_0.tokenize_yaml(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/4 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'name: test'
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_2 = 'Should have raised ParseError for invalid YAML'
    var_3 = AssertionError(var_2)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\nnested:\n  list:\n    - item1\n    - item2\n  val: 42\n'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/11 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 5/23 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/6 statements.
# Partially parsed test_validate_yaml_null_handling. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name'
    var_1 = 'name: 123'
    var_2 = 'age'
    var_3 = 'age: not_an_int'
    var_4 = 0

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)

def test_case_0():
    var_0 = '   '

def test_case_0():
    var_0 = b'name: John'

def test_case_0():
    var_0 = 'data'
    var_1 = True
    var_2 = 'data: null'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_exists. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_token_list_token_access. Retrieved 11/15 statements.
# Partially parsed test_token_dict_token_access. Retrieved 19/23 statements.
# Partially parsed test_position_equality. Retrieved 7/8 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 0
    var_2 = 4
    var_3 = 'hello world'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = '2'
    var_6 = module_0.ScalarToken(var_4, var_4, var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = '[1, 2]'
    var_10 = module_0.ListToken(var_7, var_1, var_8, var_9)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.ScalarToken(var_0, var_1, var_1, var_0)
    var_3 = 1
    var_4 = 2
    var_5 = '1'
    var_6 = module_0.ScalarToken(var_3, var_4, var_4, var_5)
    var_7 = 'b'
    var_8 = 4
    var_9 = module_0.ScalarToken(var_7, var_8, var_8, var_7)
    var_10 = 6
    var_11 = '2'
    var_12 = module_0.ScalarToken(var_4, var_10, var_10, var_11)
    var_13 = {var_0: var_6, var_7: var_12}
    var_14 = {var_0: var_6, var_7: var_12}
    var_15 = 7
    var_16 = 'a: 1, b: 2'
    var_17 = [var_0]
    var_18 = [var_0]

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 4
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = 2
    var_6 = module_0.Position(var_5, var_0, var_1)

import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.Position(var_0, var_1, var_0)
    var_3 = repr(var_2)
    assert var_3 == 'Position(line_no=1, column_no=2, char_index=1)'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 'abcde'
    var_3 = module_0.ScalarToken(var_0, var_0, var_1, var_2)



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_assert_yaml_not_none. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 3/10 statements.
# Partially parsed test_validate_yaml_success_with_types. Retrieved 3/10 statements.
# Partially parsed test_validate_yaml_error_required_field. Retrieved 4/13 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 3/11 statements.
# Partially parsed test_validate_yaml_type_error. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'

def test_case_0():
    var_0 = 'name'
    var_1 = 'age: 30'
    var_2 = 'ValidationError should have been raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = ''
    var_1 = 'Should raise error for empty content'
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 'not_an_int'



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed list'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_yaml_asserts_yaml_not_none. Retrieved 4/13 statements.
# Partially parsed test_validate_yaml_fails_when_yaml_is_none. Retrieved 6/16 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = str(var_0)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'typesystem.tokenize.tokenize_yaml'
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = 'AssertionError was not raised'
    var_5 = RuntimeError(var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_asserts_yaml_is_not_none. Retrieved 4/14 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = 'yaml'



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ' : invalid'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'some content'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'Line 84 assertion was not triggered'
    var_3 = AssertionError(var_2)



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_tokenize_yaml_with_valid_content. Retrieved 2/6 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'name: value'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'foo: bar'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_tokenize_yaml_asserts_yaml_is_not_none. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_yaml_success. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_success_mapping. Retrieved 2/14 statements.
# Partially parsed test_validate_yaml_parse_error. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 1/12 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/9 statements.


def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = 'age'
    var_1 = 'age: 25'

def test_case_0():
    var_0 = ': invalid'

def test_case_0():
    var_0 = '"not an int"'

def test_case_0():
    var_0 = ''



