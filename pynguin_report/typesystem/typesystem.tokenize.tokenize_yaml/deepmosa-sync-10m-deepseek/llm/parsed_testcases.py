####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - two'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 'two']})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - two'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 6
    var_12 = 17
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_simple_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_in_dict. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_in_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value\nanother: 123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value', 'another': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value\nanother: 123'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 10
    var_12 = 23
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'list:\n  - 1\n  - 2\ndict:\n  key: val'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'dict': {'key': 'val'}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'list:\n  - 1\n  - 2\ndict:\n  key: val'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 5
    var_11 = 9
    var_12 = 34
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'test: data'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 'data'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'test: data'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 2
    var_6 = var_4.string
    assert var_6 == '2'
    var_7 = 2
    var_8 = 4
    var_9 = 7
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- x\n- y\n- z'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'y'
    var_6 = var_4.string
    assert var_6 == 'y'
    var_7 = 2
    var_8 = 3
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: !!invalid_tag value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_fails_when_yaml_is_none. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'yaml'



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_yaml_valid_yaml. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_invalid_yaml_parse_error. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 1/10 statements.
# Partially parsed test_validate_yaml_validation_error_with_positions. Retrieved 2/16 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 1/8 statements.
# Partially parsed test_validate_yaml_with_schema_validation_error. Retrieved 6/11 statements.
# Partially parsed test_validate_yaml_nested_structure_validation_error. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'key: value'

def test_case_0():
    var_0 = 'key: ['

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'key'
    var_1 = 'other: value'

def test_case_0():
    var_0 = b'key: value'

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
    var_6 = 'age: 25'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'age': 25})
    assert var_8 is True

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
    var_6 = 'age: invalid'
    var_7 = module_2.validate_yaml(var_6, var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'count'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'data'
    var_7 = 'name'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_6: var_5, var_7: var_9}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'data:\n  count: 5\nname: test'
    var_14 = module_2.validate_yaml(var_13, var_12)
    var_15 = bool(var_14 == {'data': {'count': 5}, 'name': 'test'})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'count'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'data'
    var_7 = 'name'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_6: var_5, var_7: var_9}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'data:\n  count: invalid\nname: test'
    var_14 = module_2.validate_yaml(var_13, var_12)



# Parsed testcases at query #8
#--------------------------




import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = {}
    var_3 = {}
    var_4 = module_0.Schema(var_2, **var_3)
    var_5 = module_1.validate_yaml(var_1, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 15
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'did not find expected node content'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'a'
    var_6 = var_4.string
    assert var_6 == 'a'
    var_7 = 1
    var_8 = 3
    var_9 = 2
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_multiline_scalar. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_complex_structure. Retrieved 4/5 statements.
# Partially parsed test_tokenize_yaml_scalar_hash. Retrieved 3/4 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 18
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '&anchor value\n*anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['value', 'value'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\n*anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 7
    var_12 = 19
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    - name: John\n      age: 30\n      hobbies:\n        - reading\n        - hiking\n    - name: Jane\n      age: 25\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_1.value[0]['name']
    assert var_4 == 'John'
    var_5 = var_1.value[0]['age']
    assert var_5 == 30
    var_6 = var_1.value[0]['hobbies']
    var_7 = bool(var_1.value[0]['hobbies'] == ['reading', 'hiking'])
    assert var_7 is True
    var_8 = var_1.value[1]['name']
    assert var_8 == 'Jane'
    var_9 = var_1.value[1]['age']
    assert var_9 == 25

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = module_0.tokenize_yaml(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True
    var_4 = 'world'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = bool(var_1 != var_5)
    assert var_6 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = hash(var_1)



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '123.45'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 6
    var_11 = 5
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 15
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  hello\n  world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello\nworld\n'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 3
    var_9 = 7
    var_10 = 19
    var_11 = module_1.Position(var_8, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 14
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'a'
    var_6 = var_4.string
    assert var_6 == 'a'
    var_7 = 1
    var_8 = 3
    var_9 = 2
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 9/16 statements.


import yaml.scanner as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'ScannerError'
    var_1 = 'problem'
    var_2 = 'problem_mark'
    var_3 = 'test'
    var_4 = None
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.ScannerError(var_3)
    var_7 = 'invalid: @'
    var_8 = module_1.tokenize_yaml(var_7)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_empty_string. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_mixed_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42

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
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
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
    var_0 = 'key:\n  nested: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': {'nested': 'value'}})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'list:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': ['item1', 'item2']})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = var_1.start
    var_6 = bool(var_1.start == var_4)
    assert var_6 is True
    var_7 = 11
    var_8 = 10
    var_9 = module_1.Position(var_2, var_7, var_8)
    var_10 = var_1.end
    var_11 = bool(var_1.end == var_9)
    assert var_11 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == 'key: value'

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
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_handles_scanner_error_without_problem. Retrieved 4/8 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = module_0.ScannerError(var_0, var_1, var_0, var_1)
    var_3 = ''



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: :'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_empty_string. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_mixed_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'No content.'
    var_3 = 'no_content'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42

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
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['item1', 'item2'])
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
    var_0 = 'key:\n  nested_key: nested_value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': {'nested_key': 'nested_value'}})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'list:\n  - 1\n  - two\ndict:\n  key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'list'
    var_3 = 'dict'
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = var_1.value
    var_12 = bool(var_1.value == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.start.line_no
    assert var_2 == 1
    var_3 = var_1.start.column_no
    assert var_3 == 1
    var_4 = var_1.end.line_no
    assert var_4 == 1
    var_5 = var_1.end.column_no
    assert var_5 == 10

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == 'key: value'

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
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value\nother: 123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value', 'other': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value\nother: 123'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 9
    var_12 = 22
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'list:\n  - item1\n  - item2\ndict:\n  key: val'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': ['item1', 'item2'], 'dict': {'key': 'val'}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'list:\n  - item1\n  - item2\ndict:\n  key: val'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 5
    var_11 = 9
    var_12 = 41
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello: world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'hello': 'world'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'hello: world'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 12
    var_11 = 11
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 2
    var_6 = var_4.string
    assert var_6 == '2'
    var_7 = 2
    var_8 = 4
    var_9 = 7
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- x\n- y'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'y'
    var_6 = var_4.string
    assert var_6 == 'y'
    var_7 = 2
    var_8 = 3
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = module_1.Position(var_7, var_8, var_9)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 2/21 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'invalid: \x81\x82'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'parse_error'
    var_3 = bool('parse_error' in exc.text.lower())
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 6
    var_11 = 18
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '&anchor value\nother: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'other': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\nother: *anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 14
    var_12 = 27
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - two'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 'two']})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - two'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 6
    var_12 = 17
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '123.45'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 6
    var_11 = 5
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 15
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 19
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '&anchor value\nkey: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\nkey: *anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 12
    var_12 = 24
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key: value'
    var_2 = module_1.validate_yaml(var_1, var_0)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_yaml_valid_yaml. Retrieved 2/14 statements.
# Partially parsed test_validate_yaml_invalid_yaml_parse_error. Retrieved 2/14 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 2/14 statements.
# Partially parsed test_validate_yaml_validation_error_with_positions. Retrieved 4/17 statements.
# Partially parsed test_validate_yaml_required_field_missing. Retrieved 3/14 statements.
# Partially parsed test_validate_yaml_nested_schema_validation. Retrieved 6/28 statements.
# Partially parsed test_validate_yaml_bytes_input. Retrieved 2/14 statements.
# Partially parsed test_validate_yaml_allow_null_schema. Retrieved 3/13 statements.
# Partially parsed test_validate_yaml_invalid_key_non_string. Retrieved 3/14 statements.
# Partially parsed test_validate_yaml_default_value_applied. Retrieved 2/15 statements.
# Partially parsed test_validate_yaml_read_only_field_ignored. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'name: John'

def test_case_0():
    var_0 = 'name'
    var_1 = 'name: ['

def test_case_0():
    var_0 = 'name'
    var_1 = ''

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 'name: 123'
    var_5 = len(exc.messages())
    assert var_5 == 1
    var_6 = exc.messages()[0]
    var_7 = var_6.code
    assert var_7 == 'type'
    var_8 = var_6.start_position
    var_9 = bool(var_6.start_position is not None)
    assert var_9 is True
    var_10 = var_6.end_position
    var_11 = bool(var_6.end_position is not None)
    assert var_11 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'age: 30'
    var_2 = len(exc.messages())
    assert var_2 == 1
    var_3 = exc.messages()[0]
    var_4 = var_3.code
    assert var_4 == 'required'
    var_5 = var_3.index
    var_6 = bool(var_3.index == ['name'])
    assert var_6 is True

def test_case_0():
    var_0 = 'type'
    var_1 = 'Must be a string.'
    var_2 = {var_0: var_1}
    var_3 = 'type'
    var_4 = 'Must be a number.'
    var_5 = {var_3: var_4}
    var_6 = 'city'
    var_7 = 'address'
    var_8 = 'age'
    var_9 = 'address:\n  city: 123\nage: thirty'
    var_10 = 'type'

def test_case_0():
    var_0 = 'name'
    var_1 = b'name: Alice'

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = 'null'

def test_case_0():
    var_0 = 'name'
    var_1 = '123: value'
    var_2 = len(exc.messages())
    assert var_2 == 1
    var_3 = exc.messages()[0]
    var_4 = var_3.code
    assert var_4 == 'invalid_key'
    var_5 = var_3.index
    var_6 = bool(var_3.index == [123])
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = '{}'

def test_case_0():
    var_0 = 'name'
    var_1 = 'name: John'



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'test: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'test: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 11
    var_11 = 10
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_yaml_validation_error_with_positions. Retrieved 8/13 statements.
# Partially parsed test_validate_yaml_invalid_key_type. Retrieved 6/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'name': 'John Doe', 'age': '30'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John Doe\n  age: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'name: John Doe\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'name': 'John Doe', 'age': '30'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'person:\n  name: John\n  age: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = 'person'
    var_11 = {var_10: var_9}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = module_2.validate_yaml(var_0, var_13)
    var_15 = bool(var_14 == {'person': {'name': 'John', 'age': 30}})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: null'
    var_1 = 'name'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'name': None})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '123: value'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'value: 42'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = var_2 | var_4
    var_6 = 'value'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'value': 42})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '{}'
    var_1 = 'name'
    var_2 = 'Anonymous'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'name': 'Anonymous'})
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '123.45'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 6
    var_11 = 5
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 16
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'did not find expected node content'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_error_when_yaml_is_none. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'yaml'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 5/19 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test problem'
    var_1 = None
    var_2 = ()
    var_3 = 'dummy content'
    var_4 = module_0.tokenize_yaml(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_yaml_with_schema_validation_error. Retrieved 8/13 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '42'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 42

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'invalid: ['
    var_3 = module_1.validate_yaml(var_2, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = ''
    var_3 = module_1.validate_yaml(var_2, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '   \n   '
    var_3 = module_1.validate_yaml(var_2, var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'age'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'age: 30\nname: John'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(var_10 == {'age': 30, 'name': 'John'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'age'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'age: thirty\nname: John'
    var_10 = module_2.validate_yaml(var_9, var_8)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = b'hello'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 'hello'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'id'
    var_1 = 'title'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'id: 1\ntitle: Test'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(var_10 == {'id': 1, 'title': 'Test'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = 'null'
    var_5 = module_1.validate_yaml(var_4, var_3)
    assert var_5 is None

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key: [unclosed'
    var_3 = module_1.validate_yaml(var_2, var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_handles_scanner_error_without_problem. Retrieved 4/15 statements.


import yaml.scanner as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ScannerError(var_0)
    var_2 = 'test'
    var_3 = module_1.tokenize_yaml(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml_raises_assertion_error. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '123.45'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 6
    var_11 = 5
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 15
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 19
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '&anchor value\nkey: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\nkey: *anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 13
    var_12 = 26
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_yaml_empty_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_empty_bytes. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_parse_error. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_whitespace_only. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_unicode. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_quoted_scalar. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_flow_sequence. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_flow_mapping. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 5

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 4

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 4

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 5

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 1
    var_7 = var_1.end.column_no
    assert var_7 == 4

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 2
    var_8 = var_1.end.column_no
    assert var_8 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 10

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.end.line_no
    assert var_6 == 2
    var_7 = var_1.end.column_no
    assert var_7 == 10

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '|\n  hello\n  world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello\nworld\n'
    var_3 = var_1.start.line_no
    assert var_3 == 1
    var_4 = var_1.start.column_no
    assert var_4 == 1
    var_5 = var_1.end.line_no
    assert var_5 == 3
    var_6 = var_1.end.column_no
    assert var_6 == 7

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '&anchor value\n*anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['value', 'value'])
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'café: naïve'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'café': 'naïve'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'café: naïve'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "'42'"
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == '42'
    var_3 = var_1.string
    assert var_3 == "'42'"

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '[a, b, c]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[a, b, c]'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_multiline_scalar. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_complex_mapping. Retrieved 9/14 statements.
# Partially parsed test_tokenize_yaml_quoted_scalars. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 14
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 18
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '&anchor value\n*anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['value', 'value'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\n*anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 7
    var_12 = 19
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '? complex key\n: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = var_1.string
    assert var_3 == '? complex key\n: value'
    var_4 = 1
    var_5 = module_1.Position(var_4, var_4, var_2)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 2
    var_9 = 7
    var_10 = 20
    var_11 = module_1.Position(var_8, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '\'single\'\n"double"'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['single', 'double'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '\'single\'\n"double"'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 8
    var_12 = 16
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



