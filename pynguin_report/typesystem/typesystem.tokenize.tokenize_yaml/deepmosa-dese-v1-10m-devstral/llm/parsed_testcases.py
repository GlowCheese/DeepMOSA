####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_special_types. Retrieved 11/18 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: {b: [1, 2]}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = [var_2, var_3]
    var_9 = var_1.lookup_key(var_8)
    var_10 = var_9.value
    assert var_10 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 10
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = '3.14'
    var_4 = module_0.tokenize_yaml(var_3)
    var_5 = var_4.value
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = var_7.value
    var_9 = 'null'
    var_10 = module_0.tokenize_yaml(var_9)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_yaml_invalid_yaml. Retrieved 3/8 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
    var_2 = module_0.String()

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other_key: value'
    var_1 = 'key'
    var_2 = 'other_key'
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = module_0.String()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 22/23 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3
    var_16 = [var_11, var_12]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 1
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 2
    var_10 = [var_2]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.start
    var_13 = 1
    var_14 = 0
    var_15 = module_1.Position(var_13, var_13, var_14)
    var_16 = [var_6]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.start
    var_19 = 2
    var_20 = 4
    var_21 = module_1.Position(var_19, var_13, var_20)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2,}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 12
    var_4 = 11
    var_5 = module_1.Position(var_2, var_3, var_4)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    assert var_2 == 'value'

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: [value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'required_key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'person:\n  name: John\n  age: 30'
    var_1 = 'person'
    var_2 = 'name'
    var_3 = 'age'
    var_4 = module_0.Field()
    var_5 = module_0.Field()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = {var_1: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3
    var_16 = [var_11, var_12]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 1
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 2
    var_10 = [var_2]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.start
    var_13 = 1
    var_14 = 0
    var_15 = module_1.Position(var_13, var_13, var_14)
    var_16 = [var_6]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.start
    var_19 = 2
    var_20 = 5
    var_21 = module_1.Position(var_19, var_13, var_20)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_special_types. Retrieved 18/19 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 12
    var_6 = 11
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b:\n    - 1\n    - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 15
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b:\n    - 1\n    - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = [var_2, var_3]
    var_9 = var_1.lookup_key(var_8)
    var_10 = var_9.value
    assert var_10 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2, 3'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: |-\n    line1\n    line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.string
    assert var_6 == 'line1\n    line2'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'int: 42\nfloat: 3.14\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'int'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 42
    var_6 = 'float'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    var_10 = 'bool'
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 is True
    var_14 = 'null'
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test problem'
    var_1 = module_0.ParserError(var_0)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_multiline_content. Retrieved 15/16 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup_key(var_9)
    var_11 = var_10.value
    assert var_11 == 'a'
    var_12 = [var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 1
    var_15 = 'b'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'b'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\nlist:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = 'list'
    var_7 = 0
    var_8 = [var_6, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 'item1'
    var_11 = 1
    var_12 = [var_6, var_11]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 'item2'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 14/15 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_invalid_syntax. Retrieved 3/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    var_10 = 'c'
    var_11 = [var_6, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '\n    a: 1\n    b: 2\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 2
    var_3 = 5
    var_4 = module_1.Position(var_2, var_3, var_3)
    var_5 = 3
    var_6 = 6
    var_7 = 12
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '.'



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = module_0.String()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 14/15 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    var_6 = 'b'
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    var_10 = 'c'
    var_11 = [var_6, var_10]
    var_12 = var_1.lookup_key(var_11)
    var_13 = var_12.value
    assert var_13 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: b: c'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: {b: [1, 2]}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = 1
    var_9 = [var_2, var_3, var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 'c'
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 12
    var_6 = 11
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b:\n    - 1\n    - 2\n  c: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = [var_2, var_3]
    var_9 = var_1.lookup_key(var_8)
    var_10 = var_9.value
    assert var_10 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [b: c]'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_boolean. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 15/16 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3
    var_16 = [var_11, var_12]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\nlist:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = 'list'
    var_7 = 0
    var_8 = [var_6, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 'item1'
    var_11 = 1
    var_12 = [var_6, var_11]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 'item2'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_yaml_with_list. Retrieved 4/8 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.Integer()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = module_0.String()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key:\n  - item1\n  - item2'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = [var_2]

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = [var_2, var_3]
    var_5 = module_0.Union(var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = 'required_key'
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_syntax. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 'c'
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 4/14 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = 'yaml'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 9
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2\nc: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 9
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b:\n    - c\n    - d\ne: f'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 0
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 'c'
    var_8 = [var_2, var_3]
    var_9 = var_1.lookup_key(var_8)
    var_10 = var_9.value
    assert var_10 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: [b: c]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 5
    var_4 = 4
    var_5 = module_1.Position(var_2, var_3, var_4)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_invalid_yaml. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 13
    var_6 = 12
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_6, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 17/18 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 15
    var_6 = 14
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3
    var_16 = [var_11, var_12]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: c\n  d: e'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 'c'
    var_7 = 'd'
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 'e'
    var_11 = [var_2, var_3]
    var_12 = var_1.lookup_key(var_11)
    var_13 = var_12.value
    assert var_13 == 'b'
    var_14 = [var_2, var_7]
    var_15 = var_1.lookup_key(var_14)
    var_16 = var_15.value
    assert var_16 == 'd'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [b: c]'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'MockException'
    var_1 = ()
    var_2 = 'problem'
    var_3 = 'problem_mark'
    var_4 = None
    var_5 = {var_2: var_4, var_3: var_4}



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 2/7 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 20/21 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 2
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_1.Position(var_2, var_5, var_2)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = [var_3]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = [var_2]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)
    var_8 = 'a'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2
    var_16 = [var_8]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_12]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup(var_4)
    var_6 = var_5.value
    assert var_6 == 1
    var_7 = 1
    var_8 = [var_2, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 2
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_11, var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 3
    var_16 = [var_11, var_12]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 7
    var_8 = module_1.Position(var_5, var_6, var_7)
    var_9 = 'a'
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.start
    var_13 = module_1.Position(var_2, var_2, var_3)
    var_14 = 'b'
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.start
    var_18 = 5
    var_19 = module_1.Position(var_5, var_2, var_18)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'foo'
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #14
#--------------------------




import yaml.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ParserError(var_0, var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_problem_is_none. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'MockExc'
    var_1 = ()
    var_2 = 'problem'
    var_3 = 'problem_mark'
    var_4 = None
    var_5 = 'Mark'
    var_6 = ()
    var_7 = 'index'
    var_8 = 0
    var_9 = {var_7: var_8}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_yaml_list_validation. Retrieved 3/7 statements.
# Partially parsed test_validate_yaml_positional_error_messages. Retrieved 6/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other: value'
    var_1 = 'key'
    var_2 = 'other'
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'nested:\n  key: value'
    var_1 = 'nested'
    var_2 = 'key'
    var_3 = module_0.String()
    var_4 = {var_2: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = {var_1: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'items:\n  - item1\n  - item2'
    var_1 = 'items'
    var_2 = module_0.String()

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = module_0.String()
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'key: value'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test: value'
    var_2 = module_0.Field()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_yaml_import_failure. Retrieved 3/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_yaml_parse_error_with_problem_and_problem_mark. Retrieved 10/13 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = 'context'
    var_2 = 'problem'
    var_3 = None
    var_4 = module_0.ParserError(var_1, var_2, var_3, var_3, var_3)
    var_5 = 'Mark'
    var_6 = ()
    var_7 = 'index'
    var_8 = 10
    var_9 = {var_7: var_8}



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ParserError(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_yaml_parse_error_without_problem. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



