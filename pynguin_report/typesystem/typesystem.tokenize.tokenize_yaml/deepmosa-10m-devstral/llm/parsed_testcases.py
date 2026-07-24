####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml_raises_parse_error. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_special_types. Retrieved 14/18 statements.


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
    var_2 = var_1.string
    assert var_2 == 'hello'
    var_3 = var_1.value
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == '[1, 2, 3]'
    var_3 = var_1.value
    var_4 = bool(var_1.value == [1, 2, 3])
    assert var_4 is True
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == '{a: 1, b: 2}'
    var_3 = var_1.value
    var_4 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_4 is True
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
    var_0 = 'a:\n  b:\n    - c\n    - d'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': ['c', 'd']}})
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == {'b': ['c', 'd']})
    assert var_8 is True
    var_9 = [var_4, var_4]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    var_12 = bool(var_11 == ['c', 'd'])
    assert var_12 is True
    var_13 = [var_4, var_4, var_4]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 'c'
    var_16 = 1
    var_17 = [var_4, var_4, var_16]
    var_18 = var_1.lookup(var_17)
    var_19 = var_18.value
    assert var_19 == 'd'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2, var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'b'
    var_6 = var_4.string
    assert var_6 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: c\n  d: e\n    f: g'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 9
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'int: 1\nfloat: 1.0\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'int': 1, 'float': 1.0, 'bool': True, 'null': None})
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = 1
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = 2
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = 3
    var_14 = [var_13]
    var_15 = var_1.lookup(var_14)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_syntax. Retrieved 3/7 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_multiline_content. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_special_types. Retrieved 14/19 statements.


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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 9
    var_10 = 8
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 11
    var_10 = 10
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: 1, b: 2,}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '.'
    var_3 = bool(var_1)
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 3
    var_11 = 6
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'int': 1, 'float': 1.5, 'bool': True, 'null': None})
    assert var_3 is True
    var_4 = 'int'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = 'float'
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = 'bool'
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = 'null'
    var_14 = [var_13]
    var_15 = var_1.lookup(var_14)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_yaml_with_invalid_validation. Retrieved 6/8 statements.


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
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    assert var_3 is None

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = 1
    var_3 = 0
    var_4 = var_2 / var_3
    var_5 = module_1.validate_yaml(var_0, var_1)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other_key: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '123: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenize_yaml_without_pyyaml. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 29/32 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 22/24 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 8/9 statements.


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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'a'
    var_8 = [var_4]
    var_9 = var_1.lookup(var_8)
    var_10 = 0
    var_11 = [var_4, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 1
    var_14 = 1
    var_15 = [var_4, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 2
    var_18 = 'b'
    var_19 = [var_18]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'
    var_22 = [var_18]
    var_23 = var_1.lookup(var_22)
    var_24 = 'c'
    var_25 = [var_18, var_24]
    var_26 = var_1.lookup_key(var_25)
    var_27 = var_26.value
    assert var_27 == 'c'
    var_28 = [var_18, var_24]
    var_29 = var_1.lookup(var_28)
    var_30 = var_29.value
    assert var_30 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 7
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = 'b'
    var_22 = [var_15, var_21]
    var_23 = var_1.lookup_key(var_22)
    var_24 = var_23.value
    assert var_24 == 'b'
    var_25 = [var_15, var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = var_26.value
    assert var_27 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 8
    var_4 = 7
    var_5 = module_1.Position(var_2, var_3, var_4)

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



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'test: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_False. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 8/9 statements.
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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 2
    var_9 = module_1.Position(var_3, var_8, var_3)
    var_10 = var_1.end
    var_11 = bool(var_1.end == var_9)
    assert var_11 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
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
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 4
    var_9 = 3
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 4
    var_9 = 3
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 9
    var_10 = 8
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 12
    var_10 = 11
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 18
    var_10 = 17
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: {b: [1, 2, 3]}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: {b: 1}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup_key(var_4)
    var_6 = var_5.value
    assert var_6 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1, 'c': 2}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 5
    var_11 = 10
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2, 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 14/15 statements.
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2
    var_23 = [var_15]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'a'
    var_26 = [var_19]
    var_27 = var_1.lookup_key(var_26)
    var_28 = var_27.value
    assert var_28 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 7
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = 'b'
    var_17 = [var_15, var_16]
    var_18 = var_1.lookup(var_17)
    var_19 = var_18.value
    assert var_19 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ScannerError(var_0)
    var_2 = var_1.problem
    assert var_2 is None
    var_3 = var_1.problem_mark
    assert var_3 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_line_84_predicate_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = bool(not (var_1.problem is not None and var_1.problem_mark is not None))
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 7/8 statements.


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
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'test'
    var_3 = var_1.string
    assert var_3 == 'test'
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_0 = 'a:\n  b:\n    - 1\n    - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 0
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1
    var_10 = [var_4, var_5]
    var_11 = var_1.lookup_key(var_10)
    var_12 = var_11.value
    assert var_12 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: [1\n    2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'test'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.string
    assert var_8 == 'c'



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #14
#--------------------------




import yaml.parser as module_0

def test_case_0():
    var_0 = 'test problem'
    var_1 = None
    var_2 = module_0.ParserError(var_0, problem_mark=var_1)
    var_3 = var_2.problem_mark
    assert var_3 is None



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'example: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 4/16 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = ''
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = 'yaml'



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_yaml_is_none. Retrieved 3/6 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 26/29 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 23/24 statements.
# Partially parsed test_tokenize_yaml_bytes. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 3/7 statements.


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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 2
    var_9 = module_1.Position(var_3, var_8, var_3)
    var_10 = var_1.end
    var_11 = bool(var_1.end == var_9)
    assert var_11 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
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
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 4
    var_9 = 3
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 4
    var_9 = 3
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 9
    var_10 = 8
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True
    var_14 = [var_5]
    var_15 = var_1.lookup(var_14)
    var_16 = var_15.value
    assert var_16 == 1
    var_17 = [var_4]
    var_18 = var_1.lookup(var_17)
    var_19 = var_18.value
    assert var_19 == 2
    var_20 = 2
    var_21 = [var_20]
    var_22 = var_1.lookup(var_21)
    var_23 = var_22.value
    assert var_23 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 11
    var_10 = 10
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True
    var_14 = 'a'
    var_15 = [var_14]
    var_16 = var_1.lookup_key(var_15)
    var_17 = var_16.value
    assert var_17 == 'a'
    var_18 = [var_14]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 1
    var_21 = 'b'
    var_22 = [var_21]
    var_23 = var_1.lookup_key(var_22)
    var_24 = var_23.value
    assert var_24 == 'b'
    var_25 = [var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = var_26.value
    assert var_27 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: {b: [1, 2, 3]}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2, 3]}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'a'
    var_8 = [var_4]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    var_11 = bool(var_9.value == {'b': [1, 2, 3]})
    assert var_11 is True
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_9.lookup_key(var_13)
    var_15 = var_14.value
    assert var_15 == 'b'
    var_16 = [var_12]
    var_17 = var_9.lookup(var_16)
    var_18 = var_17.value
    var_19 = bool(var_17.value == [1, 2, 3])
    assert var_19 is True
    var_20 = 0
    var_21 = [var_20]
    var_22 = var_17.lookup(var_21)
    var_23 = var_22.value
    assert var_23 == 1
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_17.lookup(var_25)
    var_27 = var_26.value
    assert var_27 == 2
    var_28 = 2
    var_29 = [var_28]
    var_30 = var_17.lookup(var_29)
    var_31 = var_30.value
    assert var_31 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 3
    var_11 = 7
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '.'
    var_3 = bool(var_1)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_yaml_parse_error_without_problem. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ParserError(var_0)
    var_2 = var_1.problem
    assert var_2 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_with_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 11/12 statements.


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
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': ['item1', 'item2']})
    assert var_3 is True
    var_4 = 'key'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.string
    assert var_8 == 'item1'
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.string
    assert var_12 == 'item2'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_boolean. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 14/15 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_invalid_syntax. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.


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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3
    var_14 = [var_9, var_10]
    var_15 = var_1.lookup_key(var_14)
    var_16 = var_15.value
    assert var_16 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 7
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #24
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
    var_0 = 'invalid yaml content'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    assert var_3 == 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key1: value1\nkey2: value2'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)
    var_9 = bool(var_8 == {'key1': 'value1', 'key2': 'value2'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key1: value1'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '123: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = module_0.Field()
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'outer': {'inner': 'value'}})
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 11/12 statements.
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
    var_9 = 6
    var_10 = 5
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
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

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
    var_10 = 5
    var_11 = 4
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
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1, 'c': 2}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 'c'
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 29/32 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 29/31 statements.


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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup_key(var_5)
    var_7 = var_6.value
    assert var_7 == 'a'
    var_8 = [var_4]
    var_9 = var_1.lookup(var_8)
    var_10 = 0
    var_11 = [var_4, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 1
    var_14 = 1
    var_15 = [var_4, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 2
    var_18 = 'b'
    var_19 = [var_18]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.value
    assert var_21 == 'b'
    var_22 = [var_18]
    var_23 = var_1.lookup(var_22)
    var_24 = 'c'
    var_25 = [var_18, var_24]
    var_26 = var_1.lookup_key(var_25)
    var_27 = var_26.value
    assert var_27 == 'c'
    var_28 = [var_18, var_24]
    var_29 = var_1.lookup(var_28)
    var_30 = var_29.value
    assert var_30 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c\n  d: e'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c', 'd': 'e'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 5
    var_11 = 11
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = 'b'
    var_22 = [var_15, var_21]
    var_23 = var_1.lookup_key(var_22)
    var_24 = var_23.value
    assert var_24 == 'b'
    var_25 = [var_15, var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = var_26.value
    assert var_27 == 'c'
    var_28 = 'd'
    var_29 = [var_15, var_28]
    var_30 = var_1.lookup_key(var_29)
    var_31 = var_30.value
    assert var_31 == 'd'
    var_32 = [var_15, var_28]
    var_33 = var_1.lookup(var_32)
    var_34 = var_33.value
    assert var_34 == 'e'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'end of the stream'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_with_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 14/15 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_with_special_types. Retrieved 18/19 statements.


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
    var_0 = 'invalid: yaml: content: :'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_0 = '{a: {b: [1, 2]}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == {'b': [1, 2]})
    assert var_8 is True
    var_9 = 'b'
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    var_13 = bool(var_12 == [1, 2])
    assert var_13 is True
    var_14 = 0
    var_15 = [var_4, var_9, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1, 'c': 2}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 5
    var_11 = 11
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'int: 42\nfloat: 3.14\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'int': 42, 'float': 3.14, 'bool': True, 'null': None})
    assert var_3 is True
    var_4 = 'int'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 42
    var_8 = 'float'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    var_12 = bool(var_11 == 3.14)
    assert var_12 is True
    var_13 = 'bool'
    var_14 = [var_13]
    var_15 = var_1.lookup(var_14)
    var_16 = var_15.value
    assert var_16 is True
    var_17 = 'null'
    var_18 = [var_17]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 is None



# Parsed testcases at query #4
#--------------------------




import yaml.scanner as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ScannerError(problem=var_0, problem_mark=var_0)
    var_2 = var_1.problem
    assert var_2 is None



# Parsed testcases at query #5
#--------------------------




import yaml.scanner as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ScannerError(problem=var_0, problem_mark=var_0)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = module_1.validate_yaml(var_0, var_3)

import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: [value'
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = module_1.validate_yaml(var_0, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = module_2.validate_yaml(var_0, var_10)
    var_12 = bool(var_11 == {'outer': {'inner': 'value'}})
    assert var_12 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = bool(not (var_1.problem is not None and var_1.problem_mark is not None))
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 9/10 statements.


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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup_key(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1, 'c': 2}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test problem'
    var_1 = 'test context'
    var_2 = 'test note'
    var_3 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_when_pyyaml_not_installed. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 3/5 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'problem'
    var_1 = 'problem_mark'
    var_2 = module_0.ParserError(var_0, var_1)
    var_3 = var_2.problem
    assert var_3 is None
    var_4 = var_2.problem_mark
    assert var_4 is None



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_without_pyyaml. Retrieved 3/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test: value'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #15
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
    var_0 = 'invalid yaml content'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = 'invalid yaml content'

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: null'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)
    assert var_3 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'parent:\n  child: value'
    var_1 = 'parent'
    var_2 = 'child'
    var_3 = module_0.Field()
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'parent': {'child': 'value'}})
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 21/22 statements.
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2
    var_23 = [var_15]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'a'
    var_26 = [var_19]
    var_27 = var_1.lookup_key(var_26)
    var_28 = var_27.value
    assert var_28 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3
    var_18 = [var_13, var_14]
    var_19 = var_1.lookup_key(var_18)
    var_20 = var_19.value
    assert var_20 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '\n    a: 1\n    b: 2\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 5
    var_11 = 13
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.start
    var_19 = 2
    var_20 = module_1.Position(var_19, var_10, var_10)
    var_21 = bool(var_18 == var_20)
    assert var_21 is True
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup(var_23)
    var_25 = var_24.start
    var_26 = 11
    var_27 = module_1.Position(var_9, var_10, var_26)
    var_28 = bool(var_25 == var_27)
    assert var_28 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
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
    var_2 = var_1.string
    assert var_2 == 'hello'
    var_3 = var_1.value
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
    var_2 = var_1.string
    assert var_2 == '42'
    var_3 = var_1.value
    assert var_3 == 42
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
    var_2 = var_1.string
    assert var_2 == '3.14'
    var_3 = var_1.value
    var_4 = bool(var_1.value == 3.14)
    assert var_4 is True
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
    var_2 = var_1.string
    assert var_2 == 'true'
    var_3 = var_1.value
    assert var_3 is True
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
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == 'null'
    var_3 = var_1.value
    assert var_3 is None
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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == '[1, 2, 3]'
    var_3 = var_1.value
    var_4 = bool(var_1.value == [1, 2, 3])
    assert var_4 is True
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == '{a: 1, b: 2}'
    var_3 = var_1.value
    var_4 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_4 is True
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
    var_0 = 'a:\n  b:\n    - 1\n    - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 0
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1
    var_10 = [var_4, var_5]
    var_11 = var_1.lookup_key(var_10)
    var_12 = var_11.value
    assert var_12 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: [1, 2, 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.string
    assert var_2 == 'hello'
    var_3 = var_1.value
    assert var_3 == 'hello'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 10/11 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.


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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 5
    var_9 = 4
    var_10 = module_1.Position(var_3, var_8, var_9)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 9
    var_10 = 8
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 11
    var_10 = 10
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 1
    var_10 = [var_9, var_4]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = 'required_key'
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
    var_1 = 'key'
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
    var_0 = 'key: value: invalid'
    var_1 = 'key'
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
    var_0 = 'outer:\n  inner: value'
    var_1 = 'outer'
    var_2 = 'inner'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = module_2.validate_yaml(var_0, var_10)
    var_12 = bool(var_11 == {'outer': {'inner': 'value'}})
    assert var_12 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = bool(not (var_1.problem is not None and var_1.problem_mark is not None))
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 13/14 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_with_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 11/12 statements.


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

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = var_1._value
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 0
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = 1
    var_12 = [var_11]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 2
    var_15 = 2
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = var_1.lookup_key(var_6)
    var_8 = var_7.value
    assert var_8 == 'a'
    var_9 = [var_5]
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
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: [1, 2,}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b: 1\n  c: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1, 'c': 2}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 'c'
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'test: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    assert var_3 is None



# Parsed testcases at query #24
#--------------------------




import yaml.scanner as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ScannerError(problem=var_0, problem_mark=var_0)
    var_2 = var_1.problem
    assert var_2 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_yaml_parse_error_without_problem. Retrieved 2/11 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 17/18 statements.


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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{a: 1, b: 2}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{a: 1, b: 2}'
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
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 'c'}})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 7
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = 'b'
    var_20 = [var_15, var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: [b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 5
    var_4 = 4
    var_5 = module_1.Position(var_2, var_3, var_4)



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
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
    var_0 = ''
    var_1 = 'key'
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
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'type'
    var_10 = e.messages()[0].text
    assert var_10 == 'Must be a string.'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'parent:\n  child: 123'
    var_1 = 'parent'
    var_2 = 'child'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = module_2.validate_yaml(var_0, var_10)
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'type'
    var_14 = e.messages()[0].text
    assert var_14 == 'Must be a string.'
    var_15 = e.messages()[0].index
    var_16 = bool(e.messages()[0].index == ['parent', 'child'])
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = 'required_key'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'required'
    var_13 = e.messages()[0].text
    assert var_13 == 'This field is required.'
    var_14 = e.messages()[0].index
    var_15 = bool(e.messages()[0].index == ['required_key'])
    assert var_15 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = bool(not (var_1.problem is not None and var_1.problem_mark is not None))
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 2/4 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ScannerError(var_0)
    var_2 = var_1.problem
    assert var_2 is None



