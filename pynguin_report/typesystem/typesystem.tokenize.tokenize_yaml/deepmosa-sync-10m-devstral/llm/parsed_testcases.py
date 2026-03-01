####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 17/18 statements.
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
    var_10 = 12
    var_11 = 11
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
    var_13 = [var_4, var_5]
    var_14 = var_1.lookup_key(var_13)
    var_15 = var_14.value
    assert var_15 == 'b'
    var_16 = [var_4, var_9]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'c'

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
    var_0 = 'a: [1, 2, 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_False. Retrieved 1/3 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = module_0.ScannerError()
    var_1 = var_0.problem
    assert var_1 is None



# Parsed testcases at query #3
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
    var_8 = bool(False)
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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other_key: value'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'required'

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
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'key': None})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'null'



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_yaml_valid_input. Retrieved 2/6 statements.
# Partially parsed test_validate_yaml_invalid_yaml. Retrieved 2/8 statements.
# Partially parsed test_validate_yaml_validation_error. Retrieved 2/7 statements.
# Partially parsed test_validate_yaml_empty_content. Retrieved 2/7 statements.
# Partially parsed test_validate_yaml_missing_required_field. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'

def test_case_0():
    var_0 = 'key: [value'
    var_1 = 'key'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = e.messages()[0].code
    assert var_3 == 'type'

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'other: value'
    var_1 = 'key'
    var_2 = 'other'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = e.messages()[0].code
    assert var_4 == 'required'



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_yaml_list_input. Retrieved 3/7 statements.


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
    var_8 = bool(False)
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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = e.messages()[0].code
    assert var_9 == 'type'

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
    var_8 = bool(False)
    assert var_8 is True

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'items:\n  - item1\n  - item2'
    var_1 = 'items'
    var_2 = {}
    var_3 = module_0.String(**var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'value: 123'
    var_1 = 'value'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = {var_1: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = module_2.validate_yaml(var_0, var_11)
    var_13 = bool(var_12 == {'value': 123})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'value: null'
    var_1 = 'value'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'value': None})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other: value'
    var_1 = 'required'
    var_2 = 'other'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = e.messages()[0].code
    assert var_12 == 'required'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_special_values. Retrieved 14/15 statements.


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

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'null: null\nbool: true\nfloat: 3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'null': None, 'bool': True, 'float': 3.14})
    assert var_3 is True
    var_4 = 'null'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 is None
    var_8 = 'bool'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 is True
    var_12 = 'float'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    var_16 = bool(var_15 == 3.14)
    assert var_16 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_exc_problem_is_none. Retrieved 2/3 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ScannerError(var_0)
    var_2 = var_1.problem
    assert var_2 is None



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'problem'
    var_1 = 'context'
    var_2 = 'note'
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 23/24 statements.
# Partially parsed test_tokenize_yaml_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
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
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a: 1\nb: 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 7
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True
    var_16 = 'a'
    var_17 = [var_16]
    var_18 = var_1.lookup(var_17)
    var_19 = var_18.value
    assert var_19 == 1
    var_20 = 'b'
    var_21 = [var_20]
    var_22 = var_1.lookup(var_21)
    var_23 = var_22.value
    assert var_23 == 2
    var_24 = [var_16]
    var_25 = var_1.lookup_key(var_24)
    var_26 = var_25.value
    assert var_26 == 'a'
    var_27 = [var_20]
    var_28 = var_1.lookup_key(var_27)
    var_29 = var_28.value
    assert var_29 == 'b'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a:\n  b:\n    - 1\n    - 2\nc: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}, 'c': 3})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 0
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1
    var_10 = 1
    var_11 = [var_4, var_5, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 'c'
    var_15 = [var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 16/17 statements.
# Partially parsed test_tokenize_yaml_multiline. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_invalid_syntax. Retrieved 3/7 statements.
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
    var_0 = '{a: {b: [1, 2, 3]}}'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2, 3]}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 0
    var_7 = [var_4, var_5, var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 1
    var_10 = 1
    var_11 = [var_4, var_5, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 2
    var_14 = 2
    var_15 = [var_4, var_5, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = 'b'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 2
    var_12 = [var_4]
    var_13 = var_1.lookup_key(var_12)
    var_14 = var_13.start
    var_15 = 1
    var_16 = 0
    var_17 = module_1.Position(var_15, var_15, var_16)
    var_18 = bool(var_14 == var_17)
    assert var_18 is True
    var_19 = [var_8]
    var_20 = var_1.lookup_key(var_19)
    var_21 = var_20.start
    var_22 = 2
    var_23 = 4
    var_24 = module_1.Position(var_22, var_15, var_23)
    var_25 = bool(var_21 == var_24)
    assert var_25 is True

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
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_exc_problem_mark_is_none. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = var_1.problem_mark
    assert var_2 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = str(var_3)
    assert var_4 == "'pyyaml' must be installed."



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_problem_is_not_none. Retrieved 2/3 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ParserError(var_0)
    var_2 = bool(not var_1.problem is not None)
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = str(var_3)
    assert var_4 == "'pyyaml' must be installed."



# Parsed testcases at query #23
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'MockException'
    var_1 = ()
    var_2 = 'problem'
    var_3 = 'problem_mark'
    var_4 = 'test'
    var_5 = None
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = var_10.problem_mark
    assert var_11 is None



# Parsed testcases at query #24
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = bool(var_2 == {'key': 'value'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value: invalid'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = 'invalid'
    var_4 = bool('invalid' in e.text.lower())
    assert var_4 is True

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
    var_4 = bool(var_3 == {'key': 'value'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = bool(var_2 == {'key': 'value'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name: John\nage: 30'
    var_8 = module_2.validate_yaml(var_7, var_6)
    var_9 = bool(var_8 == {'name': 'John', 'age': 30})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name: John'
    var_8 = module_2.validate_yaml(var_7, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name: John\nage: thirty'
    var_8 = module_2.validate_yaml(var_7, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name'
    var_8 = 'address'
    var_9 = module_0.Field()
    var_10 = {var_7: var_9, var_8: var_6}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'name: John\naddress:\n  street: 123 Main\n  city: Anytown'
    var_14 = module_2.validate_yaml(var_13, var_12)
    var_15 = bool(var_14 == {'name': 'John', 'address': {'street': '123 Main', 'city': 'Anytown'}})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'items'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'items:\n  - item1\n  - item2'
    var_6 = module_2.validate_yaml(var_5, var_4)
    var_7 = bool(var_6 == {'items': ['item1', 'item2']})
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 2/3 statements.


import yaml.scanner as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ScannerError(var_0)
    var_2 = var_1.problem
    assert var_2 is None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = 'test problem'
    var_1 = module_0.ParserError(var_0)
    var_2 = var_1.problem
    assert var_2 is None
    var_3 = var_1.problem_mark
    assert var_3 is None



# Parsed testcases at query #29
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
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'other_key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(error.messages())
    assert var_9 == 1
    var_10 = error.messages()[0].code
    assert var_10 == 'required'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'key': 123})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'key': None})
    assert var_10 is True

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
    var_8 = bool(False)
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_8 = bool(False)
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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other_key: value'
    var_1 = 'key'
    var_2 = 'other_key'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'required'

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
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'key': None})
    assert var_10 is True

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_list_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_multiline_content. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_special_types. Retrieved 14/18 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(e.text.endswith('.'))
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

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
    var_10 = 12
    var_11 = 11
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
    var_0 = 'key: value\nlist:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value', 'list': ['item1', 'item2']})
    assert var_3 is True
    var_4 = 'list'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 'item1'
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 'item2'

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




import builtins as module_0

def test_case_0():
    var_0 = 'MockException'
    var_1 = ()
    var_2 = 'problem'
    var_3 = 'problem_mark'
    var_4 = None
    var_5 = 'Mark'
    var_6 = ()
    var_7 = 'index'
    var_8 = 0
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = {var_2: var_4, var_3: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = var_18.problem
    assert var_19 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 3/7 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = str(var_0)
    assert var_2 == "'pyyaml' must be installed."



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'invalid yaml: ['
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_yaml_parse_error_with_none_problem. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = ''



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list_token. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict_token. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 14/15 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml_raises_parse_error. Retrieved 2/5 statements.


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
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['item1', 'item2'])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 7
    var_11 = 12
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1: value1\nkey2: value2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key1': 'value1', 'key2': 'value2'})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 10
    var_11 = 19
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'list:\n  - item1\n  - item2\nnested:\n  key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': ['item1', 'item2'], 'nested': {'key': 'value'}})
    assert var_3 is True
    var_4 = 0
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == ['item1', 'item2'])
    assert var_8 is True
    var_9 = 1
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    var_13 = bool(var_12 == {'key': 'value'})
    assert var_13 is True
    var_14 = 'key'
    var_15 = [var_9, var_14]
    var_16 = var_1.lookup_key(var_15)
    var_17 = var_16.value
    assert var_17 == 'value'

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
    var_0 = 'invalid: yaml: content: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_9 = 12
    var_10 = 11
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
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: b: c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'mapping values are not allowed here'



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int_token. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list_token. Retrieved 18/19 statements.
# Partially parsed test_tokenize_yaml_dict_token. Retrieved 22/23 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 19/20 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml_raises_error. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 8/9 statements.


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
    var_18 = [var_13, var_14]
    var_19 = var_1.lookup_key(var_18)
    var_20 = var_19.value
    assert var_20 == 'c'

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello\nworld'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello\nworld'
    var_3 = var_1.string
    assert var_3 == 'hello\nworld'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 5
    var_11 = 10
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

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



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_bytes_content. Retrieved 8/9 statements.


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
    var_2 = var_1.value
    assert var_2 == 'foo'
    var_3 = var_1.string
    assert var_3 == 'foo'
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
    var_0 = '- foo\n- bar'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['foo', 'bar'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- foo\n- bar'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 4
    var_12 = 9
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'foo: bar\nbaz: qux'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'foo': 'bar', 'baz': 'qux'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'foo: bar\nbaz: qux'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 7
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'foo:\n  - bar\n  - baz: qux'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'foo': ['bar', {'baz': 'qux'}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'foo:\n  - bar\n  - baz: qux'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = 20
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'foo:\n  - bar\n  - baz: qux'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'bar'
    var_6 = 1
    var_7 = 'baz'
    var_8 = [var_6, var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 'qux'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'foo:\n  bar: baz'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'bar'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'bar'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'foo: bar: baz'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'foo'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'foo'
    var_3 = var_1.string
    assert var_3 == 'foo'
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



# Parsed testcases at query #15
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
    var_8 = bool(False)
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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other: value'
    var_1 = 'key'
    var_2 = 'other'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].code
    assert var_13 == 'required'

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
    var_8 = bool(False)
    assert var_8 is True

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
    var_1 = 'inner'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'outer'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = module_2.validate_yaml(var_0, var_10)
    var_12 = bool(var_11 == {'outer': {'inner': 'value'}})
    assert var_12 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = str(var_3)
    assert var_4 == "'pyyaml' must be installed."



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_yaml_list_input. Retrieved 3/7 statements.


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

import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = module_1.validate_yaml(var_0, var_3)
    var_5 = 'unclosed'
    var_6 = bool('unclosed' in exc.text.lower())
    assert var_6 is True

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

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other: value'
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

import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = module_1.validate_yaml(var_0, var_3)

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'items:\n  - one\n  - two'
    var_1 = 'items'
    var_2 = {}
    var_3 = module_0.String(**var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'value: 123'
    var_1 = 'value'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = {var_1: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'value': 123})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'value: null'
    var_1 = 'value'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'value': None})
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_with_invalid_yaml. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_with_special_types. Retrieved 14/19 statements.


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
    var_8 = var_7.value
    assert var_8 == 'item1'
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 'item2'

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
    var_7 = 'float'
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = 'bool'
    var_11 = [var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = 'null'
    var_14 = [var_13]
    var_15 = var_1.lookup(var_14)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1: value1\nline2: value2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'line1'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.start.line_no
    assert var_5 == 1
    var_6 = [var_2]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.end.line_no
    assert var_8 == 1
    var_9 = 'line2'
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.start.line_no
    assert var_12 == 2
    var_13 = [var_9]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.end.line_no
    assert var_15 == 2

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = module_0.tokenize_yaml(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = None



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



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




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

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
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(var_10 == {'name': 'John', 'age': 30})
    assert var_11 is True

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
    var_6 = 'name: [John'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True

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
    var_6 = ''
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

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
    var_9 = 'name: John'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

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
    var_9 = 'name: John\nage: thirty'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'name: null'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {'name': None})
    assert var_10 is True

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
    var_6 = 'name: null'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name'
    var_10 = 'address'
    var_11 = {}
    var_12 = module_0.String(**var_11)
    var_13 = {var_9: var_12, var_10: var_8}
    var_14 = {}
    var_15 = module_1.Schema(var_13, **var_14)
    var_16 = 'name: John\naddress:\n  street: 123 Main St\n  city: Anytown'
    var_17 = module_2.validate_yaml(var_16, var_15)
    var_18 = bool(var_17 == {'name': 'John', 'address': {'street': '123 Main St', 'city': 'Anytown'}})
    assert var_18 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_line_84_evaluates_to_false. Retrieved 2/4 statements.


import yaml.parser as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ParserError(var_0)
    var_2 = var_1.problem_mark
    assert var_2 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 10/11 statements.


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
    var_0 = 'a:\n  b: [1, 2]'
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
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_0 = 'a: 1\nb: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.start.line_no
    assert var_7 == 1
    var_8 = 'b'
    var_9 = [var_8]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.start.line_no
    assert var_11 == 2



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 23/24 statements.
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
    var_0 = 'a: 1\nb: 2\nc: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2, 'c': 3})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a: 1\nb: 2\nc: 3'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 11
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  b:\n    - c\n    - d\ne: f'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': ['c', 'd']}, 'e': 'f'})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 3
    var_11 = 19
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    var_19 = bool(var_18 == {'b': ['c', 'd']})
    assert var_19 is True
    var_20 = 'b'
    var_21 = [var_15, var_20]
    var_22 = var_1.lookup(var_21)
    var_23 = var_22.value
    var_24 = bool(var_23 == ['c', 'd'])
    assert var_24 is True
    var_25 = [var_15, var_20, var_5]
    var_26 = var_1.lookup(var_25)
    var_27 = var_26.value
    assert var_27 == 'c'
    var_28 = [var_15, var_20]
    var_29 = var_1.lookup_key(var_28)
    var_30 = var_29.value
    assert var_30 == 'b'

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
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_yaml_not_installed. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_tokenize_yaml_with_scalar_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_with_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_boolean. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_multiline_content. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

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
    var_9 = 8
    var_10 = 7
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
    var_0 = 'a: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #30
#--------------------------




import yaml.parser as module_0

def test_case_0():
    var_0 = 'problem'
    var_1 = 'problem_mark'
    var_2 = None
    var_3 = module_0.ParserError(var_0, var_1, var_2, var_2, var_2)
    var_4 = var_3.problem
    assert var_4 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_yaml_list_input. Retrieved 3/7 statements.


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
    var_8 = bool(False)
    assert var_8 is True

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
    var_8 = bool(False)
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
    var_8 = bool(False)
    assert var_8 is True
    var_9 = e.messages()[0].code
    assert var_9 == 'type'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'other_key: value'
    var_1 = 'key'
    var_2 = 'other_key'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = e.messages()[0].code
    assert var_12 == 'required'
    var_13 = e.messages()[0].index
    var_14 = bool(e.messages()[0].index == ['key'])
    assert var_14 is True

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

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = 'optional'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = 'default'
    var_6 = 'default'
    var_7 = {var_6: var_5}
    var_8 = module_0.String(**var_7)
    var_9 = {var_1: var_4, var_2: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = module_2.validate_yaml(var_0, var_11)
    var_13 = bool(var_12 == {'key': 'value', 'optional': 'default'})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'key': None})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = e.messages()[0].code
    assert var_9 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key:\n  - item1\n  - item2'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_0.String(**var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_tokenize_yaml_without_pyyaml. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = 'key: value'



