####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_with_string_content. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_bytes_content. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_integer. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_float. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_boolean_true. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_boolean_false. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_null. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_nested_dict. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_list_of_dicts. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_multiline_string. Retrieved 2/5 statements.
# Partially parsed test_tokenize_yaml_with_special_characters. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['item1', 'item2'])
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 123

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
    var_0 = '"hello world"'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello world'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'outer': {'inner': 'value'}})
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '- key: value1\n- key: value2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [{'key': 'value1'}, {'key': 'value2'}])
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'line1'
    var_3 = bool('line1' in var_1.value)
    assert var_3 is True
    var_4 = 'line2'
    var_5 = bool('line2' in var_1.value)
    assert var_5 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1._content
    var_3 = bool(var_1._content == var_0)
    assert var_3 is True

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = "special: '@#$%'"
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{ invalid: yaml: content }'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_yaml_with_validation_error. Retrieved 8/13 statements.
# Partially parsed test_validate_yaml_with_required_field_missing. Retrieved 8/13 statements.


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
    var_6 = b'name: Alice'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'name': 'Alice'})
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
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name: [invalid: yaml: content:'
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
    var_9 = 'name: John\nage: not_an_integer'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
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
    var_9 = 'user'
    var_10 = {var_9: var_8}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'user:\n  name: John\n  age: 30'
    var_14 = module_2.validate_yaml(var_13, var_12)
    var_15 = bool(var_14 == {'user': {'name': 'John', 'age': 30}})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'items'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {'items': ['apple', 'banana', 'cherry']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'status'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = 'active'
    var_5 = 'default'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'name: John'
    var_12 = module_2.validate_yaml(var_11, var_10)
    var_13 = bool(var_12 == {'name': 'John', 'status': 'active'})
    assert var_13 is True

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
    var_6 = '   \n\n  \t  '
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True



