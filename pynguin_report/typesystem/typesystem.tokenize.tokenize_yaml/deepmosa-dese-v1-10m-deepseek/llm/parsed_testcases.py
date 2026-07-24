####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_list. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
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
    var_6 = 12
    var_7 = module_1.Position(var_5, var_5, var_6)

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
    var_7 = 14
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- - 1\n  - 2\n- - 3\n  - 4'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 20
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: b: c'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_list. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '{a: {b: 2}}'
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
    var_0 = '[1, [2, 3]]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 12
    var_6 = 11
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '{a: 1, b: 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0.tokenize_yaml(var_1)
    assert var_2 is None



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #5
#--------------------------




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
    var_0 = 'name: John\nage:'
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
    var_0 = 'name: John\nage: thirty'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)

import typesystem.schemas as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = module_1.validate_yaml(var_0, var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'name: John\nage: 30'
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

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John'
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
    var_0 = 'name: John'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = 30
    var_5 = module_0.Field(default=var_4)
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value:'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: null'
    var_1 = 'key'
    var_2 = True
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)



# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John\nage: 30'
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
    var_0 = 'name: John\n: 30'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John\nage: thirty'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = b'name: John\nage: 30'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #11
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 7
    var_7 = 14
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key1: value1\nkey2: value2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 10
    var_7 = 23
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: 123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 8
    var_6 = 7
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: 123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: true'
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
    var_0 = 'key: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 9
    var_6 = 8
    var_7 = module_1.Position(var_2, var_5, var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_yaml_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'yaml'



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

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

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  - 1\n  - 2\nb: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = 15
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_yaml_with_validation_error. Retrieved 6/7 statements.
# Partially parsed test_validate_yaml_with_required_field_error. Retrieved 6/7 statements.
# Partially parsed test_validate_yaml_with_positional_validation_error. Retrieved 6/7 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)

import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'key: value\ninvalid'
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
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '{}'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 5
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 13
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #20
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\nkey2: value2\n\tkey3: value3'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 5
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 13
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_yaml_with_missing_required_field. Retrieved 8/9 statements.
# Partially parsed test_validate_yaml_with_invalid_data_type. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
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
    var_0 = 'name: John Doe'
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
    var_0 = 'name: John Doe\nage: thirty'
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
    var_0 = ''
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
    var_0 = 'name: John Doe\nage:'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)



# Parsed testcases at query #23
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
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
    var_6 = 12
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  - b: 1\n    c: 2\nd: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = 22
    var_8 = module_1.Position(var_5, var_6, var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_yaml_without_pyyaml_installed. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: value\ninvalid'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'key: 123'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = ''
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = b'key: value'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '{}'
    var_1 = 'key'
    var_2 = module_0.Field()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.validate_yaml(var_0, var_4)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #29
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #30
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_yaml_does_not_raise_when_pyyaml_is_installed. Retrieved 5/13 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = 'yaml'



# Parsed testcases at query #32
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list_token. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict_token. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\ninvalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_yaml_with_missing_required_field. Retrieved 11/12 statements.
# Partially parsed test_validate_yaml_with_invalid_yaml. Retrieved 9/10 statements.
# Partially parsed test_validate_yaml_with_null_value_and_disallow_null. Retrieved 10/11 statements.


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
    var_0 = 'name: John'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = module_2.validate_yaml(var_0, var_6)
    var_9 = var_7.messages
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John\nage:'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = None
    var_8 = module_2.validate_yaml(var_0, var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: null'
    var_1 = 'name'
    var_2 = True
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: null'
    var_1 = 'name'
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = None
    var_7 = module_2.validate_yaml(var_0, var_5)
    var_8 = var_6.messages
    var_9 = len(var_8)
    assert var_9 == 1



# Parsed testcases at query #3
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #4
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'key: value'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 10
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 14
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\ninvalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_yaml_pyyaml_not_installed. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 13
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_yaml_without_pyyaml_installed. Retrieved 4/9 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 14
    var_8 = module_1.Position(var_5, var_6, var_7)



# Parsed testcases at query #14
#--------------------------




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
    var_0 = 'name: John\nage: thirty'
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
    var_0 = ''
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
    var_0 = 'name: John'
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
    var_0 = 'name: John\nage: thirty'
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



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 5
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_with_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_with_nested_structure. Retrieved 9/10 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 5
    var_7 = 14
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)



# Parsed testcases at query #21
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_83_evaluates_to_false. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid_yaml_content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_yaml_invalid_input. Retrieved 8/9 statements.
# Partially parsed test_validate_yaml_missing_required_field. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John\nage: 25'
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
    var_0 = 'name: John\nage: twenty-five'
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
    var_0 = 'name: John'
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
    var_0 = ''
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
    var_0 = 'name: John\nage: 25\ninvalid'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_raises_assertion_error_when_pyyaml_not_installed. Retrieved 3/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 8/9 statements.


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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 3
    var_7 = 6
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_yaml_with_pyyaml_installed. Retrieved 5/14 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = 'yaml'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2\n- 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 3
    var_6 = 8
    var_7 = module_1.Position(var_5, var_5, var_6)

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

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'a:\n  - 1\n  - 2\nb: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = 15
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 5
    var_6 = 4
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'a: b: c'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 11/12 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.


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
    var_0 = '123'
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
    var_0 = '123.45'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = 2
    var_8 = 3
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: [1, 2'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_yaml_with_invalid_content. Retrieved 6/7 statements.
# Partially parsed test_validate_yaml_with_empty_content. Retrieved 6/7 statements.
# Partially parsed test_validate_yaml_with_invalid_yaml. Retrieved 6/7 statements.
# Partially parsed test_validate_yaml_with_missing_required_field. Retrieved 6/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: John'
    var_5 = module_2.validate_yaml(var_4, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: 123'
    var_5 = module_2.validate_yaml(var_4, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = ''
    var_5 = module_2.validate_yaml(var_4, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: {'
    var_5 = module_2.validate_yaml(var_4, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'age: 30'
    var_5 = module_2.validate_yaml(var_4, var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'name: null'
    var_6 = module_2.validate_yaml(var_5, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'value'
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
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 6
    var_7 = 13
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 10
    var_6 = 9
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  nested_key: nested_value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 20
    var_7 = 28
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\ninvalid'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_tokenize_yaml_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_int. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_bool. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_nested_list. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 6
    var_6 = 5
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
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
    var_0 = '3.14'
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
    var_0 = 'true'
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
    var_0 = 'null'
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
    var_0 = '- 1\n- 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 4
    var_7 = 6
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 11
    var_6 = 10
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key:\n  nested: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 15
    var_7 = 18
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '- - 1\n- - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 2
    var_6 = 6
    var_7 = 9
    var_8 = module_1.Position(var_5, var_6, var_7)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: : value'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #33
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content'
    var_1 = module_0.tokenize_yaml(var_0)



