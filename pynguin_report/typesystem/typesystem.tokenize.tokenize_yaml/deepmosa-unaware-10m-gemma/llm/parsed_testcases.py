####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.value[var_12]
    var_14 = 'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'key'
    var_17 = var_15.value[var_16]
    var_18 = b'name: test'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '   \n  '
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = ': invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '\n    root:\n      list:\n        - 1\n        - 2\n      map:\n        a: true\n    '
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'root'
    var_29 = var_27.value[var_28]
    var_30 = 'list'
    var_31 = var_29.value[var_30]



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '123'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '12.34'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value\nnum: 1'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'foo: bar'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: : value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: value\n  invalid_indentation'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '\n    nested:\n      list:\n        - 1\n        - 2\n      dict:\n        a: b\n    '
    var_25 = module_0.tokenize_yaml(var_24)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = '42'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = '- apple\n- banana'
    var_6 = module_0.String()
    var_7 = 'name: John\nage: not_a_number'
    var_8 = 'age'
    var_9 = 'name: : invalid'
    var_10 = ''
    var_11 = module_1.validate_yaml(var_10, var_0)
    var_12 = b'name: Jane\nage: 25'
    var_13 = module_0.Integer()
    var_14 = '1.5'



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = '- one\n- two'
    var_5 = module_0.String()
    var_6 = [var_5]
    var_7 = 'name: John\nage: not_an_int'
    var_8 = 'name: John\n  age: : : 30'
    var_9 = ''
    var_10 = 'Hello World'
    var_11 = module_0.String()
    var_12 = '\n    user:\n      name: Alice\n      tags:\n        - admin\n        - editor\n    '



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'items'
    var_5 = module_0.String()
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'items: [a, b, c]'
    var_10 = 'name: John\nage: not_an_int'
    var_11 = 'age'
    var_12 = 'name: John\nage: : : :'
    var_13 = ''
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = module_0.Integer()
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = [var_19]
    var_21 = {var_14: var_17, var_15: var_18, var_16: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = 'a: 1\nb: true\nc: [2, 3]'
    var_24 = 'name: John\nage: bad'
    var_25 = 'key'
    var_26 = module_0.String()
    var_27 = {var_25: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'key: null'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'name: John\nage: 30'
    var_7 = 'tags'
    var_8 = module_0.String()
    var_9 = {var_7: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'tags: [python, pytest]'
    var_12 = module_0.String()
    var_13 = 'name: : unexpected'
    var_14 = module_2.validate_yaml(var_13, var_5)
    var_15 = ''
    var_16 = module_2.validate_yaml(var_15, var_5)
    var_17 = 'name: John\nage: not_an_int'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = b'key: value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'user'
    var_1 = 'items'
    var_2 = 'id'
    var_3 = 'active'
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.Schema(var_6)
    var_8 = module_0.Integer()
    var_9 = '\n    user:\n      id: 123\n      active: "true"\n    items:\n      - 1\n      - 2\n    '



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = module_0.Integer()
    var_3 = '123'
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = '\n    name: John Doe\n    age: 30\n    '
    var_7 = 'items'
    var_8 = 'items: [a, b, c]'
    var_9 = module_0.Integer()
    var_10 = 'not_an_int'
    var_11 = module_0.String()
    var_12 = 'age: 30'
    var_13 = 'name'
    var_14 = 'key: : value'
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = '   '
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = b'name: ByteTest'
    var_21 = module_0.String()
    var_22 = {var_13: var_21}
    var_23 = module_2.Schema(var_22)
    var_24 = '\n    meta:\n      id: 1\n    tags:\n      - python\n      - testing\n    '
    var_25 = 'meta'
    var_26 = 'tags'
    var_27 = 'id'
    var_28 = module_0.Integer()
    var_29 = {var_27: var_28}
    var_30 = module_2.Schema(var_29)
    var_31 = module_0.String()



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = '\n    name: John Doe\n    age: 30\n    '
    var_3 = b'name: Jane Doe\nage: 25'
    var_4 = '\n    data:\n      - item1\n      - item2\n    '
    var_5 = 'items'
    var_6 = module_0.String()
    var_7 = 'items: [a, b, c]'
    var_8 = '\n    name: John Doe\n    age: not_an_integer\n    '
    var_9 = '\n    name: "unclosed quote\n    age: 30\n    '
    var_10 = ''
    var_11 = '   \n   '
    var_12 = '\n    int_val: 10\n    float_val: 10.5\n    bool_val: true\n    null_val: null\n    '



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: : John'
    var_5 = 'name: John\nage: not_a_number'
    var_6 = ''
    var_7 = module_0.String()
    var_8 = '- items: hello\n- items: world'
    var_9 = 'items'
    var_10 = module_0.String()
    var_11 = {var_9: var_10}
    var_12 = module_1.Schema(var_11)
    var_13 = 'is_active: true\nscore: 95.5'
    var_14 = module_0.String()
    var_15 = module_0.String()
    var_16 = 'is_active'
    var_17 = 'score'
    var_18 = module_0.String()
    var_19 = module_0.String()
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_1.Schema(var_20)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = '\n    name: "John Doe"\n    age: 30\n    is_active: true\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'is_active'
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = module_0.Boolean()
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = '\n    users:\n      - id: 1\n        tags: [admin, editor]\n    '
    var_10 = 'users'
    var_11 = 'id'
    var_12 = 'tags'
    var_13 = module_0.Integer()
    var_14 = module_0.String()
    var_15 = '\n    key: : value\n    '
    var_16 = 'key'
    var_17 = module_0.String()
    var_18 = {var_16: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = module_2.validate_yaml(var_15, var_19)
    var_21 = '   '
    var_22 = module_2.validate_yaml(var_21, var_19)
    var_23 = '\n    name: 123\n    age: "not an integer"\n    is_active: true\n    '
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = module_0.Boolean()
    var_27 = {var_22: var_24, var_2: var_25, var_3: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = b'key: value'
    var_30 = module_0.String()
    var_31 = {var_16: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = 'data: null'
    var_34 = 'data'
    var_35 = module_0.String()
    var_36 = {var_34: var_35}
    var_37 = module_1.Schema(var_36)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = module_0.String()
    var_5 = 'tags: [python, testing, yaml]'
    var_6 = 'name: John\nage: not_an_int'
    var_7 = 'age'
    var_8 = 'name: John\nage: : : :'
    var_9 = ''
    var_10 = '\n    user:\n      id: 123\n      meta:\n        active: true\n    '



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = '- id: 1\n- id: 2'
    var_5 = 'items'
    var_6 = 'name: John\nage: not_an_int'
    var_7 = 'age'
    var_8 = 'name: John\nage: : : :'
    var_9 = '   '
    var_10 = '\n    user:\n      name: Alice\n      roles:\n        - admin\n        - editor\n    '
    var_11 = 'user'
    var_12 = 'name'
    var_13 = module_0.String()
    var_14 = {var_12: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = {var_11: var_15}
    var_17 = module_1.Schema(var_16)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'int_val: 10\nfloat_val: 10.5\nbool_val: true\nnull_val: null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'int_val'
    var_3 = var_1.value[var_2]



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: John'
    var_5 = 'age'
    var_6 = 'tags'
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'age: 30\ntags: python'
    var_12 = 'user'
    var_13 = 'id'
    var_14 = module_0.Integer()
    var_15 = {var_13: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = {var_12: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'user:\n  id: 123'
    var_20 = module_0.Integer()
    var_21 = {var_5: var_20}
    var_22 = module_1.Schema(var_21)
    var_23 = 'age: not_an_int'
    var_24 = 'key: [unclosed_bracket'
    var_25 = module_2.validate_yaml(var_24, var_22)
    var_26 = ''
    var_27 = module_2.validate_yaml(var_26, var_22)
    var_28 = b'age: 25'
    var_29 = module_0.Integer()
    var_30 = {var_5: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'items'
    var_33 = module_0.String()
    var_34 = {var_32: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = 'items:\n  - one\n  - two'
    var_37 = module_0.Integer()
    var_38 = {var_32: var_37}
    var_39 = module_1.Schema(var_38)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John Doe\nage: 30'
    var_3 = '- item1\n- item2'
    var_4 = module_0.String()
    var_5 = 'name: : John'
    var_6 = ''
    var_7 = 'name: John Doe\nage: not_a_number'
    var_8 = 'age'
    var_9 = b'name: Jane\nage: 25'
    var_10 = '\n    user:\n      name: Alice\n      details:\n        active: true\n    '
    var_11 = 'user'
    var_12 = 'name'
    var_13 = module_0.String()
    var_14 = {var_12: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = {var_11: var_15}
    var_17 = module_1.Schema(var_16)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = module_0.Integer()
    var_3 = '123'
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = '\n    name: John\n    age: 30\n    '
    var_7 = module_0.String()
    var_8 = '- apple\n- banana'
    var_9 = module_0.Integer()
    var_10 = 'not_an_int'
    var_11 = 'key: : value'
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = '   '
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = b'name: Bob\nage: 25'
    var_18 = '\n    items:\n      - one\n      - two\n    meta: info\n    '



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'name: John\nage: 30'
    var_7 = 'tags'
    var_8 = module_0.String()
    var_9 = module_0.Field()
    var_10 = {var_7: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = 'tags:\n  - python\n  - pytest'
    var_13 = module_0.Integer()
    var_14 = {var_1: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = 'age: not_an_integer'
    var_17 = 'key: : value'
    var_18 = module_2.validate_yaml(var_17, var_5)
    var_19 = '   '
    var_20 = module_2.validate_yaml(var_19, var_5)
    var_21 = b'name: Jane\nage: 25'
    var_22 = 'user'
    var_23 = 'id'
    var_24 = 'active'
    var_25 = module_0.Integer()
    var_26 = module_0.String()
    var_27 = module_0.Field()
    var_28 = {var_23: var_25, var_24: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = {var_22: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = 'user:\n  id: 1\n  active: true'
    var_33 = module_0.Integer()
    var_34 = module_0.Field()
    var_35 = {var_22: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = 'active: true'
    var_38 = None
    var_39 = module_0.Field()
    var_40 = {var_24: var_39}
    var_41 = module_1.Schema(var_40)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '\n    name: "John Doe"\n    age: "not_an_int"\n    active: "true"\n    '
    var_5 = 'age'
    var_6 = '\n    name: "John Doe"\n    age: : : :\n    '
    var_7 = ''
    var_8 = b"name: 'Byte Test'\nage: 20\nactive: 'false'"
    var_9 = module_0.String()
    var_10 = '\n    items:\n      - "apple"\n      - "banana"\n    '
    var_11 = '\n    user:\n      name: "Alice"\n      details:\n        id: 123\n    '



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '123'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '12.34'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 0
    var_17 = var_13.value[var_16]
    var_18 = 'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'value'
    var_21 = 4
    var_22 = 9
    var_23 = module_1.ScalarToken(var_20, var_21, var_22, var_18)
    var_24 = 'parent:\n  child: 1'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'parent'
    var_27 = var_25.value[var_26]
    var_28 = b'foo: bar'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'key: : value'
    var_31 = module_0.tokenize_yaml(var_30)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = str(var_0)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 0
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 7
    var_4 = module_0._get_position(var_0, var_3)
    var_5 = 8
    var_6 = module_0._get_position(var_0, var_5)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '123'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '45.67'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value\nnum: 10'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'foo: bar'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: : value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'line1\nline2: : error'
    var_23 = module_0.tokenize_yaml(var_22)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = '\n    name: John Doe\n    age: not_an_int\n    active: true\n    '
    var_5 = 'age'
    var_6 = '\n    name: [unclosed_bracket\n    age: 30\n    '
    var_7 = ''
    var_8 = b'name: Byte User\nage: 25\nactive: false'
    var_9 = '\n    users:\n      - name: Alice\n        age: 20\n      - name: Bob\n        age: 40\n    '
    var_10 = '- item1\n- item2'
    var_11 = module_0.String()
    var_12 = 'name: Alice\nage: 25'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '45.67'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key: value\nfoo: bar'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = b'name: test'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '   \n  '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '{unclosed_bracket'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: value\n  invalid: : mapping'
    var_23 = module_0.tokenize_yaml(var_22)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: : : 30'
    var_2 = ''
    var_3 = 'name: John\nage: not_an_int'
    var_4 = b'name: Jane\nage: 25'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '\n    list:\n      - item1\n      - item2\n    dict:\n      key: value\n    '
    var_1 = module_0.tokenize_yaml(var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '123'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '12.34'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    nested:\n      list:\n        - 1\n        - 2\n      val: true\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'foo: bar'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: : value'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'line1\nline2\nline3'
    var_25 = 7
    var_26 = module_0._get_position(var_24, var_25)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '\n    count: 10\n    ratio: 0.5\n    is_valid: true\n    '
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = module_0.String()
    var_8 = 'ratio'
    var_9 = '\n    name: "John\n    age: 30\n    '
    var_10 = '   '
    var_11 = '\n    name: 123\n    age: "not_an_int"\n    '
    var_12 = b"name: 'Byte User'\nage: 25\nactive: 'true'"
    var_13 = '- item1\n- item2'
    var_14 = module_0.String()
    var_15 = 'value'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = '\n    name: John Doe\n    age: [unclosed list\n    '
    var_5 = ''
    var_6 = '\n    name: John Doe\n    age: not_a_number\n    active: true\n    '
    var_7 = 'age'
    var_8 = b'name: Byte User\nage: 25\nactive: false'
    var_9 = '\n    - item1\n    - item2\n    '
    var_10 = module_1.tokenize_yaml(var_9)
    var_11 = 'key: value'
    var_12 = module_1.tokenize_yaml(var_11)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '123'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '45.67'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value\nfoo: bar'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'parent:\n  child: 123'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'parent'
    var_21 = var_19.value[var_20]
    var_22 = b'name: test'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: : value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: value\n  invalid: [unclosed list'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'key: : value'
    var_29 = module_0.tokenize_yaml(var_28)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John Doe\nage: 30'
    var_3 = 'name'
    var_4 = b'name: Jane Doe\nage: 25'
    var_5 = 'name: John\n  age: : 30'
    var_6 = ''
    var_7 = 'name: John\nage: not_a_number'
    var_8 = module_0.String()
    var_9 = '- apple\n- banana'
    var_10 = module_1.tokenize_yaml(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = '\n    user:\n      profile:\n        id: 123\n        active: true\n    '
    var_14 = 'user: {id: 1}'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: : John'
    var_3 = ''
    var_4 = 'name: John\nage: not_an_int'
    var_5 = 'age'
    var_6 = '- item1\n- item2'
    var_7 = 'items'
    var_8 = module_0.String()
    var_9 = '- item1\n- item2'
    var_10 = module_0.String()
    var_11 = '\n    user:\n      name: Alice\n      roles:\n        - admin\n        - editor\n    '

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 12
    var_4 = module_0._get_position(var_0, var_3)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: not_an_int'
    var_2 = 'name: John'
    var_3 = 2
    var_4 = 'name: John\n  age: : 30'
    var_5 = ''
    var_6 = b'name: Jane\nage: 25'
    var_7 = module_0.String()
    var_8 = 'items: [apple, banana]'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 12
    var_4 = module_0._get_position(var_0, var_3)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: Jane\nage: 25\nactive: true'
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = 'active: true'
    var_8 = module_0.String()
    var_9 = 'name: : John'
    var_10 = ''
    var_11 = 'name: John\nage: not_an_int'
    var_12 = 'age'
    var_13 = b'name: Bob\nage: 40'
    var_14 = '- item1\n- item2'
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = 'key: value'
    var_18 = module_0.String()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    name: "John Doe"\n    age: 30\n    active: true\n    '
    var_1 = b"name: 'Jane'\nage: 25\nactive: false"
    var_2 = '\n    name: 123\n    age: "not_an_int"\n    active: "maybe"\n    '
    var_3 = []
    var_4 = 'name'
    var_5 = 'age'
    var_6 = '\n    name: "John\n    age: 30\n    '
    var_7 = '   '
    var_8 = '- item1\n- item2'
    var_9 = '\n    user:\n      name: Alice\n      meta:\n        id: 1\n    '



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\nage: not_an_int'
    var_5 = 'name: John\n  age: : : : '
    var_6 = ''
    var_7 = module_0.String()
    var_8 = '- apple\n- banana'
    var_9 = '\n    user:\n      name: Alice\n      roles:\n        - admin\n        - editor\n    '



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: John Doe'
    var_5 = 'users'
    var_6 = 'id'
    var_7 = 'active'
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = {var_5: var_11}
    var_13 = module_1.Schema(var_12)
    var_14 = 'users:\n  id: 123\n  active: true'
    var_15 = module_0.Integer()
    var_16 = module_0.Field()
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = {var_5: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 'age'
    var_22 = module_0.Integer()
    var_23 = {var_21: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = 'age: not_a_number'
    var_26 = ''
    var_27 = module_2.validate_yaml(var_26, var_24)
    var_28 = 'key: : value'
    var_29 = module_2.validate_yaml(var_28, var_24)
    var_30 = b'name: Jane Doe'
    var_31 = module_0.String()
    var_32 = {var_29: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'items'
    var_35 = module_0.Integer()
    var_36 = [var_35]
    var_37 = {var_34: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'items:\n  - 1\n  - 2\n  - 3'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 7
    var_4 = module_0._get_position(var_0, var_3)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: not_an_int'
    var_2 = 'age'
    var_3 = 'name: : John'
    var_4 = ''
    var_5 = b'name: Jane\nage: 25'
    var_6 = 'items'
    var_7 = module_0.String()
    var_8 = {var_6: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = 'items: [a, b, c]'
    var_11 = '\n    user:\n      name: Alice\n      roles:\n        - admin\n        - editor\n    '
    var_12 = 'user'
    var_13 = 'name'
    var_14 = 'roles'
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = [var_16]
    var_18 = module_1.Schema(var_17)
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = {var_12: var_20}
    var_22 = module_1.Schema(var_21)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\nage: not_an_int'
    var_3 = []
    var_4 = 'age'
    var_5 = 'name: John\n  age: 30\n    broken: :'
    var_6 = ''
    var_7 = module_0.String()
    var_8 = 'items: hello'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 0
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 7
    var_4 = module_0._get_position(var_0, var_3)
    var_5 = 6
    var_6 = module_0._get_position(var_0, var_5)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: John Doe'
    var_5 = 'id'
    var_6 = 'tags'
    var_7 = module_0.Integer()
    var_8 = 'active'
    var_9 = module_0.String()
    var_10 = {var_8: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = {var_5: var_7, var_6: var_11}
    var_13 = module_1.Schema(var_12)
    var_14 = 'id: 123\ntags:\n  active: true'
    var_15 = 'age'
    var_16 = module_0.Integer()
    var_17 = {var_15: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'age: not_an_int'
    var_20 = 'key: [unclosed_bracket'
    var_21 = module_2.validate_yaml(var_20, var_3)
    var_22 = ''
    var_23 = module_2.validate_yaml(var_22, var_3)
    var_24 = b'name: ByteContent'
    var_25 = 'items'
    var_26 = module_0.Integer()
    var_27 = module_1.Schema(var_26)
    var_28 = {var_25: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 'items: [1, 2, 3]'
    var_31 = module_0.Integer()
    var_32 = {var_25: var_31}
    var_33 = module_1.Schema(var_32)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: not_an_int'
    var_2 = 'age'
    var_3 = 'name: : John'
    var_4 = ''
    var_5 = b'name: Jane\nage: 25'
    var_6 = 'items'
    var_7 = module_0.String()
    var_8 = {var_6: var_7}
    var_9 = module_1.Schema(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_2.tokenize_yaml(var_10)

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 0
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 7
    var_4 = module_0._get_position(var_0, var_3)
    var_5 = 6
    var_6 = module_0._get_position(var_0, var_5)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'name: John\nage: 30'
    var_7 = 'items'
    var_8 = '- apple\n- banana'
    var_9 = module_0.String()
    var_10 = 'items:\n  - apple\n  - banana'
    var_11 = module_0.String()
    var_12 = 'name: : John'
    var_13 = module_2.validate_yaml(var_12, var_5)
    var_14 = ''
    var_15 = module_2.validate_yaml(var_14, var_5)
    var_16 = 'name: John\nage: not_an_int'
    var_17 = b'name: John\nage: 30'
    var_18 = '\n    user:\n      id: 123\n      active: true\n      tags:\n        - admin\n        - editor\n    '
    var_19 = 'user'
    var_20 = 'id'
    var_21 = 'active'
    var_22 = 'tags'
    var_23 = module_0.Integer()
    var_24 = module_0.String()
    var_25 = module_0.String()
    var_26 = module_0.Integer()
    var_27 = module_0.Boolean()
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = module_0.Boolean()
    var_31 = {var_20: var_29, var_21: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = {var_19: var_32}
    var_34 = module_1.Schema(var_33)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '123'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '12.34'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'key: value\nnum: 42'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'name: tester'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '\n    root:\n      list: [1, 2, 3]\n      nested:\n        bool: true\n    '
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: : value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: value\n  invalid_indentation'
    var_27 = module_0.tokenize_yaml(var_26)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: 123\nage: 30'
    var_2 = 'age: 30'
    var_3 = 'name:John\nage:30'
    var_4 = 'name:\n  unindent: error\n    bad: structure'
    var_5 = 'name: : value'
    var_6 = ''
    var_7 = b'name: Jane\nage: 25'
    var_8 = module_0.String()
    var_9 = 'items: [a, b, c]'

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)
    var_3 = 12
    var_4 = module_0._get_position(var_0, var_3)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = '- item1\n- item2'
    var_2 = module_0.String()
    var_3 = 'name: John\nage: not_an_int'
    var_4 = 'age'
    var_5 = 'name: : unexpected'
    var_6 = ''
    var_7 = b'name: Jane\nage: 25'
    var_8 = 'name: John'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\nage: not_a_number'
    var_3 = 'age'
    var_4 = 'name: John'
    var_5 = 'name: John\n  age: : : 30'
    var_6 = ''
    var_7 = '\n- name: Alice\n  age: 20\n- name: Bob\n  age: 40\n'



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'name: John'
    var_5 = 'users'
    var_6 = 'id'
    var_7 = 'active'
    var_8 = module_0.Integer()
    var_9 = module_0.Boolean()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_1.Schema(var_10)
    var_12 = '\n    users:\n      - id: 1\n        active: true\n      - id: 2\n        active: false\n    '
    var_13 = 'age'
    var_14 = module_0.Integer()
    var_15 = {var_13: var_14}
    var_16 = module_1.Schema(var_15)
    var_17 = 'age: not_a_number'
    var_18 = 'required_field'
    var_19 = module_0.String()
    var_20 = {var_18: var_19}
    var_21 = module_1.Schema(var_20)
    var_22 = 'other_field: exists'
    var_23 = 'key: : invalid_yaml'
    var_24 = module_2.validate_yaml(var_23, var_3)
    var_25 = '   '
    var_26 = module_2.validate_yaml(var_25, var_3)
    var_27 = b'name: ByteContent'
    var_28 = 'f'
    var_29 = 'n'
    var_30 = module_0.Float()
    var_31 = 'f: 12.34\nn: null'



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '123'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '12.34'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '- item1\n- item2'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'key: value\nfoo: bar'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = b'name: tester'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '\n    list:\n      - subkey: subval\n    num: 42\n    '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = ': broken'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'key: [unclosed_bracket'
    var_29 = module_0.tokenize_yaml(var_28)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: : John'
    var_5 = '   '
    var_6 = 'name: John\nage: not_a_number'
    var_7 = 'age'
    var_8 = 'integer'
    var_9 = '\n    users:\n      - name: Alice\n        age: 20\n      - name: Bob\n        age: 40\n    '
    var_10 = 'users'
    var_11 = module_0.Integer()
    var_12 = '123'
    var_13 = 'abc'



