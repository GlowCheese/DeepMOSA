####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '12.34'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = b'foo: bar'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '   \n  '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = ': value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'line1\nline2\nline3'
    var_23 = 6
    var_24 = module_0._get_position(var_22, var_23)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = '\n    name: John Doe\n    age: : : :\n    '
    var_5 = ''
    var_6 = '\n    name: John Doe\n    age: not_an_integer\n    active: true\n    '
    var_7 = 'age'
    var_8 = b'name: Byte User\nage: 25\nactive: false'
    var_9 = '\n    items:\n      - item1\n      - item2\n    metadata:\n      key: value\n    '



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = 'age'
    var_5 = b"name: 'Jane'\nage: 25\nactive: 'false'"
    var_6 = '\n    name: 123\n    age: "not_an_int"\n    active: "true"\n    '
    var_7 = 'name'
    var_8 = '\n    name: "John"\n    age: : : invalid\n    '
    var_9 = '   '
    var_10 = '\n    items:\n      - apple\n      - banana\n    '
    var_11 = '- apple\n- banana'
    var_12 = '\n    name: null\n    age: 40\n    active: "true"\n    '



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

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
    var_16 = 'utf-8'
    var_17 = module_1.encode(var_16)
    var_18 = len(var_17)
    var_19 = 1
    var_20 = var_18 - var_19
    var_21 = 'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = b'key: value'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = 'key: : value'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = 'line1\nline2\nline3'
    var_28 = 6
    var_29 = module_0._get_position(var_27, var_28)
    var_30 = 7
    var_31 = module_0._get_position(var_27, var_30)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = b'name: Jane Doe\nage: 25\nactive: false'
    var_5 = '\n    name: John\n    age: : : : \n    '
    var_6 = '   '
    var_7 = '\n    name: John\n    age: not_an_integer\n    active: true\n    '
    var_8 = 'age'
    var_9 = module_0.String()
    var_10 = '- item1\n- item2'
    var_11 = module_0.String()

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 6
    var_2 = module_0._get_position(var_0, var_1)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '- item1\n- item2'
    var_5 = module_0.String()
    var_6 = '\n    name: "John\n    age: 30\n    '
    var_7 = ''
    var_8 = '\n    name: "John Doe"\n    age: "not_an_integer"\n    active: "true"\n    '
    var_9 = 'age'
    var_10 = b"name: 'Byte Test'\nage: 25\nactive: 'false'"
    var_11 = '\n    user:\n      name: "Nested"\n      age: 40\n    active: "true"\n    '
    var_12 = module_0.String()



# Parsed testcases at query #7
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
    var_5 = 'age'
    var_6 = module_0.Integer()
    var_7 = {var_5: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'age: 30'
    var_10 = 'tags'
    var_11 = 'items'
    var_12 = '- python\n- testing'
    var_13 = module_0.String()
    var_14 = '- python\n- testing'
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = {var_5: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'age: not_an_int'
    var_20 = 'age: abc'
    var_21 = 'key: : value'
    var_22 = module_2.validate_yaml(var_21, var_3)
    var_23 = '   '
    var_24 = module_2.validate_yaml(var_23, var_3)
    var_25 = b'name: Jane Doe'
    var_26 = 'user'
    var_27 = 'roles'
    var_28 = 'id'
    var_29 = 'active'
    var_30 = module_0.Integer()
    var_31 = module_0.String()
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = module_0.String()
    var_35 = '\n    user:\n      id: 1\n      active: "true"\n    roles:\n      - admin\n      - editor\n    '



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '\n    age: 25\n    score: 95.5\n    is_valid: true\n    metadata: null\n    '
    var_5 = module_0.Integer()
    var_6 = module_0.Integer()
    var_7 = module_0.String()
    var_8 = module_0.String()
    var_9 = '\n    name: "John Doe"\n    age: "not_an_integer"\n    active: "true"\n    '
    var_10 = 'age'
    var_11 = '\n    name: "John Doe"\n    age: : : :\n    '
    var_12 = '   '
    var_13 = b"name: 'Byte Content'\nage: 10\nactive: 'yes'"
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = '- item1\n- item2'
    var_18 = module_0.String()
    var_19 = module_0.String()
    var_20 = '- 1\n- 2'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '\n    name: "John Doe"\n    age: : : :\n    '
    var_5 = '\n    name: "John Doe"\n    age: "not_an_int"\n    active: "true"\n    '
    var_6 = 'age'
    var_7 = ''
    var_8 = b"name: 'Byte Test'\nage: 20\nactive: 'false'"
    var_9 = module_0.String()
    var_10 = '\n    items:\n      - apple\n      - banana\n    '
    var_11 = module_1.Schema()



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = b'name: Jane\nage: 25\nactive: false'
    var_5 = '\n    name: John\n    age: : unexpected_colon\n    '
    var_6 = ''
    var_7 = '\n    name: John\n    age: not_a_number\n    active: true\n    '
    var_8 = 'age'
    var_9 = '\n    name: John\n    '
    var_10 = module_0.String()
    var_11 = '- item1\n- item2'



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = b"name: 'Jane'\nage: 25\nactive: 'false'"
    var_5 = '\n    name: "John\n    age: : :\n    '
    var_6 = '   '
    var_7 = '\n    name: "John"\n    age: "not_a_number"\n    active: "true"\n    '
    var_8 = 'age'
    var_9 = '\n    name: "John"\n    active: "true"\n    '

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = 'name: John Doe\nage: 30\nactive: true'
    var_4 = 'name: John Doe\nage: not_an_int\nactive: true'
    var_5 = 'age'
    var_6 = 'name: John Doe\n  age: 30\n    invalid: :'
    var_7 = ''
    var_8 = b'name: Jane Doe\nage: 25\nactive: false'
    var_9 = module_0.String()
    var_10 = '- item1\n- item2'
    var_11 = module_0.String()
    var_12 = 'tags: [python, testing]'
    var_13 = '\n    int_val: 10\n    float_val: 10.5\n    bool_val: true\n    null_val: null\n    '
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = module_0.String()



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello world'
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
    var_16 = 'key: value\nnum: 10'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '\n    root:\n      list:\n        - 1\n        - 2\n      map:\n        a: true\n    '
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: : value'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: value\n  broken: : error'
    var_25 = module_0.tokenize_yaml(var_24)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = b"name: 'Jane'\nage: 25\nactive: 'false'"
    var_5 = '\n    items:\n      - id: 1\n      - id: 2\n    '
    var_6 = 'val: 10'
    var_7 = '\n    name: "John"\n    age: : : : \n    '
    var_8 = '   '
    var_9 = '\n    name: "John"\n    age: "not_an_integer"\n    active: "true"\n    '
    var_10 = 'age'
    var_11 = '\n    name: "John"\n    '



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello world'
    var_2 = module_0.Integer()
    var_3 = '123'
    var_4 = module_0.Integer()
    var_5 = '- 1\n- 2\n- 3'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'name: John\nage: 30'
    var_12 = module_0.String()
    var_13 = '123'
    var_14 = module_0.Integer()
    var_15 = 'not_an_int'
    var_16 = 'key:\n  value\n    bad_indent: true'
    var_17 = 'key'
    var_18 = module_0.String()
    var_19 = {var_17: var_18}
    var_20 = ''
    var_21 = module_1.validate_yaml(var_20, var_0)
    var_22 = b'foo: bar'
    var_23 = 'foo'
    var_24 = module_0.String()
    var_25 = {var_23: var_24}
    var_26 = 'users'
    var_27 = 'id'
    var_28 = 'active'
    var_29 = module_0.Integer()
    var_30 = module_0.String()
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = '\n    users:\n      - id: 1\n        active: "true"\n      - id: 2\n        active: "false"\n    '



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = 'name: John\nage: 30\nactive: true'
    var_4 = 'name: John\nage: : : 30'
    var_5 = '   '
    var_6 = 'name: John\nage: not_a_number\nactive: true'
    var_7 = 'age'
    var_8 = b'name: Jane\nage: 25\nactive: false'
    var_9 = '- item1\n- item2'
    var_10 = module_1.tokenize_yaml(var_9)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = '\n    - item1\n    - item2\n    '
    var_5 = module_0.String()
    var_6 = [var_5]
    var_7 = '\n    name: "John\n    age: 30\n    '
    var_8 = '   '
    var_9 = '\n    name: "John"\n    age: "not_a_number"\n    active: "true"\n    '
    var_10 = 'age'
    var_11 = 0
    var_12 = b"name: 'Byte Test'\nage: 25\nactive: 'false'"
    var_13 = '\n    float_val: 1.5\n    int_val: 10\n    bool_val: true\n    null_val: null\n    '
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_0.String()



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: : John'
    var_5 = '   '
    var_6 = 'name: John\nage: not_an_int'
    var_7 = 'age'
    var_8 = 'items:\n  - 1\n  - 2'
    var_9 = 'items:\n  - 1\n  - "abc"'



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = var_0
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = '123'
    var_4 = module_0.tokenize_yaml(var_3)
    var_5 = '45.67'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = 'true'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = 'null'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '- item1\n- item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'key: value\nfoo: bar'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = b'name: test'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = '   '
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'key: : value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = 'first\nsecond'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = 'line1\nline2'
    var_26 = module_0.tokenize_yaml(var_25)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello world'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '123'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = var_7.value
    var_9 = '45.67'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = var_10.value
    var_12 = 'true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '- item1\n- item2'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'key: value\nnum: 10'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = b'name: python'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: : value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'line1\nline2\nline3'
    var_27 = 10
    var_28 = module_0._get_position(var_26, var_27)



# Parsed testcases at query #7
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
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = b'name: test'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '   '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: : value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '\n    list:\n      - num: 1\n        flag: true\n      - num: 2\n        flag: false\n    '
    var_23 = module_0.tokenize_yaml(var_22)

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
    var_2 = '\n    name: "John Doe"\n    age: 30\n    tags:\n      - python\n      - developer\n    meta:\n      location: "New York"\n    '
    var_3 = '\n    name: "John Doe"\n    age: : : : \n    '
    var_4 = '   '
    var_5 = '\n    name: 123\n    age: "not_an_int"\n    tags: "not_a_list"\n    meta: []\n    '
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'tags'
    var_9 = b"name: 'Byte Test'\nage: 1\ntags: []\nmeta: {}"
    var_10 = module_0.String()
    var_11 = 'Just a string'
    var_12 = '123'
    var_13 = module_0.Integer()



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = 'name: John\nage: 30\nactive: true'
    var_4 = b'name: Jane\nage: 25\nactive: false'
    var_5 = 'name: John\nage: not_an_int\nactive: true'
    var_6 = 'name: John\nactive: true'
    var_7 = 0
    var_8 = 'name: John\n  age: : : invalid'
    var_9 = ''
    var_10 = 'items:\n  - 1\n  - 2'
    var_11 = module_0.Integer()
    var_12 = 'items:\n  - val: 1\n  - val: 2'



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    tags:\n      - python\n      - testing\n    active: "true"\n    '
    var_4 = '\n    name: "John Doe"\n    age: : : :\n    '
    var_5 = '   '
    var_6 = '\n    name: "John Doe"\n    age: "not_an_integer"\n    tags: []\n    active: "true"\n    '
    var_7 = 'age'
    var_8 = b"name: 'Byte Test'\nage: 20\ntags: []\nactive: 'false'"
    var_9 = '\n    metadata:\n      id: 123\n      labels: [internal, secret]\n    '



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = b"name: 'Jane'\nage: 25\nactive: 'false'"
    var_5 = '\n    name: "John"\n    age: : : : \n    '
    var_6 = '   '
    var_7 = '\n    name: "John"\n    age: "not_a_number"\n    active: "true"\n    '
    var_8 = 'age'
    var_9 = '\n    name: "John"\n    active: "true"\n    '
    var_10 = module_0.String()
    var_11 = '\n    - item1\n    - item2\n    '
    var_12 = module_0.String()

import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 7
    var_2 = module_0._get_position(var_0, var_1)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\nage: : : 30'
    var_5 = ''
    var_6 = 'name: John\nage: not_an_int'
    var_7 = 'age'
    var_8 = module_0.String()
    var_9 = 'items: [a, b, c]'
    var_10 = module_0.String()
    var_11 = '\n    user:\n      name: Alice\n      details:\n        active: true\n    '



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: John Doe\n    age: 30\n    active: true\n    '
    var_4 = b'name: Jane Doe\nage: 25\nactive: false'
    var_5 = '\n    name: John\n    age: not_a_number\n    active: true\n    '
    var_6 = 'age'
    var_7 = '\n    name: John\n    age: : : :\n    '
    var_8 = '   '
    var_9 = '\n    items:\n      - 1\n      - 2\n      - 3\n    '
    var_10 = '\n    user:\n      name: Alice\n      meta:\n        id: 123\n    '



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = '\n    name: "John Doe"\n    age: 30\n    active: "true"\n    '
    var_4 = 'name'
    var_5 = 'age'
    var_6 = '- apple\n- banana\n- cherry'
    var_7 = 'items'
    var_8 = '- apple\n- banana'
    var_9 = '\n    name: "John\n    age: 30\n    '
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = 'age: not_a_number'
    var_14 = b"name: 'Byte Test'\nage: 25\nactive: 'true'"
    var_15 = 'key: value'
    var_16 = 'key'
    var_17 = module_0.String()
    var_18 = {var_16: var_17}
    var_19 = module_1.Schema(var_18)



