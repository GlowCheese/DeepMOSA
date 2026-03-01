####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = 'items:\n  - 1\n  - 2\n  - 3'
    var_4 = 'name: John\ninvalid: yaml: content'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30'
    var_7 = b'name: John'



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '- item1\n- item2'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'invalid: yaml: content'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'list:\n  - item1\n  - item2\nnested:\n  key: value'
    var_15 = module_0.tokenize_yaml(var_14)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\nname: John\nage: 30\n'
    var_1 = '\nname: John\nage: thirty\n'
    var_2 = '\nname: John\nage: -5\n'
    var_3 = 'age'
    var_4 = 'field'
    var_5 = ''
    var_6 = '\nname: John\n'
    var_7 = '\nname: John\nage: 30\nextra_field: value\n'
    var_8 = '\nuser:\n  name: John\n  age: 30\nsettings:\n  - dark_mode: true\n  - notifications: false\n'
    var_9 = b'\nname: John\nage: 30\n'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'invalid: yaml: content: :'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    nested:\n      list:\n        - item1\n        - item2\n      dict:\n        key1: value1\n        key2: value2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 'items:\n  - apple\n  - banana'
    var_3 = 'items:\n  - 123\n  - banana'
    var_4 = "user:\n  name: John\n  age: '30'"
    var_5 = 'user:\n  name: John\n  age: 30'
    var_6 = ''
    var_7 = 'name: John\ninvalid: yaml: content'
    var_8 = b'name: John'
    var_9 = b'name: 123'



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: value: extra'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = '\n    user:\n        name: Jane\n        age: 30\n    '
    var_5 = '\n    items:\n        - 1\n        - 2\n        - 3\n    '
    var_6 = '\n    user:\n        name: Jane\n        age: thirty\n    '
    var_7 = '\n    items:\n        - 1\n        - two\n        - 3\n    '



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = len(var_19)
    var_22 = var_21 - var_20
    var_23 = b'key: value'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = '\n    nested:\n      list:\n        - item1\n        - item2\n      dict:\n        key1: value1\n        key2: value2\n    '
    var_26 = module_0.tokenize_yaml(var_25)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: John Doe\nage: :invalid'
    var_2 = 'name: Jo\nage: 25'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    description: null\n    '



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags: [python, yaml]\n    '
    var_6 = b'name: Test'



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'hello'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = '42'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = '3.14'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = 'true'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'null'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = '[1, 2, 3]'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = '{"a": 1}'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'invalid: yaml: content: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'hello'
    var_22 = module_0.tokenize_yaml(var_21)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = ''
    var_3 = 'user:\n  name: Jane\n  age: 25\ndetails:\n  invalid_key: value'
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = b'name: John\nage: 30'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = 1
    var_4 = ''
    var_5 = '\n    user:\n        name: Alice\n        age: 30\n    tags:\n        - python\n        - testing\n    '
    var_6 = '\n    count: 42\n    active: true\n    '
    var_7 = '\n    count: not_a_number\n    active: true\n    '



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'user: {name: Alice, age: 25}\nitems: [1, 2, 3]'
    var_5 = b'name: Bob\nage: 40'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = 'items:\n  - apple\n  - banana'
    var_5 = 'user:\n  name: John\n  age: 30'
    var_6 = 'count: not_a_number'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: 123\nage: thirty'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_7 = b'name: Bob\nage: 40'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = '- item1\n- item2'
    var_2 = 'key: value'
    var_3 = 'key: value\n  invalid yaml: [unclosed'
    var_4 = ''
    var_5 = b'key: value'
    var_6 = 'name: John\nage: 30'
    var_7 = 'name: John\nage: thirty'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = 'key: value: extra'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '- item1\n- item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'outer:\n  inner: value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = 'int: 42\nfloat: 3.14\nbool: true\nnull: null'
    var_16 = module_0.tokenize_yaml(var_15)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: John\nage: invalid'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'name: John\nage: 30\nextra: field'
    var_7 = 'age: 30'
    var_8 = 'user'
    var_9 = 'email'
    var_10 = 'user:\n  name: Jane\n  email: jane@example.com'
    var_11 = 'items'
    var_12 = 'items:\n  - 1\n  - 2\n  - 3'
    var_13 = b'name: John\nage: 30'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags: [python, testing]\n    '
    var_6 = 'value: null'
    var_7 = b'name: Test'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'items:\n  - apple\n  - banana'
    var_5 = 'user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark'
    var_6 = b'name: Bob\nage: 40'
    var_7 = 'name: José\nage: 35'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: 10'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = '\n    name: John Doe\n    age: thirty\n    '
    var_2 = '\n    name: John Doe\n    age: -5\n    '
    var_3 = ''
    var_4 = '\n    name: John Doe\n    '
    var_5 = '\n    name: John Doe\n    age: 30\n    extra_field: extra\n    '
    var_6 = '\n    user:\n        name: John Doe\n        age: 30\n    settings:\n        - dark_mode: true\n        - notifications: false\n    '
    var_7 = b'\n    name: John Doe\n    age: 30\n    '



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'scalar_value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'int_val: 42\nfloat_val: 3.14\nbool_val: true\nnull_val: null'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'nested:\n  key: value\n  list:\n    - item1\n    - item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = b'key: value'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = ''
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key: value: invalid'
    var_15 = module_0.tokenize_yaml(var_14)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n    '
    var_6 = b'name: John'



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'hello'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = 'key: value'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'invalid: yaml: content'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = b'key: value'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = '\n    key1: value1\n    key2:\n      - item1\n      - item2\n    '
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = 'key: \'value with "quotes"\''
    var_26 = module_0.tokenize_yaml(var_25)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: John\nage: invalid_yaml:'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'user'
    var_7 = 'user:\n  name: Jane\n  age: 25'
    var_8 = 'tags'
    var_9 = 'tags:\n  - python\n  - testing'
    var_10 = b'name: John\nage: 30'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '3.14'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '{a: 1, b: 2}'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '{a: [1, 2], b: {c: 3}}'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'invalid: yaml: content: [unclosed'
    var_19 = module_0.tokenize_yaml(var_18)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = 'user:\n  name: Jane\n  age: 25'
    var_5 = 'items:\n  - 1\n  - 2\n  - 3'
    var_6 = b'name: John\nage: 30'



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '- item1\n- item2'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'key:\n  - item1\n  - item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key: value: extra'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = b'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = module_0.tokenize_yaml(var_4)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = 'user:\n  name: Alice'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = b'name: Bob'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'age: 150'
    var_3 = ''
    var_4 = '\n    user:\n      name: Jane Doe\n      email: jane@example.com\n    '
    var_5 = 'items: [1, 2, 3, 4]'
    var_6 = "items: [1, 2, 'three', 4]"
    var_7 = b'name: Test User'



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = 'scalar_value'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = 'key: [unclosed'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = b'key: value'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = '\n    nested:\n      mapping:\n        key: value\n      sequence:\n        - item1\n        - item2\n    '
    var_20 = module_0.tokenize_yaml(var_19)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'age: 150'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    '
    var_6 = 'value: null'
    var_7 = 'email: not-an-email'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'age'
    var_3 = 'loc'
    var_4 = 'name: Jane'
    var_5 = ''
    var_6 = 'code'
    var_7 = 'no_content'
    var_8 = 'name: Bob\nage: 25\nextra: field'
    var_9 = 'user:\n  name: Alice\n  age: 28\nitems:\n  - item1\n  - item2'
    var_10 = 'name: 123\nage: thirty'
    var_11 = 'name'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: J\nage: -5'
    var_2 = 'min_length'
    var_3 = 'minimum'
    var_4 = 'name: John\nage: invalid'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: John Doe\nage: invalid_yaml: 30'
    var_2 = 'parse_error'
    var_3 = 'name: Joe\nage: -5'
    var_4 = 'min_length'
    var_5 = 'minimum'
    var_6 = ''
    var_7 = 'no_content'
    var_8 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_9 = b'name: Bob'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_5 = b'name: Bob\nage: 40'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    tags:\n        - python\n        - testing\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    description: null\n    '



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: : 30'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25'
    var_7 = 'user'
    var_8 = 'tags:\n  - python\n  - yaml\n  - test'
    var_9 = 'tags'
    var_10 = b'name: Bob\nage: 40'



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = 'key: value: extra'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = '\n    - item1\n    - item2\n    '
    var_16 = module_0.tokenize_yaml(var_15)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: not_a_number'
    var_2 = 0
    var_3 = ''
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = 'user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark'
    var_6 = b'name: Bob\nage: 40'
    var_7 = 'name: John\nage: 30\ninvalid_key'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    name: John\n    age: 30\n    '
    var_1 = '\n    name: John\n    age: 30\n    invalid: [unclosed\n    '
    var_2 = '\n    name: 123\n    age: not_a_number\n    '
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = b'name: Bob\nage: 40'



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = b'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = ''
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = 'key: value: extra'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '- item1\n- item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = 'outer:\n  inner: value'
    var_16 = module_0.tokenize_yaml(var_15)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = 'user:\n  age: 30\n  score: 100'
    var_5 = 'items:\n  - 1\n  - 2\n  - 3'
    var_6 = b'name: Alice'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'user:\n  name: Jane\n  age: 25'
    var_6 = 'items:\n  - 1\n  - 2\n  - 3'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 1
    var_7 = 0
    var_8 = module_1.Position(var_6, var_6, var_7)
    var_9 = 'key: value: extra'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '- item1\n- item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_16 = module_0.tokenize_yaml(var_15)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'user:\n  name: John\n  age: 30\nitems:\n  - apple\n  - banana'
    var_5 = b'name: John\nage: 30'



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = '- item1\n- item2'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = 'scalar_value'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '42'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = '3.14'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = 'true'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'null'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: value: extra'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: -5'
    var_3 = 'minimum'
    var_4 = ''
    var_5 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_6 = 'items: [1, 2, 3]'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: value\n  invalid yaml: ['
    var_2 = ''
    var_3 = 'key: 123'
    var_4 = b'key: value'
    var_5 = 'name: test'
    var_6 = 'name: 123'



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_2
    var_9 = '- item1\n- item2'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_2
    var_13 = 'scalar_value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_2
    var_17 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = 'age'
    var_4 = 'loc'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: :30'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30'
    var_7 = 'user'
    var_8 = 'items:\n  - apple\n  - banana'
    var_9 = 'items'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = '\n    name: John Doe\n    age: invalid\n    '
    var_2 = '\n    name: ""\n    age: -5\n    '
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = '\n    name: Jane Doe\n    '
    var_7 = '\n    name: Bob\n    age: 25\n    extra_field: "should be ignored"\n    '
    var_8 = '\n    user:\n      name: Alice\n      email: alice@example.com\n    settings:\n      - dark_mode: true\n      - notifications: false\n    '
    var_9 = 'settings'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = 'value: null'
    var_7 = b'name: John'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: not_a_number'
    var_3 = 1
    var_4 = ''
    var_5 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_6 = '\n    items:\n        - apple\n        - banana\n    '



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: twenty'
    var_3 = ''
    var_4 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = '\n    user:\n      name: Bob\n      age: not_a_number\n    '



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: Jo\nage: 17'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_7 = b'name: Bob\nage: 40'



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '3.14'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '{a: 1, b: 2}'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'hello'
    var_19 = module_0.tokenize_yaml(var_18)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = '\n    user:\n        name: Jane\n        email: jane@example.com\n    age: 30\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n        - cherry\n    '
    var_6 = b'name: John'
    var_7 = 'name: Bob'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'user:\n  name: Alice\n  age: 25\n tags: [python, yaml]'
    var_5 = b'name: Bob\nage: 40'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = 'user:\n  name: Alice\n  age: 30'
    var_6 = b'name: Bob'



