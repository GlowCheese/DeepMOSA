####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = b'name: John\nage: 30'
    var_5 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_6 = 'name: John\nage: thirty'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: 123\nage: thirty'
    var_3 = ''
    var_4 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_5 = b'name: Bob\nage: 40'



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value: extra'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'outer:\n  inner: value'
    var_11 = module_0.tokenize_yaml(var_10)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: 30'
    var_3 = ''
    var_4 = b'name: John\nage: 30'
    var_5 = 'user: {name: John, age: 30}\nitems: [1, 2, 3]'
    var_6 = "name: 'John Doe'\nage: 30"



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\nname: John Doe\nage: 30\n'
    var_1 = '\nname: John Doe\nage: thirty\n'
    var_2 = '\nname: John Doe\n'
    var_3 = ''
    var_4 = b'\nname: Jane Doe\nage: 25\n'
    var_5 = '\nuser:\n  name: Alice\n  role: admin\nsettings:\n  - dark_mode: true\n  - notifications: false\n'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: John\nage: thirty'
    var_4 = 'name: John\nage: -5'
    var_5 = ''
    var_6 = 'user'
    var_7 = 'roles'
    var_8 = 'user:\n  name: Admin\n  roles:\n    - admin\n    - user'
    var_9 = b'name: John\nage: 30'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = '\n    user:\n        name: John\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    ratio: 3.14\n    '
    var_6 = b'name: John'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 30'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n    '



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: [John'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'name: John\nextra: field'
    var_7 = 'name: John'
    var_8 = 'user'
    var_9 = 'user:\n  name: John\n  age: 30'
    var_10 = 'tags'
    var_11 = 'tags:\n  - python\n  - testing'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = 'user:\n  name: John\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = b'name: John'
    var_7 = 'name: 123\nage: abc'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = 'value: null'
    var_7 = b'name: John'
    var_8 = 'name: John\nage: 30'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = '\n    name: John Doe\n    age: 30\n    invalid: [unclosed\n    '
    var_2 = '\n    name: 123\n    age: not_a_number\n    '
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = '\n    user:\n        name: Jane\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = b'name: Test\nage: 20'



# Parsed testcases at query #13
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
    var_17 = 'list:\n  - item1\n  - item2\nnested:\n  key: value'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: 123'
    var_2 = 'age: 30'
    var_3 = ''
    var_4 = 'name: John Doe\nage: 30\ninvalid: [unclosed'
    var_5 = '\nuser:\n  name: John Doe\n  age: 30\nitems:\n  - item1\n  - item2\n'
    var_6 = '\ncount: 42\nactive: true\nratio: 3.14\n'
    var_7 = '\nname: null\n'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'user:\n  name: Alice\n  age: 25\n tags:\n  - python\n  - testing'
    var_6 = b'name: Bob\nage: 40'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = '\n    name: John Doe\n    age: 30\n    invalid: [unclosed bracket\n    '
    var_2 = '\n    name: J\n    age: -5\n    '
    var_3 = 'min_length'
    var_4 = 'minimum'
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = '\n    bool_val: true\n    float_val: 3.14\n    null_val: null\n    '



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = 'items:\n  - item1\n  - item2'
    var_5 = 'user:\n  name: Alice\n  age: 25'
    var_6 = b'name: John'
    var_7 = 'name: Bob\nage: 15'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: -5'
    var_3 = ''
    var_4 = b'name: Alice'
    var_5 = '\n    user:\n        id: 1\n        name: Bob\n    items:\n        - apple\n        - banana\n    '



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: invalid'
    var_4 = ''
    var_5 = 'name: John'
    var_6 = b'name: John\nage: 30'
    var_7 = 'user:\n  name: John\n  age: 30'
    var_8 = 'user'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\ninvalid: yaml: content'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = 'user:\n  name: John\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = 'age: 25\nprice: 19.99\nactive: true\ndescription: null'
    var_7 = b'name: John'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key'
    var_2 = '- item1\n- item2'
    var_3 = 'items'
    var_4 = 'key: [unclosed'
    var_5 = 'key: 123'
    var_6 = ''
    var_7 = 'outer:\n  inner: value'
    var_8 = 'outer'
    var_9 = 'int_val: 42\nfloat_val: 3.14\nbool_val: true\nnull_val: null'
    var_10 = 'int_val'
    var_11 = 'float_val'
    var_12 = 'bool_val'
    var_13 = 'null_val'
    var_14 = None
    var_15 = b'key: value'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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
    var_17 = '{a: 1, b: 2}'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = '{a: [1, 2], b: {c: 3}}'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'invalid: yaml: content: [unclosed'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = b'hello'
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = b'name: John\nage: 30'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'user: {name: Alice, age: 28}\nitems: [1, 2, 3]'
    var_6 = 'name: \'John "The Boss" Doe\'\nage: 40'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 15'
    var_3 = ''
    var_4 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = b'name: Bob'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: John\nage: twenty'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = b'name: Bob'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\ninvalid: yaml: content'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = '\n    user:\n        name: John\n        age: 30\n    items:\n        - item1\n        - item2\n    '
    var_5 = b'name: John'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'name: 123'
    var_3 = 'name'
    var_4 = ''
    var_5 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_6 = b'name: Bob'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: invalid_yaml_syntax: 30'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = b'name: Jane\nage: 25'
    var_7 = 'user:\n  name: Alice\n  age: 28'
    var_8 = 'user'
    var_9 = 'items:\n  - apple\n  - banana'
    var_10 = 'items'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = 'name: John'
    var_5 = 'name: John\nextra: field'
    var_6 = b'name: John\nage: 30'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: [unclosed'
    var_2 = 'key: 123'
    var_3 = ''
    var_4 = '\n    outer:\n      inner: value\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = b'key: value'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\nuser:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana\n'
    var_5 = '\ncount: 42\nprice: 19.99\nactive: true\ntags:\n  - python\n  - yaml\n'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 25'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n    '
    var_6 = b'name: Bob'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = 'items: [1, two, 3]'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid'
    var_2 = ''
    var_3 = 'name: Bob'
    var_4 = '\n    user:\n        name: Alice\n    items:\n        - apple\n        - banana\n    '
    var_5 = 'count: not_a_number'
    var_6 = 'other_field: value'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = 'user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark'
    var_6 = b'name: Bob\nage: 40'



# Parsed testcases at query #41
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
    var_10 = 'invalid: yaml: content: ['
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = "key: 'value with spaces'"
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'outer:\n  inner:\n    - item1\n    - item2'
    var_17 = module_0.tokenize_yaml(var_16)



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: J'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n    '
    var_6 = '\n    items:\n        - apple\n        - 123\n    '



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    description: null\n    '
    var_6 = b'name: John'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'name: John'
    var_6 = True
    var_7 = 'name: John\nage: 30\nextra: field'
    var_8 = 'extra'
    var_9 = 'user: {name: John, age: 30}\nitems: [1, 2, 3]'
    var_10 = b'name: John\nage: 30'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: 123'
    var_2 = 'key: [unclosed'
    var_3 = ''
    var_4 = b'key: value'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = ''
    var_3 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_4 = '42'
    var_5 = 'forty-two'
    var_6 = b'name: Jane\nage: 25'
    var_7 = 'user: {name: Alice, age: 28}\nitems: [1, 2, 3]'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags: [python, testing]\n    '



# Parsed testcases at query #48
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
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key: value: extra'
    var_15 = module_0.tokenize_yaml(var_14)



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = ''
    var_3 = 'name: John\nage: 17'
    var_4 = 'minimum'
    var_5 = 'min_length'
    var_6 = 'user:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana'
    var_7 = b'name: Bob'



# Parsed testcases at query #50
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
    var_17 = '{a: 1, b: 2}'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = '{a: [1, 2], b: {c: 3}}'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'key: value\nlist:\n  - item1\n  - item2'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = len(var_21)
    var_24 = var_23 - var_2
    var_25 = b'hello'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = 'invalid: yaml: content: [unclosed'
    var_28 = module_0.tokenize_yaml(var_27)



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = 'items:\n  - 1\n  - 2\n  - 3'
    var_4 = 'user:\n  name: John\n  age: 30'
    var_5 = ''
    var_6 = 'name: John\ninvalid: yaml: content'
    var_7 = b'name: John'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Jo\nage: thirty'
    var_3 = 'minimum length'
    var_4 = 'valid integer'
    var_5 = ''
    var_6 = '\nuser:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana\n  - cherry\n'
    var_7 = b'name: Bob'
    var_8 = 'description: This is a test with \'quotes\' and "double quotes"'
    var_9 = 'active: true\nvalue: null'



# Parsed testcases at query #53
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
    var_17 = 'line1: value1\nline2: value2'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = '\n    user:\n        name: Jane\n        email: jane@example.com\n    age: 30\n    '
    var_5 = '\n    items:\n        - apple\n        - banana\n        - cherry\n    '
    var_6 = b'name: John'



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Jo\nage: 15'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25'
    var_7 = 'items:\n  - 1\n  - 2\n  - 3'
    var_8 = b'name: Bob'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 15'
    var_3 = ''
    var_4 = 'user:\n  age: 30\n  score: 100'
    var_5 = 'items:\n  - 1\n  - 2\n  - 3'
    var_6 = b'name: Alice'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = '\n    name: Jane\n    age: 30\n    address:\n        street: 123 Main St\n        city: New York\n    '
    var_5 = '\n    name: Jane\n    age: thirty\n    address:\n        street: 123 Main St\n        city: New York\n    '
    var_6 = '\n    items:\n        - item1\n        - item2\n        - item3\n    '
    var_7 = '\n    items:\n        - name: item1\n          value: 100\n        - name: item2\n          value: two hundred\n    '



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 'items:\n  - 1\n  - 2\n  - 3'
    var_3 = 'items:\n  - 1\n  - two\n  - 3'
    var_4 = ''
    var_5 = 'name: John\ninvalid: yaml: content'
    var_6 = 'user:\n  name: Jane'
    var_7 = 'user:\n  name: 123'
    var_8 = b'name: John'
    var_9 = b'name: 123'



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: 123'
    var_2 = 'key: [unclosed'
    var_3 = ''
    var_4 = 'name: John\nage: 30'
    var_5 = 'name: John\nage: thirty'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'name: 123'
    var_3 = ''
    var_4 = 'user:\n  name: Jane\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = b'name: Test User'



# Parsed testcases at query #62
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
    var_19 = 'invalid: yaml: content:'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: 123'
    var_2 = 'key: [unclosed'
    var_3 = ''
    var_4 = b'key: value'
    var_5 = 'name: ab'
    var_6 = 'name: abc'



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: value\n  - invalid yaml'
    var_2 = 'name: Alice\nage: 30'
    var_3 = 'name: Al\nage: -5'
    var_4 = 'name'
    var_5 = 'age'
    var_6 = ''
    var_7 = 'name: Bob\nage: twenty'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [unclosed'
    assert var_1 == 1
    var_2 = 'age: not_a_number'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = b'name: Bob'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = 0
    var_4 = ''
    var_5 = 'user: {name: Alice, age: 25}\nitems: [1, 2, 3]'
    var_6 = b'name: Bob\nage: 40'



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: twenty'
    var_3 = ''
    var_4 = 'user:\n  name: Alice\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: John Doe\ninvalid: yaml: content'
    var_2 = 'name: Joe\nage: 17'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_7 = b'name: Bob'



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: 123'
    var_2 = 'key: [unclosed'
    var_3 = ''
    var_4 = b'key: value'
    var_5 = 'name: validname'
    var_6 = 'name: verylongname'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: [John'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'user'
    var_7 = 'roles'
    var_8 = 'user:\n  name: Alice\n  roles:\n    - admin\n    - user'
    var_9 = b'name: Bob'



# Parsed testcases at query #71
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
    var_11 = 'key:\n  - item1\n  - item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = b'key: value'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'key: value\n  invalid yaml'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Jo\nage: 15'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags: [python, testing]\n    '
    var_8 = 'optional: null'
    var_9 = b'name: Test'



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'user:\n  name: John\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'



# Parsed testcases at query #74
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
    var_11 = 'key:\n  - item1\n  - item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'key: [value'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = '.'
    var_16 = b'hello'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_19 = module_0.tokenize_yaml(var_18)



# Parsed testcases at query #75
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
    var_15 = '- a\n- b\n- c'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'a: 1\nb: 2\nc: 3'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'a:\n  b:\n    - c\n    - d'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'hello'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = 'a: [\n  b: c\n]'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = '.'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 15'
    var_3 = ''
    var_4 = 'user:\n  age: 30\n  score: 100'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = b'name: Alice'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'invalid: yaml: content'
    assert var_1 == 1
    var_2 = 'age: -5'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = '\n    user:\n        name: Bob\n        age: -10\n    '



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: Bob\nage: 15'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: invalid_yaml_syntax:'
    var_4 = 'name: John\nage: thirty'
    var_5 = ''
    var_6 = 'name: John\nage: 30\ncity: New York'
    var_7 = 'name: John'
    var_8 = 'user:\n  name: John\n  age: 30'
    var_9 = 'user'
    var_10 = 'items:\n  - apple\n  - banana'
    var_11 = 'items'
    var_12 = b'name: John\nage: 30'
    var_13 = 'active: true\nverified: false'
    var_14 = 'active'
    var_15 = 'verified'
    var_16 = 'name: John\nmiddle_name: null'
    var_17 = 'middle_name'



# Parsed testcases at query #80
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value: extra'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'scalar_value'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_13 = module_0.tokenize_yaml(var_12)



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = 'optional: null'
    var_7 = b'name: Jane'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = '\nname: John\nage: 30\n'
    var_1 = 'name: John\nage: invalid_yaml_syntax:'
    var_2 = '\nname: J\nage: -5\n'
    var_3 = 'min_length'
    var_4 = 'minimum'
    var_5 = ''
    var_6 = '\nname: John\n'
    var_7 = False
    var_8 = '\nname: John\nextra_field: value\n'



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'age: 30'
    var_5 = 'name: John\nage: 30\nextra: field'
    var_6 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #84
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = b'name: Jane'



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: John Doe\nage: invalid_yaml: 30'
    var_2 = 'name: Jo\nage: -5'
    var_3 = 1
    var_4 = 'min_length'
    var_5 = 'minimum'
    var_6 = ''
    var_7 = '\nuser:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana\n'
    var_8 = 'count: not_a_number\nactive: yes'
    var_9 = 'int'



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: '
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\nuser:\n  name: Alice\n  age: 30\n'
    var_5 = '\nitems:\n  - apple\n  - banana\n  - cherry\n'
    var_6 = '\nitems:\n  - 1\n  - 2\n  - 3\n'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name:'
    var_2 = '\n    name: Jane\n    age: 25\n    hobbies:\n      - reading\n      - hiking\n    '
    var_3 = '\n    name: Jane\n    age: -5\n    hobbies:\n      - reading\n      - 123\n    '
    var_4 = 'minimum'
    var_5 = 'type'
    var_6 = ''
    var_7 = 'name: John\nage: 25\ninvalid: yaml: content'
    var_8 = b'name: John'



# Parsed testcases at query #2
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
    var_18 = '\n    key:\n      - item1\n      - item2\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'hello'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'invalid: yaml: content: [unclosed'
    var_23 = module_0.tokenize_yaml(var_22)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = '\nuser:\n  name: Alice\n  age: 30\n'
    var_3 = '\nitems:\n  - 1\n  - 2\n  - 3\n'
    var_4 = 'name: [John'
    var_5 = ''
    var_6 = b'name: Bob'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_yaml: 30'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = 'user:\n  name: John\n  age: 30'
    var_5 = 'items:\n  - apple\n  - banana'
    var_6 = b'name: John'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_5 = '\n    items:\n      - 1\n      - 2\n      - 3\n    '
    var_6 = '\n    name: Bo\n    age: 15\n    '
    var_7 = 'min_length'
    var_8 = 'minimum'



# Parsed testcases at query #6
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
    var_9 = b'key: value'
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = '- item1\n- item2'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'scalar'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = '42'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = '3.14'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'true'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'null'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = 'key: value: extra'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = '.'



# Parsed testcases at query #7
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
    var_23 = '\n    nested:\n      mapping:\n        key: value\n      sequence:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: 17'
    var_3 = 'minimum length'
    var_4 = 'minimum value'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\ntags:\n  - python\n  - testing'
    var_7 = b'name: Bob\nage: 40'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: J\nage: -5'
    var_2 = 'name: John\nage: :invalid'
    var_3 = ''
    var_4 = 'user:\n  name: John\n  age: 30\nitems:\n  - apple\n  - banana'
    var_5 = b'name: John\nage: 30'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_3 = ''
    var_4 = True
    var_5 = 'name: John\nage: 30\nextra: field'
    var_6 = 'age: 30'
    var_7 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_8 = b'name: John\nage: 30'



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
    var_19 = 'invalid: yaml: content: :'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = b'key: value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    nested:\n      mapping:\n        key: value\n      list:\n        - item1\n        - item2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: Bob'
    var_3 = 1
    var_4 = 0
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_7 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags:\n        - python\n        - testing\n    '
    var_8 = 'value: null'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    users:\n      - Alice\n      - Bob\n    config:\n      debug: true\n      timeout: 30\n    '
    var_5 = b'name: Jane'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    assert var_1 == 1
    var_2 = 'age: 17'
    var_3 = ''
    var_4 = b'name: Alice'
    var_5 = '\n    users:\n      - name: John\n        role: admin\n      - name: Jane\n        role: user\n    '



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = 'name: John'
    var_5 = 'name: John\nage: 30\nextra: field'
    var_6 = 'user:\n  name: Jane\n  age: 25'
    var_7 = 'items:\n  - apple\n  - banana'



# Parsed testcases at query #16
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
    var_19 = b'hello'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'invalid: yaml: content: [unclosed'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '.'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = b'name: Alice'
    var_5 = '\n    user:\n        name: John\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_6 = '\n    count: 42\n    price: 19.99\n    active: true\n    description: null\n    '



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'user:\n  name: Alice\n  age: 28\nitems:\n  - apple\n  - banana'
    var_6 = 'name: \'John "The Boss" Doe\'\nage: 30'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'name: John'
    var_6 = 'name: John\nextra: field'
    var_7 = 'extra'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\nname: John Doe\nage: 30\n'
    var_1 = '\nname: John Doe\nage: 30\ninvalid: [unclosed bracket\n'
    var_2 = '\nname: John Doe\nage: thirty\n'
    var_3 = 'age'
    var_4 = ''
    var_5 = '\nname: John Doe\nage: 30\nextra: field\n'
    var_6 = '\nname: John Doe\n'
    var_7 = b'name: John Doe\nage: 30'



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'scalar_value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = ''
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'invalid: yaml: content'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_15 = module_0.tokenize_yaml(var_14)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John'
    var_3 = ''
    var_4 = 'items:\n  - apple\n  - banana'
    var_5 = 'user:\n  name: John\n  age: 30'
    var_6 = b'name: John'
    var_7 = "name: John O'Brien"



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = 'loc'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_7 = 'name: \'John "The Boss" Doe\'\nage: 40'
    var_8 = 'active: true\noptional: null'
    var_9 = b'name: Jane\nage: 28'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_yaml: 30'
    var_2 = 'name: Jo\nage: 17'
    var_3 = 'minimum length is 5'
    var_4 = 'must be greater than or equal to 18'
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = b'name: Bob'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'user:\n  name: Alice\n  age: 25\ntags:\n  - python\n  - testing'
    var_6 = b'name: Bob\nage: 40'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_5 = b'name: Bob\nage: 40'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 'items:\n  - 1\n  - 2\n  - 3'
    var_3 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_4 = ''
    var_5 = 'user:\n  name: John\n  age: 30'
    var_6 = b'name: John'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    assert var_1 == 1
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\nuser:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana\n'
    var_5 = '\ncount: 42\nprice: 19.99\nactive: true\n'
    var_6 = 'value: null'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = '\nuser:\n  name: John\n  age: 30\nitems:\n  - apple\n  - banana\n'
    var_5 = 'count: not_a_number'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = ''
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = 'user:\n  name: Alice\n  age: 30'
    var_6 = b'name: Bob'
    var_7 = 'name: John\nage: 30\ninvalid: yaml: content'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = ''
    var_3 = 'name: John\nage: 30\ninvalid: :'
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'user:\n  name: Alice\n  age: 28'
    var_6 = 'items:\n  - 1\n  - 2\n  - 3'
    var_7 = 'name: Bob'



# Parsed testcases at query #32
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
    var_20 = 'key:\n  - item1\n  - item2'
    var_21 = module_0.tokenize_yaml(var_20)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_2 = 'name: Bob'
    var_3 = 1
    var_4 = 0
    var_5 = ''
    var_6 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_7 = '\n    count: 42\n    price: 19.99\n    active: true\n    '
    var_8 = 'value: null'
    var_9 = b'name: John'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: 123\nage: thirty'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = 'name: Jane'
    var_7 = True
    var_8 = 'name: Bob\nage: 25\nextra: field'
    var_9 = 'name: \nage: -5'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_3 = ''
    var_4 = 'forbid'
    var_5 = 'name: John\nage: 30\nextra: field'
    var_6 = 'age: 30'
    var_7 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_8 = b'name: John\nage: 30'



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = len(var_2)
    var_5 = 1
    var_6 = var_4 - var_5
    var_7 = '- item1\n- item2'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = len(var_7)
    var_10 = var_9 - var_5
    var_11 = 'scalar_value'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = len(var_11)
    var_14 = var_13 - var_5
    var_15 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'key: [unclosed'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = b'key: value'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_22 = module_0.tokenize_yaml(var_21)



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_yaml_syntax:'
    var_2 = 'name: John\nage: thirty'
    var_3 = 'age'
    var_4 = 'loc'
    var_5 = ''
    var_6 = 'name: John'
    var_7 = False
    var_8 = 'name: John\nage: 30\nextra_field: value'
    var_9 = 'extra_field'
    var_10 = 'user:\n  name: John\n  age: 30\nitems:\n  - item1\n  - item2'
    var_11 = b'name: John\nage: 30'



# Parsed testcases at query #38
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
    var_14 = 'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = b'hello'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'invalid: yaml: content: [unclosed'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: value\nlist:\n  - item1\n  - item2'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = len(var_20)
    var_23 = 1
    var_24 = var_22 - var_23



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = b'name: Alice'
    var_5 = '\n    user:\n        name: John\n        age: 30\n    items:\n        - apple\n        - banana\n    '
    var_6 = '\n    count: 42\n    price: 3.14\n    active: true\n    description: null\n    '



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = ''
    var_3 = 'name: [John'
    var_4 = '\n    name: Jane\n    age: 30\n    address:\n      street: 123 Main St\n      city: New York\n    '
    var_5 = '\n    name: Jane\n    age: thirty\n    address:\n      street: 123 Main St\n      city: New York\n    '
    var_6 = '\n    items:\n      - item1\n      - item2\n      - item3\n    '
    var_7 = '\n    items:\n      - name: item1\n        value: 10\n      - name: item2\n        value: twenty\n    '



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = 'age'
    var_4 = 'loc'
    var_5 = ''
    var_6 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_7 = b'name: Bob\nage: 40'



# Parsed testcases at query #42
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
    var_23 = '\n    nested:\n      list:\n        - item1\n        - item2\n      dict:\n        key1: value1\n        key2: value2\n    '
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #43
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
    var_11 = 'list:\n  - item1\n  - item2\nnested:\n  key: value'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = 'key: value: extra'
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = 'int: 42\nfloat: 3.14\nbool: true\nnull: null'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = b'key: value'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #44
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
    var_9 = b''
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = module_1.Position(var_6, var_6, var_7)
    var_12 = 'key: value: extra'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'scalar'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    int: 1\n    float: 1.1\n    bool: true\n    null: null\n    '
    var_19 = module_0.tokenize_yaml(var_18)



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = '\nuser:\n  name: Alice\n  age: 30\n'
    var_3 = '\nitems:\n  - 1\n  - 2\n  - 3\n'
    var_4 = 'name: [John'
    var_5 = ''
    var_6 = b'name: Bob'



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = '\n    name: John Doe\n    age: thirty\n    '
    var_2 = '\n    name: John Doe\n    age: 3000\n    '
    var_3 = ''
    var_4 = '\n    name: John Doe\n    '
    var_5 = b'\n    name: John Doe\n    age: 30\n    '



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: John Doe\ninvalid: [unclosed'
    var_2 = 'name: Jo\nage: 25'
    var_3 = 1
    var_4 = ''
    var_5 = 'items:\n  - item1\n  - item2'
    var_6 = '\n    user:\n      name: Alice\n      age: 30\n    settings:\n      theme: dark\n      notifications: true\n    '
    var_7 = 'count: not_a_number\nactive: yes'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'items:\n  - apple\n  - banana'
    var_5 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_6 = b'name: Bob\nage: 25'



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_type'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_5 = 'user:\n  name: Jane'
    var_6 = 'items:\n  - apple\n  - banana'
    var_7 = b'name: John'
    var_8 = 'name: ThisNameIsTooLong'



# Parsed testcases at query #50
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value: extra'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'outer:\n  inner: value'
    var_11 = module_0.tokenize_yaml(var_10)



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'user:\n  name: Alice\nitems:\n  - 1\n  - 2'
    var_6 = b'name: Bob\nage: 25'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: -5'
    var_3 = 'age'
    var_4 = ''
    var_5 = 'user: {name: Alice, age: 25}\nitems: [1, 2, 3]'
    var_6 = b'name: Bob\nage: 40'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: [John'
    var_2 = 'age: 25'
    var_3 = ''
    var_4 = b'name: Jane'
    var_5 = 'user:\n  name: Alice\n  age: 30'



# Parsed testcases at query #54
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value: extra'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'outer:\n  inner: value'
    var_11 = module_0.tokenize_yaml(var_10)



# Parsed testcases at query #55
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
    var_17 = 'nested:\n  list:\n    - item1\n    - item2'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #56
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
    var_17 = 'a: b\nc: d'
    var_18 = module_0.tokenize_yaml(var_17)



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_yaml: 30'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    price: 19.99\n    active: true\n    tags: [python, testing]\n    '
    var_6 = b'name: Jane'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = 1
    var_4 = ''
    var_5 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_6 = 'user'
    var_7 = 'items'
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'Alice'
    var_11 = 25
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'apple'
    var_14 = 'banana'
    var_15 = [var_13, var_14]
    var_16 = {var_6: var_12, var_7: var_15}
    var_17 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_18 = 'count'
    var_19 = 'active'
    var_20 = 'price'
    var_21 = 42
    var_22 = True
    var_23 = 19.99
    var_24 = {var_18: var_21, var_19: var_22, var_20: var_23}
    var_25 = 'value: null'



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = 'optional: null'
    var_7 = b'name: John'



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = '\n    user:\n        name: John\n        age: 30\n    items:\n        - item1\n        - item2\n    '
    var_5 = b'name: Jane'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'name: John'
    var_5 = 'name: John\nage: 30'
    var_6 = 'user:\n  name: John\nitems:\n  - apple\n  - banana'
    var_7 = b'name: John\nage: 30'



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'key: 123'
    var_2 = 'key: [unclosed'
    var_3 = ''
    var_4 = b'key: value'
    var_5 = 'name: John'
    var_6 = 'name: '



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = 1
    var_4 = ''
    var_5 = 'user: {name: Alice}\nitems: [1, 2, 3]'
    var_6 = b'name: Bob\nage: 25'
    var_7 = "name: 'John Doe'\nage: 30"



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = 'items:\n  - apple\n  - banana'
    var_4 = 'items:\n  - apple\n  - 123'
    var_5 = ''
    var_6 = 'name: John\ninvalid: yaml: content'
    var_7 = 'user:\n  age: 30\n  height: 180'
    var_8 = 'user:\n  age: thirty\n  height: 180'
    var_9 = b'name: John'



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'invalid: yaml: content'
    assert var_1 == 1
    var_2 = 'age: -5'
    var_3 = ''
    var_4 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = b'name: Bob'



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = ''
    var_3 = 'name: John\nage: 30\ninvalid: yaml: content'
    var_4 = 'name: John'
    var_5 = True
    var_6 = 'name: John\nage: 30\nextra: field'
    var_7 = 'user:\n  name: John\n  age: 30\nsettings:\n  - dark_mode: true\n  - notifications: false'
    var_8 = b'name: John\nage: 30'



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: invalid_yaml: 30'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\n    user:\n        name: Alice\n        age: 25\n    items:\n        - apple\n        - banana\n    '
    var_5 = '\n    count: 42\n    active: true\n    price: 19.99\n    '
    var_6 = 'optional: null'
    var_7 = b'name: John'



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'user:\n  name: Alice\n  role: admin\nsettings:\n  - dark_mode: true\n  - notifications: false'



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: 30\ninvalid: [unclosed'
    var_2 = 'name: Jo'
    var_3 = ''
    var_4 = 'items:\n  - a\n  - b\n  - c'
    var_5 = 'user:\n  name: Alice\n  age: 25\nsettings:\n  theme: dark'
    var_6 = b'name: Bob'
    var_7 = 'count: 42'
    var_8 = 'count: not_a_number'



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = 'items:\n  - 1\n  - 2\n  - 3'
    var_5 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_6 = b'name: Bob\nage: 25'
    var_7 = 'name: José\nage: 40'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = 'items:\n  - 1\n  - 2'
    var_4 = 'name: John\ninvalid: yaml: content'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30'
    var_7 = 'name: Bob'



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: Tom\nage: 15'
    var_3 = 'minimum length'
    var_4 = 'text'
    var_5 = 'minimum value'
    var_6 = ''
    var_7 = 'user:\n  name: Alice\n  age: 25\nitems:\n  - apple\n  - banana'
    var_8 = b'name: Bob\nage: 40'



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_int'
    var_2 = 'name: J\nage: -5'
    var_3 = 'min_length'
    var_4 = 'minimum'
    var_5 = ''
    var_6 = b'name: Alice\nage: 25'
    var_7 = 'user:\n  name: Bob\n  age: 40\nitems:\n  - apple\n  - banana'



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 'name: [John Doe'
    var_2 = 'age: twenty'
    var_3 = ''
    var_4 = '\n    user:\n      name: Alice\n      age: 30\n    '
    var_5 = 'items: [1, 2, 3]'
    var_6 = 'items: [1, two, 3]'



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: :invalid'
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\nuser:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana\n'
    var_5 = '\ncount: 42\nprice: 19.99\nactive: true\ntag: null\n'
    var_6 = b'name: John'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid'
    var_2 = 'name: John\nage: thirty'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'user:\n  name: Alice\n  age: 28\nitems:\n  - item1\n  - item2'
    var_6 = "name: 'John Doe'\nage: 30"
    var_7 = 'active: true\noptional: null'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John\nage: '
    var_2 = 'name: Bob'
    var_3 = ''
    var_4 = '\nuser:\n  name: Alice\n  age: 30\nitems:\n  - apple\n  - banana\n'
    var_5 = 'count: not_a_number'
    var_6 = b'name: John'



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: invalid_age'
    var_2 = 'name: John'
    var_3 = ''
    var_4 = b'name: Jane\nage: 25'



# Parsed testcases at query #79
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
    var_19 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_20 = module_0.tokenize_yaml(var_19)



# Parsed testcases at query #80
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = b'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value: extra'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'scalar'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_15 = module_0.tokenize_yaml(var_14)



