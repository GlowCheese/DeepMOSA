####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello: world'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = 'hello: thisistoolong'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = 100
    var_11 = module_0.Integer(minimum=var_9, maximum=var_10)
    var_12 = 'value: 42'
    var_13 = module_1.validate_yaml(var_12, var_11)
    var_14 = 'value: 150'
    var_15 = module_1.validate_yaml(var_14, var_11)
    var_16 = var_15[var_6]
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = '\n    name: John Doe\n    age: 30\n    '
    var_19 = '\n    name: This name is way too long to be valid according to our schema\n    age: 200\n    '
    var_20 = var_15[var_6]
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = ''
    var_23 = '   \n  \t  \n'
    var_24 = '\n    name: John\n    age: thirty\n    extra: [unclosed list\n    '
    var_25 = b'name: Alice\nage: 25'
    var_26 = module_0.String()
    var_27 = '\n    person:\n      name: Bob\n      age: 40\n    tags: developer\n    '
    var_28 = module_0.Integer()
    var_29 = module_0.Array(var_28)
    var_30 = '[1, 2, 3]'
    var_31 = module_1.validate_yaml(var_30, var_29)
    var_32 = "[1, 'two', 3]"
    var_33 = module_1.validate_yaml(var_32, var_29)
    var_34 = var_33[var_6]
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = module_0.Boolean()
    var_37 = 'active: true'
    var_38 = 'active: false'
    var_39 = 'name: null'
    var_40 = '{}'



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test: value'
    assert var_0 == 1
    var_1 = 'name: John\nage: 25'
    var_2 = 'name: John\n  age: 25'
    var_3 = 0
    var_4 = 'mapping values are not allowed here'
    var_5 = ''
    var_6 = b'name: Alice\nage: 30'
    var_7 = 'name: VeryLongNameExceedsLimit\nage: -5'
    var_8 = module_0.Boolean()
    var_9 = '\n    user:\n      name: Bob\n      age: 40\n    active: true\n    '
    var_10 = module_0.String()
    var_11 = module_0.Array(var_10)
    var_12 = '- item1\n- item2\n- item3'
    var_13 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_14 = module_0.Integer()
    var_15 = module_0.Float()
    var_16 = module_0.Boolean()
    var_17 = 'key: [unclosed list'



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '3.14'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'false'
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
    var_18 = 1
    var_19 = var_13.value[var_18]
    var_20 = 'key: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key'
    var_23 = var_21.value[var_22]
    var_24 = 'key:\n  nested: 42'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = var_25.value[var_22]
    var_27 = b'hello'
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = '   \n  \t  '
    var_32 = module_0.tokenize_yaml(var_31)
    var_33 = 'key: [unclosed'
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = '|\n  line1\n  line2'
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - hiking\n    active: true\n    '
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = 'hobbies'
    var_40 = var_38.value[var_39]
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 2



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    outer:\n      inner: nested\n      list:\n        - item1\n        - item2\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'outer'
    var_5 = var_3.value[var_4]
    var_6 = 'list'
    var_7 = var_3.value[var_4][var_6]
    var_8 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- first\n- second\n- third'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = ''
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '   \n  \t  \n'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = b'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'key: [unclosed list'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'simple string'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'outer:\n  inner: test'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = len(var_22)
    var_25 = 1
    var_26 = var_24 - var_25
    var_27 = '\n    # This is a comment\n    key: value  # inline comment\n    '
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = 'special: \'quoted string with "quotes"\''
    var_30 = module_0.tokenize_yaml(var_29)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = var_1.value[var_2]
    var_4 = 'number: 42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'number'
    var_7 = var_5.value[var_6]
    var_8 = 'float: 3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'float'
    var_11 = var_9.value[var_10]
    var_12 = 'flag: true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'flag'
    var_15 = var_13.value[var_14]
    var_16 = 'empty: null'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'empty'
    var_19 = var_17.value[var_18]
    var_20 = '- item1\n- item2'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_21.value
    var_25 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = 'users'
    var_28 = var_26.value[var_27]
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = b'key: value'
    var_32 = module_0.tokenize_yaml(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = '   \n  \t  \n'
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = 'key: [unclosed'
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = 'key: : value'
    var_40 = module_0.tokenize_yaml(var_39)
    var_41 = 'key: value'
    var_42 = module_0.tokenize_yaml(var_41)
    var_43 = 'first: value1\nsecond: value2'
    var_44 = module_0.tokenize_yaml(var_43)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - hiking\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2\n- item3'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key: [unclosed list'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'single_value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    users:\n      - name: Alice\n        age: 25\n        active: true\n      - name: Bob\n        age: 30\n        active: false\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'users'
    var_21 = var_19.value[var_20]
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'first: value1\nsecond: value2'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = len(var_23)
    var_26 = 1
    var_27 = var_25 - var_26



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    outer:\n      inner: nested\n      list:\n        - item1\n        - item2\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'outer'
    var_5 = var_3.value[var_4]
    var_6 = 'list'
    var_7 = var_5.value[var_6]
    var_8 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = var_9.value
    var_11 = '- first\n- second\n- third'
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = b'key: value'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key: [unclosed'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'single_value'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = '\n    users:\n      - name: Alice\n        age: 30\n        active: true\n      - name: Bob\n        age: 25\n        active: false\n    '
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = 'users'
    var_26 = var_24.value[var_25]
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'first:\n  second: value'
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = len(var_29)
    var_32 = 1
    var_33 = var_31 - var_32



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '3.14'
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
    var_14 = 0
    var_15 = var_11.value[var_14]
    var_16 = 1
    var_17 = var_11.value[var_16]
    var_18 = 'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key'
    var_21 = var_19.value[var_20]
    var_22 = 'key:\n  nested: 42'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = var_23.value[var_20]
    var_25 = var_24.value
    var_26 = b'test'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '   \n  \t  \n'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'key: [unclosed'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'key: : value'
    var_35 = module_0.tokenize_yaml(var_34)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: not_an_int'
    var_2 = 'user:\n  name: Alice\n  age: 25'
    var_3 = 'items:\n  - 1\n  - 2\n  - 3'
    var_4 = 'optional: test'
    var_5 = b'name: Bob\nage: 40'
    var_6 = ''
    var_7 = 'name: John\n: invalid'
    var_8 = '"test string"'
    var_9 = 'active: true'
    var_10 = 'value: null'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = "name: John\nage: 'thirty'"
    var_3 = ''
    var_4 = b'name: Alice\nage: 25'
    var_5 = 'user:\n  name: Bob\n  details:\n    active: true'
    var_6 = 'items:\n  - 1\n  - 2\n  - 3'
    var_7 = "'simple string'"
    var_8 = 'name: John'



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello: world'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = 'hello: thisiswaytoolong'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = 100
    var_11 = module_0.Integer(minimum=var_9, maximum=var_10)
    var_12 = 'value: 42'
    var_13 = module_1.validate_yaml(var_12, var_11)
    var_14 = 'value: 150'
    var_15 = module_1.validate_yaml(var_14, var_11)
    var_16 = var_15[var_6]
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = '\n    name: John Doe\n    age: 30\n    '
    var_19 = '\n    name: This name is way too long to be valid according to our schema\n    age: 200\n    '
    var_20 = var_15[var_6]
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_15[var_6]
    var_23 = {error.code for error in var_22}
    var_24 = ''
    var_25 = '   \n  \t  \n'
    var_26 = '\n    name: John\n    age: thirty\n      extra: indent\n    '
    var_27 = b'name: Alice\nage: 25'
    var_28 = module_0.String()
    var_29 = module_0.String()
    var_30 = module_0.String()
    var_31 = '\n    name: Bob\n    address:\n      street: 123 Main St\n      city: Anytown\n    '
    var_32 = 'optional_field: test'
    var_33 = var_15[var_6]
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'required_field: null'
    var_36 = var_15[var_6]
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = module_0.Boolean()
    var_39 = 'active: true'
    var_40 = 'price: 19.99'
    var_41 = '\n    tags:\n      - python\n      - testing\n      - yaml\n    '



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: -5'
    var_2 = 'name: John\n  age: 25'
    var_3 = ''
    var_4 = b'name: Alice\nage: 30'
    var_5 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_6 = 5
    var_7 = 'short'
    var_8 = 'too_long'
    var_9 = 'items:\n  - apple\n  - banana\n  - cherry'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = "name: John\nage: 'thirty'"
    var_3 = ''
    var_4 = b'name: Alice\nage: 25'
    var_5 = 'user:\n  name: Bob\n  scores: [1, 2, 3]'
    var_6 = 'title'
    var_7 = 'title: Test'
    var_8 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    list:\n      - item1\n      - item2\n    '



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - swimming\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    string: hello\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2\n- item3'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'simple string'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '   \n  \t  \n'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'key: [unclosed list'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: : value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '\n    users:\n      - name: Alice\n        age: 25\n        active: true\n      - name: Bob\n        age: 30\n        active: false\n    settings:\n      theme: dark\n      notifications: true\n    '
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'users'
    var_25 = var_23.value[var_24]
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'first:\n  second: value'
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = len(var_27)
    var_30 = 1
    var_31 = var_29 - var_30
    var_32 = 'message: \'Hello, "world"!\''
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = '\n    description: |\n      This is a\n      multiline\n      string\n    '
    var_35 = module_0.tokenize_yaml(var_34)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: Johnathan'
    var_4 = 5
    var_5 = module_0.String(max_length=var_4)
    var_6 = '\n    name: Alice\n    age: 25\n    '
    var_7 = '\n    name: Bob\n    age: -5\n    '
    var_8 = ''
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'name: [unclosed list'
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = '\n    items:\n      - first\n      - second\n    '
    var_15 = 'items'
    var_16 = b'name: Charlie'
    var_17 = module_0.String()
    var_18 = 'active: true'
    var_19 = module_0.String()
    var_20 = 'value: null'
    var_21 = True
    var_22 = module_0.String()



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test: value'
    var_2 = 'string'
    var_3 = module_0.Field()
    assert var_3 == 1
    var_4 = module_1.validate_yaml(var_1, var_3)
    assert var_4 == 1
    var_5 = str(var_1)
    var_6 = 'yaml'
    var_7 = 'name: John'
    var_8 = 10
    var_9 = module_0.String(max_length=var_8)
    var_10 = 'name: John\n  age: 30'
    var_11 = module_0.String()
    var_12 = 'name: Alice\nage: 25'
    var_13 = 'name: Alice\nage: -5'
    var_14 = ''
    var_15 = b'name: Bob'
    var_16 = 'items:\n  - first\n  - second'
    var_17 = 'name: Jonathan\nage: 16'
    var_18 = 'name: null'
    var_19 = True
    var_20 = module_0.String()



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = '\n    name: John\n    age: 30\n    active: true\n    '
    var_2 = '\n    name: John\n    age: thirty  # not a number\n    active: true\n    '
    var_3 = '\n    name: JohnathanDoe\n    age: 30\n    active: true\n    '
    var_4 = '\n    name: JohnathanDoe\n    age: 200\n    active: maybe\n    '
    var_5 = ''
    var_6 = b'\n    name: Alice\n    age: 25\n    active: false\n    '
    var_7 = 5
    var_8 = module_0.String(max_length=var_7)
    var_9 = 'test_value'
    var_10 = '\n    name: John\n    age: 30\n      extra: indented\n    '
    var_11 = '   \n   \t   \n'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = "name: John\nage: 'thirty'"
    var_3 = ''
    var_4 = b'name: Alice\nage: 25'
    var_5 = 'title: Test\nmetadata:\n  author: Bob\n  version: 1.0'
    var_6 = 'name: John'
    var_7 = '\n    users:\n      - name: Alice\n        age: 25\n      - name: Bob\n        age: 30\n    '
    var_8 = 'users'
    var_9 = '\n    users:\n      - name: Alice\n        age: "twenty five"\n    '
    var_10 = 'name: null\nage: 30'
    var_11 = 'active: true'
    var_12 = 'price: 19.99'



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 'hello'
    var_4 = 'this_string_is_too_long'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = 100
    var_9 = module_0.Integer(minimum=var_7, maximum=var_8)
    var_10 = '42'
    var_11 = module_1.validate_yaml(var_10, var_9)
    assert var_11 == 42
    var_12 = '150'
    var_13 = module_1.validate_yaml(var_12, var_9)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = '\n    name: John Doe\n    age: 30\n    '
    var_16 = '\n    name: John Doe\n    '
    var_17 = len(var_13)
    assert var_17 == 1
    var_18 = '\n    name: John Doe\n    age: -5\n    '
    var_19 = len(var_13)
    assert var_19 == 1
    var_20 = ''
    var_21 = len(var_13)
    assert var_21 == 1
    var_22 = '\n    name: John Doe\n    age: [unclosed list\n    '
    var_23 = len(var_13)
    assert var_23 == 1
    var_24 = b'name: Alice\nage: 25'
    var_25 = module_0.String()
    var_26 = '\n    items:\n      - first\n      - second\n      - third\n    '
    var_27 = module_0.Boolean()
    var_28 = 'active: true'
    var_29 = 'value: null'
    var_30 = module_0.Float()
    var_31 = 'price: 19.99'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = 1
    var_3 = "name: John\nage: 'thirty'"
    var_4 = 0
    var_5 = ''
    var_6 = b'name: Alice\nage: 25'
    var_7 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_8 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_9 = 'name: John'
    var_10 = "'hello world'"
    var_11 = "'not a number'"



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - swimming\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = ''
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = b'key: value'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key: [unclosed list'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'simple string'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    users:\n      - name: Alice\n        age: 25\n      - name: Bob\n        age: 30\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'users'
    var_21 = var_19.value[var_20]
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = "message: 'Hello, World!'"
    var_24 = module_0.tokenize_yaml(var_23)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'name: abc'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = 'name: abcdef'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'name: John\nage: 25'
    var_10 = 0
    var_11 = var_5[var_10]
    var_12 = 'name: Johnathan\nage: -5'
    var_13 = var_5[var_6]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_5[var_6]
    var_16 = [error.code for error in var_15]
    var_17 = ''
    var_18 = module_1.validate_yaml(var_17, var_1)
    var_19 = '   \n  \t\n'
    var_20 = module_1.validate_yaml(var_19, var_1)
    var_21 = 'name: [1, 2'
    var_22 = module_1.validate_yaml(var_21, var_1)
    var_23 = b'name: test'
    var_24 = module_1.validate_yaml(var_23, var_1)
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25)
    var_27 = '- item1\n- item2\n- item3'
    var_28 = module_1.validate_yaml(var_27, var_26)
    var_29 = module_0.String()
    var_30 = module_0.String()
    var_31 = module_0.String()
    var_32 = 'name: Acme\naddress:\n  street: Main St\n  city: Metropolis'
    var_33 = 'name: John\nage: not_a_number'
    var_34 = var_28[var_6]
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = var_28[var_6][var_10]
    var_37 = 'position'
    var_38 = hasattr(var_36, var_37)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = "name: John\nage: 'thirty'"
    var_3 = ''
    var_4 = b'name: Alice\nage: 25'
    var_5 = 'user:\n  name: Bob\n  scores: [1, 2, 3]'
    var_6 = 'id'
    var_7 = 'active'
    var_8 = 'id: 123\nactive: true'
    var_9 = "id: 123\nactive: 'yes'"
    var_10 = 'items: [1, 2, 3]'
    var_11 = 'value: null'



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: Johnathan'
    var_4 = 5
    var_5 = module_0.String(max_length=var_4)
    var_6 = '\n    name: Alice\n    age: 25\n    '
    var_7 = '\n    name: Bob\n    age: -5\n    '
    var_8 = 'name: [unclosed list'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = module_0.String()
    var_15 = '\n    items: |\n      - item1\n      - item2\n    '
    var_16 = b'number: 42'
    var_17 = module_0.Integer()
    var_18 = 'active: true'
    var_19 = module_0.String()
    var_20 = 'value: null'
    var_21 = True
    var_22 = module_0.String()



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = "name: John\nage: 'thirty'"
    var_2 = ''
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'user:\n  name: Bob\n  email: bob@example.com'
    var_5 = 'hello'
    var_6 = 'name: John\n  age: 30'
    var_7 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_8 = 'active: true'
    var_9 = 'value: null'



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: -5'
    var_2 = 'name: John\n  age: 25'
    var_3 = ''
    var_4 = b'name: Alice\nage: 30'
    var_5 = 'items:\n  - apple\n  - banana\nconfig:\n  enabled: true'
    var_6 = 10
    var_7 = 100
    var_8 = module_0.Integer(minimum=var_6, maximum=var_7)
    var_9 = '50'
    var_10 = '5'
    var_11 = 'score: 95.5\nactive: true\nnickname: null'



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = '\n    name: John\n    age: 30\n    active: true\n    '
    var_2 = '\n    name: John\n    age: 30\n    active: true\n      extra: indented\n    '
    var_3 = '\n    name: JohnathanTooLong\n    age: 200\n    active: maybe\n    '
    var_4 = ''
    var_5 = b'\n    name: Alice\n    age: 25\n    active: false\n    '
    var_6 = 5
    var_7 = module_0.String(max_length=var_6)
    var_8 = 'name: Bob'
    var_9 = module_0.Integer()
    var_10 = '\n    id: 1\n    data:\n      name: Eve\n      age: 28\n      active: true\n    '



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: Johnathan'
    var_4 = 5
    var_5 = module_0.String(max_length=var_4)
    var_6 = '\n    name: Alice\n    age: 25\n    '
    var_7 = '\n    name: Bob\n    age: -5\n    '
    var_8 = 'name: [unclosed: list'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'name: Charlie'
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = module_0.String()
    var_18 = module_0.String()
    var_19 = '\n    name: David\n    address:\n      street: Main St\n      city: Boston\n    '
    var_20 = '\n    - apple\n    - banana\n    - cherry\n    '
    var_21 = module_0.String()
    var_22 = module_0.Array(var_21)
    var_23 = '\n    users:\n      - name: Eve\n        age: 30\n      - name: Frank\n        age: -1\n    '
    var_24 = module_0.String()
    var_25 = 0



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello: world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = var_1.value[var_2]
    var_4 = 'number: 42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'number'
    var_7 = var_5.value[var_6]
    var_8 = 'pi: 3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'pi'
    var_11 = var_9.value[var_10]
    var_12 = 'flag: true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'flag'
    var_15 = var_13.value[var_14]
    var_16 = 'empty: null'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'empty'
    var_19 = var_17.value[var_18]
    var_20 = '- item1\n- item2'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_21.value
    var_25 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = 'users'
    var_28 = var_26.value[var_27]
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 0
    var_32 = var_28.value[var_31]
    var_33 = b'key: value'
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = ''
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = '   \n  \t  \n'
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = 'key: [unclosed list'
    var_40 = module_0.tokenize_yaml(var_39)
    var_41 = 'key: : value'
    var_42 = module_0.tokenize_yaml(var_41)
    var_43 = 'first: line\nsecond: value'
    var_44 = module_0.tokenize_yaml(var_43)
    var_45 = '\n    config:\n      enabled: true\n      timeout: 30.5\n      retries: 3\n      servers:\n        - host: "server1"\n          port: 8080\n        - host: "server2"\n          port: 8081\n    '
    var_46 = module_0.tokenize_yaml(var_45)
    var_47 = 'config'
    var_48 = var_46.value[var_47]
    var_49 = 'servers'
    var_50 = var_48.value[var_49]
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 2



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: '
    var_2 = 'name: 123\nage: thirty'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = ''
    var_6 = b'name: Alice\nage: 25'
    var_7 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_8 = '42'
    var_9 = 'not_a_number'
    var_10 = 'numbers:\n  - 1\n  - 2\n  - 3'
    var_11 = 'name: John\n: invalid'
    var_12 = 'name: null\nage: 25'
    var_13 = 'flag: true\nactive: false'
    var_14 = 'price: 19.99\ntax: 0.07'



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - swimming\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    string: hello\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = b'key: value'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '   \n  \t  \n'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key: [unclosed list'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '- item1\n- item2'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'simple string'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '\n    users:\n      - name: Alice\n        age: 25\n        active: true\n      - name: Bob\n        age: 30\n        active: false\n    '
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'users'
    var_23 = var_21.value[var_22]
    var_24 = 'first:\n  second: value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = len(var_24)
    var_27 = 1
    var_28 = var_26 - var_27



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello: world'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = 'hello: thisiswaytoolong'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = 100
    var_11 = module_0.Integer(minimum=var_9, maximum=var_10)
    var_12 = 'value: 42'
    var_13 = module_1.validate_yaml(var_12, var_11)
    var_14 = 'value: 150'
    var_15 = module_1.validate_yaml(var_14, var_11)
    var_16 = var_15[var_6]
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = '\n    name: John Doe\n    age: 30\n    '
    var_19 = '\n    name: John Doe\n    '
    var_20 = var_15[var_6]
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = '\n    name: John Doe\n    age: "thirty"\n    '
    var_23 = var_15[var_6]
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = ''
    var_26 = '\n    name: John Doe\n    age: 30\n      extra: indented wrong\n    '
    var_27 = b'name: Jane Doe\nage: 25'
    var_28 = module_0.String()
    var_29 = module_0.String()
    var_30 = module_0.String()
    var_31 = '\n    name: Alice\n    address:\n      street: 123 Main St\n      city: Springfield\n    '
    var_32 = module_0.String()
    var_33 = module_0.Array(var_32)
    var_34 = '- item1\n- item2\n- item3'
    var_35 = module_1.validate_yaml(var_34, var_33)
    var_36 = '\n    active: true\n    value: null\n    '



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import yaml.scanner as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'parsed_token'
    var_2 = 'validated_value'
    var_3 = 'validated_value'
    var_4 = []
    var_5 = 'key: value'
    var_6 = 'test error'
    var_7 = module_0.ScannerError(var_6)
    var_8 = 'invalid: [yaml'
    var_9 = module_1.validate_yaml(var_8, var_4)
    var_10 = 'No content.'
    var_11 = ''
    var_12 = module_1.validate_yaml(var_11, var_4)
    var_13 = 'validated_value'
    var_14 = 'error'
    var_15 = [var_14]
    var_16 = b'key: value'
    var_17 = 'field'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = []
    var_21 = 'field: value'
    var_22 = 'field_value'
    var_23 = []



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 'key'
    var_5 = var_1.value[var_4]
    var_6 = 'number: 42'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'number'
    var_9 = var_7.value[var_8]
    var_10 = 'pi: 3.14'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'pi'
    var_13 = var_11.value[var_12]
    var_14 = 'flag: true'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'flag'
    var_17 = var_15.value[var_16]
    var_18 = 'nothing: null'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'nothing'
    var_21 = var_19.value[var_20]
    var_22 = '- item1\n- item2'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_23.value
    var_27 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = 'users'
    var_30 = var_28.value[var_29]
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = var_30.value
    var_34 = b'key: value'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = ''
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = '   \n  \t  \n'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = 'key: [unclosed list'
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = 'first: value\nsecond: item'
    var_43 = module_0.tokenize_yaml(var_42)
    var_44 = 'first'
    var_45 = var_43.value[var_44]
    var_46 = 'special: value with spaces & symbols!'
    var_47 = module_0.tokenize_yaml(var_46)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - hiking\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = var_8.value
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    string: hello\n    '
    var_12 = module_0.tokenize_yaml(var_11)
    var_13 = ''
    var_14 = module_0.tokenize_yaml(var_13)
    var_15 = b'key: value'
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'key: [unclosed list'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = '\n    users:\n      - name: Alice\n        age: 25\n        active: true\n      - name: Bob\n        age: 30\n        active: false\n    metadata:\n      count: 2\n      timestamp: 2023-01-01\n    '
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = 'users'
    var_22 = var_20.value[var_21]
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 'hello'
    var_4 = 'this_is_too_long'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 0
    var_7 = 100
    var_8 = module_0.Integer(minimum=var_6, maximum=var_7)
    var_9 = '42'
    var_10 = module_1.validate_yaml(var_9, var_8)
    assert var_10 == 42
    var_11 = '150'
    var_12 = module_1.validate_yaml(var_11, var_8)
    var_13 = 'name: John\nage: 30'
    var_14 = 'name: John'
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = module_0.String()
    var_18 = '\n    name: Alice\n    address:\n      street: 123 Main St\n      city: Boston\n    '
    var_19 = b'name: Bob\nage: 25'
    var_20 = ''
    var_21 = 'name: [unclosed: list'
    var_22 = module_0.String()
    var_23 = module_0.Array(var_22)
    var_24 = '- item1\n- item2\n- item3'
    var_25 = module_1.validate_yaml(var_24, var_23)
    var_26 = 'active: true\nvalue: null'
    var_27 = 'price: 19.99'
    var_28 = 'number'
    var_29 = module_0.Field()
    var_30 = module_1.validate_yaml(var_27, var_29)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - hiking\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hobbies'
    var_5 = var_3.value[var_4]
    var_6 = '\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    string: hello\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'integer'
    var_9 = var_7.value[var_8]
    var_10 = 'float'
    var_11 = var_7.value[var_10]
    var_12 = 'boolean'
    var_13 = var_7.value[var_12]
    var_14 = 'null_value'
    var_15 = var_7.value[var_14]
    var_16 = 'string'
    var_17 = var_7.value[var_16]
    var_18 = ''
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'key: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: [unclosed list'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '- item1\n- item2'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = '\n    users:\n      - name: Alice\n        age: 25\n        active: true\n      - name: Bob\n        age: 30\n        active: false\n    '
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'users'
    var_31 = var_29.value[var_30]
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 'just a string'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = '42'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'true'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = 'null'
    var_41 = module_0.tokenize_yaml(var_40)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = var_3.value
    var_5 = '3.14'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = var_6.value
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'items:\n  - name: test\n    value: 42'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'items'
    var_23 = var_21.value[var_22]
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = var_23.value[var_26]
    var_28 = b'test: value'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '   \n  \t  \n'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'key: [unclosed'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'multiline: |\n  line1\n  line2'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'base: &base\n  key: value\ncopy: *base'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = 'first:\n  second: value'
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = len(var_40)
    var_43 = 1
    var_44 = var_42 - var_43
    var_45 = 'first'
    var_46 = len(var_40)
    var_47 = var_46 - var_43



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = 1
    var_3 = "name: John\nage: 'thirty'"
    var_4 = ''
    var_5 = b'name: Alice\nage: 25'
    var_6 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_7 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_8 = 'Hello World'
    var_9 = '123'



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello: world'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = 'hello: thisiswaytoolong'
    var_5 = module_1.validate_yaml(var_4, var_1)
    var_6 = 1
    var_7 = var_5[var_6]
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = 100
    var_11 = module_0.Integer(minimum=var_9, maximum=var_10)
    var_12 = 'value: 42'
    var_13 = module_1.validate_yaml(var_12, var_11)
    var_14 = 'value: 150'
    var_15 = module_1.validate_yaml(var_14, var_11)
    var_16 = var_15[var_6]
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = '\n    name: John Doe\n    age: 30\n    '
    var_19 = var_15[var_9]
    var_20 = '\n    name: John Doe\n    '
    var_21 = var_15[var_6]
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = '\n    name: John Doe\n    age: -5\n    '
    var_24 = var_15[var_6]
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = module_0.String()
    var_27 = module_0.String()
    var_28 = module_0.String()
    var_29 = '\n    name: Alice\n    address:\n      street: 123 Main St\n      city: Springfield\n    '
    var_30 = b'value: test'
    var_31 = module_1.validate_yaml(var_30, var_1)
    var_32 = ''
    var_33 = module_1.validate_yaml(var_32, var_1)
    var_34 = 'invalid: [unclosed list'
    var_35 = module_1.validate_yaml(var_34, var_1)
    var_36 = '\n    first: value1\n    second: value2\n    '
    var_37 = module_1.validate_yaml(var_36, var_1)
    var_38 = module_0.Boolean()
    var_39 = 'flag: true'
    var_40 = module_1.validate_yaml(var_39, var_38)
    var_41 = 'flag: false'
    var_42 = module_1.validate_yaml(var_41, var_38)
    var_43 = module_0.Float()
    var_44 = 'value: 3.14'
    var_45 = module_1.validate_yaml(var_44, var_43)
    var_46 = 'value: null'
    var_47 = module_1.validate_yaml(var_46, var_1)
    var_48 = module_0.String()
    var_49 = module_0.Array(var_48)
    var_50 = 'items: [a, b, c]'
    var_51 = module_1.validate_yaml(var_50, var_49)
    var_52 = module_0.Integer()
    var_53 = '\n    id: 1\n    tags: [tag1, tag2]\n    metadata:\n      version: "1.0"\n    '



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: Johnathan'
    var_4 = 5
    var_5 = module_0.String(max_length=var_4)
    var_6 = '\n    name: Alice\n    age: 25\n    '
    var_7 = '\n    name: Bob\n    age: -5\n    '
    var_8 = 'name: [unclosed: list'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = module_0.String()
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = '\n    name: Charlie\n    address:\n      street: Main St\n      city: Metropolis\n    '
    var_18 = '\n    name: David\n    address:\n      street: ""\n      city: ""\n    '
    var_19 = 'active: true'
    var_20 = module_0.Boolean()
    var_21 = 'count: 42'
    var_22 = module_0.Integer()
    var_23 = b'name: Eve'
    var_24 = module_0.String()
    var_25 = '\n    - apple\n    - banana\n    - cherry\n    '
    var_26 = module_0.String()
    var_27 = module_0.Array(var_26)
    var_28 = '\n    users:\n      - name: Frank\n        age: 30\n        active: true\n      - name: Grace\n        age: 25\n        active: false\n    '
    var_29 = module_0.String()
    var_30 = module_0.Boolean()
    var_31 = 'users'



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test: value'
    var_2 = 'string'
    var_3 = module_0.Field()
    assert var_3 == 1
    var_4 = module_1.validate_yaml(var_1, var_3)
    var_5 = 'yaml'
    var_6 = 'name: John'
    var_7 = 10
    var_8 = module_0.String(max_length=var_7)
    var_9 = 'name: John\n  age: 30'
    var_10 = 0
    var_11 = 'mapping values are not allowed here'
    var_12 = ''
    var_13 = 'name: Johnathan'
    var_14 = module_0.String(max_length=var_7)
    var_15 = 'name: John\nage: 25'
    var_16 = 'name: Johnathan\nage: -5'
    var_17 = b'name: Alice'
    var_18 = 'users:\n  - Alice\n  - Bob'
    var_19 = 'users:\n  - Alice\n  - 123'



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = var_3.value
    var_5 = '3.14'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = var_6.value
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'list:\n  - item\n  - 42'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'list'
    var_23 = var_21.value[var_22]
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = b'test'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '   \n  '
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'key: [unclosed'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'key: value\n  indented: wrong'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = '\nstring: hello\ninteger: 42\nfloat: 3.14\nboolean: true\nnull_value: null\nlist:\n  - item1\n  - 2\ndict:\n  nested: value\n'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = var_37.value[var_22]
    var_39 = 'dict'
    var_40 = var_37.value[var_39]



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test: value'
    var_1 = 0
    var_2 = 10
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = 30
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'name: John\nage: 30'
    var_9 = 'name'
    var_10 = 'age'
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = []
    var_15 = (var_13, var_14)
    var_16 = 'name: John\nage: 30'
    var_17 = 5
    var_18 = 'invalid: [yaml'
    var_19 = ''
    var_20 = 8
    var_21 = 'test'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = 'test: value'
    var_25 = 'test'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = []
    var_29 = (var_27, var_28)
    var_30 = b'test: value'
    var_31 = 'not_a_number'
    var_32 = {var_28: var_31}
    var_33 = 'age: not_a_number'
    var_34 = 1
    var_35 = 0
    var_36 = 'age'
    var_37 = 'not_a_number'
    var_38 = {var_36: var_37}
    var_39 = (var_38, var_30)
    var_40 = 'age: not_a_number'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = "name: John\nage: 'thirty'"
    var_2 = 'name: John'
    var_3 = ''
    var_4 = b'name: Alice\nage: 25'
    var_5 = '\n    name: Bob\n    address:\n      street: Main St\n      city: Springfield\n    '
    var_6 = '\n    name: Bob\n    address:\n      street: Main St\n    '
    var_7 = 'name: John\nage: : 30'
    var_8 = 'name: John\nage: 30'
    var_9 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_10 = '   \n  \t\n'
    var_11 = '\n    name: Test\n    score: 95.5\n    active: true\n    tags:\n      - python\n      - testing\n      - yaml\n    '



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John Doe'
    var_1 = 100
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: '
    var_4 = 'A'
    var_5 = 101
    var_6 = var_4 * var_5
    var_7 = var_3 + var_6
    var_8 = module_0.String(max_length=var_1)
    var_9 = '\n    name: Alice\n    age: 30\n    '
    var_10 = '\n    name: \n    age: -5\n    '
    var_11 = 'name: [unclosed: list'
    var_12 = ''
    var_13 = b'name: Bob'
    var_14 = module_0.String()
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = '\n    name: Tech Corp\n    address:\n      street: 123 Main St\n      city: Metropolis\n    '
    var_18 = '\n    name: Tech Corp\n    address:\n      street: 123 Main St\n    '
    var_19 = module_0.Integer()
    var_20 = module_0.Array(var_19)
    var_21 = '\n    - 1\n    - 2\n    - 3\n    '
    var_22 = '\n    - 1\n    - "two"\n    - 3\n    '



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John'
    var_2 = "name: John\nage: 'thirty'"
    var_3 = 'user:\n  name: Alice\n  details: {}\n'
    var_4 = 'name: John\n  age: 30'
    var_5 = ''
    var_6 = b'name: Bob\nage: 25'
    var_7 = 'id'
    var_8 = 'active'
    var_9 = 'id: 123\nactive: true'
    var_10 = "items:\n  - 1\n  - 2\n  - 'three'"
    var_11 = '\n    users:\n      - id: 1\n        name: Alice\n        tags: [admin, user]\n      - id: 2\n        name: Bob\n        tags: [user]\n    '
    var_12 = 'users'



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test: value'
    assert var_0 == 1
    var_1 = 'name: John\nage: 25'
    var_2 = 'name: John\n  age: 25'
    var_3 = 0
    var_4 = 'could not find expected'
    var_5 = ''
    var_6 = 'name: Johnathan\nage: -5'
    var_7 = 'max_length'
    var_8 = 'minimum'
    var_9 = module_0.Boolean()
    var_10 = '\n    user:\n      name: Alice\n      age: 30\n    active: true\n    '
    var_11 = b'name: Bob\nage: 40'
    var_12 = '\n    items:\n      - apple\n      - banana\n      - cherry\n    counts:\n      - 1\n      - 2\n      - 3\n    '
    var_13 = 'title'
    var_14 = 'count'
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.Object(properties=var_17)
    var_19 = 'title: Test\ncount: 42'
    var_20 = 'name: null\nage: 30'



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'name: John'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = 'name: Johnathan'
    var_4 = 5
    var_5 = module_0.String(max_length=var_4)
    var_6 = '\n    name: Alice\n    age: 25\n    '
    var_7 = '\n    name: Bob\n    age: -5\n    '
    var_8 = 'name: [unclosed list'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'enabled: true'
    var_15 = module_0.Boolean()
    var_16 = module_0.String()
    var_17 = module_0.String()
    var_18 = module_0.String()
    var_19 = '\n    name: Charlie\n    address:\n      street: Main St\n      city: Metropolis\n    '
    var_20 = '\n    name: Dana\n    address:\n      street: ""\n      city: ""\n    '
    var_21 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_22 = module_0.String()
    var_23 = module_0.Array(var_22)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'list:\n  - item1\n  - item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'list'
    var_5 = var_3.value[var_4]
    var_6 = '\nstring: hello\ninteger: 42\nfloat: 3.14\nboolean: true\nnull_value: null\n'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'string'
    var_9 = var_7.value[var_8]
    var_10 = 'integer'
    var_11 = var_7.value[var_10]
    var_12 = 'float'
    var_13 = var_7.value[var_12]
    var_14 = 'boolean'
    var_15 = var_7.value[var_14]
    var_16 = 'null_value'
    var_17 = var_7.value[var_16]
    var_18 = b'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '   \n  \t  \n'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [unclosed list'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '- item1\n- item2'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'simple scalar'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '\nparent:\n  child:\n    grandchild: value\n  list:\n    - nested: item\n'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'parent'
    var_33 = var_31.value[var_32]
    var_34 = 'child'
    var_35 = var_33.value[var_34]
    var_36 = var_33.value[var_4]
    var_37 = 0
    var_38 = var_33.value[var_4]
    var_39 = var_38.value[var_37]
    var_40 = module_0.tokenize_yaml(var_24)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = var_1.value[var_2]
    var_4 = 'number: 42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'number'
    var_7 = var_5.value[var_6]
    var_8 = 'pi: 3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'pi'
    var_11 = var_9.value[var_10]
    var_12 = 'flag: true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'flag'
    var_15 = var_13.value[var_14]
    var_16 = 'empty: null'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'empty'
    var_19 = var_17.value[var_18]
    var_20 = '- item1\n- item2'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_21.value
    var_25 = 'list:\n  - a\n  - b'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = 'list'
    var_28 = var_26.value[var_27]
    var_29 = var_26.value[var_27]
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = b'key: value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = ''
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = '   \n  \t  \n'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'key: [unclosed'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = 'outer:\n  inner:\n    key: value'
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = len(var_40)
    var_43 = 1
    var_44 = var_42 - var_43
    var_45 = 'outer'
    var_46 = var_41.value[var_45]



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = "name: John\nage: 'thirty'"
    var_2 = 'name: John'
    var_3 = 'name: John\n  age: 30'
    var_4 = ''
    var_5 = '\n    user:\n      name: Alice\n      address:\n        city: London\n        country: UK\n    '
    var_6 = 'items: [1, 2, 3]'
    var_7 = "items: [1, 'two', 3]"
    var_8 = b'name: Bob\nage: 25'
    var_9 = 'title'
    var_10 = 10
    var_11 = 'title: Hello World'
    var_12 = '\n    string: hello\n    integer: 42\n    float_num: 3.14\n    boolean: true\n    null_value: null\n    '
    var_13 = '\n    users:\n      - id: 0\n        email: not-an-email\n      - id: 2\n        email: valid@example.com\n      - id: -1\n        email: another@bad\n    '



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John'
    var_3 = 'name: John\nage: [30'
    var_4 = ''
    var_5 = '\n    user:\n      name: Alice\n      address:\n        street: Main St\n        city: Boston\n    '
    var_6 = 'numbers: [1, 2, 3]'
    var_7 = 'active: true'
    var_8 = 'optional: null'
    var_9 = b'name: Bob\nage: 25'
    var_10 = '\n    name: VeryLongNameThatExceedsLimit\n    age: -5\n    email: not-an-email\n    '
    var_11 = 'title'
    var_12 = 'count'
    var_13 = 'title: Test\ncount: 42'



# Parsed testcases at query #22
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = 'test: value'
    var_1 = 'No content.'
    var_2 = 'no_content'
    var_3 = 1
    var_4 = 0
    var_5 = module_0.Position(var_3, var_3, var_4)
    var_6 = ''
    var_7 = 'test'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = []
    var_11 = 'test: value'
    var_12 = 'Field required.'
    var_13 = 'required'
    var_14 = 1
    var_15 = 0
    var_16 = module_0.Position(var_14, var_14, var_15)
    var_17 = module_0.ParseError(text=var_12, code=var_13, position=var_16)
    var_18 = 'Invalid type.'
    var_19 = 'type'
    assert var_19 == 2
    var_20 = 2
    var_21 = 10
    var_22 = module_0.Position(var_20, var_14, var_21)
    var_23 = module_0.ParseError(text=var_18, code=var_19, position=var_22)
    var_24 = [var_17, var_23]
    var_25 = None
    var_26 = 'invalid: yaml'
    var_27 = 'Scanner error.'
    var_28 = 'parse_error'
    var_29 = 1
    var_30 = 0
    var_31 = module_0.Position(var_29, var_29, var_30)
    var_32 = 'invalid: [yaml'
    var_33 = len(var_20)
    assert var_33 == 1
    var_34 = 'bytes'
    var_35 = 'test'
    var_36 = {var_34: var_35}
    var_37 = []
    var_38 = b'bytes: test'
    var_39 = 'name'
    var_40 = 'age'
    var_41 = 'John'
    var_42 = 30
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = []
    var_45 = 'name: John\nage: 30'
    var_46 = 'string value'
    var_47 = []
    var_48 = 'string value'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: :'
    var_2 = 1
    var_3 = "name: John\nage: 'thirty'"
    var_4 = ''
    var_5 = b'name: Alice\nage: 25'
    var_6 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_7 = "'test string'"
    var_8 = '123'
    var_9 = '   \n  \n'



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'number: 42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'pi: 3.14'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'flag: true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'empty: null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'users:\n  - name: Alice\n    age: 30'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'users'
    var_15 = var_13.value[var_14]
    var_16 = b'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '   \n  \t  \n'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: [unclosed list'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'simple string'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '\n    server:\n      host: localhost\n      port: 8080\n      ssl: true\n    '
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '- string\n- 123\n- true\n- null'
    var_29 = module_0.tokenize_yaml(var_28)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    outer:\n      inner:\n        - item1\n        - item2\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'outer'
    var_5 = var_3.value[var_4]
    var_6 = 'inner'
    var_7 = var_3.value[var_4]
    var_8 = var_7.value[var_6]
    var_9 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = 'string'
    var_12 = var_10.value[var_11]
    var_13 = 'integer'
    var_14 = var_10.value[var_13]
    var_15 = 'float'
    var_16 = var_10.value[var_15]
    var_17 = 'boolean'
    var_18 = var_10.value[var_17]
    var_19 = 'null_value'
    var_20 = var_10.value[var_19]
    var_21 = ''
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = b'key: value'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = 'key: [unclosed list'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = 'users'
    var_30 = var_28.value[var_29]
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 0
    var_34 = var_30.value[var_33]



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'test: value'
    assert var_0 == 1
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = 'name: John\nage: 30'
    var_4 = 'name: John\nage: not_a_number'
    var_5 = 'string'
    var_6 = 5
    var_7 = module_0.Field()
    var_8 = '"hello"'
    var_9 = '"toolong"'
    var_10 = ''
    var_11 = 'name: John\n  age: 30'
    var_12 = b'name: Alice\nage: 25'
    var_13 = 'user:\n  name: Bob\n  age: 40'
    var_14 = 'items:\n  - 1\n  - 2\n  - 3'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = var_3.value
    var_5 = '3.14'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = var_6.value
    var_8 = 'true'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '- item1\n- item2'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_15.value
    var_19 = 'key: value'
    var_20 = module_0.tokenize_yaml(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'key'
    var_24 = var_20.value[var_23]
    var_25 = 'key:\n  nested: value'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = var_26.value[var_23]
    var_28 = b'test: value'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '   \n  \t  \n'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'key: [unclosed'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = '\n    name: John\n    age: 30\n    active: true\n    scores: [95.5, 87.0, 92.3]\n    metadata:\n      created: 2023-01-01\n      tags: [python, testing]\n    '
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'scores'
    var_39 = var_37.value[var_38]
    var_40 = var_37.value[var_38]
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 3
    var_43 = 'metadata'
    var_44 = var_37.value[var_43]



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  age: 30'
    var_2 = 1
    var_3 = "name: John\nage: 'thirty'"
    var_4 = ''
    var_5 = b'name: Alice\nage: 25'
    var_6 = 'user:\n  name: Bob\n  age: 40\nactive: true'
    var_7 = 'items:\n  - apple\n  - banana\n  - cherry'
    var_8 = "'hello world'"
    var_9 = '42'



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = str(var_1)
    assert var_4 == 1
    var_5 = 'yaml'
    var_6 = module_0.String()
    var_7 = 'name: John\nage: 25'
    var_8 = 'name: John\n  age: 25'
    var_9 = 0
    var_10 = ''
    var_11 = 'name: Johnathan\nage: 25'
    var_12 = module_0.String()
    var_13 = 'user:\n  name: John\n  age: 25\nactive: true'
    var_14 = b'name: John\nage: 25'
    var_15 = 5
    var_16 = module_0.String(max_length=var_15)
    var_17 = 'test'
    var_18 = 'toolong'
    var_19 = module_0.String()
    var_20 = 'items:\n  - item1\n  - item2\n  - item3'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'validated_value'
    var_2 = []
    var_3 = 'validated_value'
    var_4 = []
    var_5 = 'key: value'
    var_6 = None
    var_7 = 'error1'
    var_8 = 'error2'
    var_9 = [var_7, var_8]
    var_10 = 'invalid: yaml'
    var_11 = 'bytes_value'
    var_12 = []
    var_13 = b'bytes: input'
    var_14 = 'No content.'
    var_15 = 'no_content'
    var_16 = ''
    var_17 = 'parse error.'
    var_18 = 'parse_error'
    var_19 = 'invalid yaml content'



