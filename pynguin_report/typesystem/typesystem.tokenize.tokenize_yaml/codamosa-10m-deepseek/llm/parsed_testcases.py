####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John Doe\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John Doe\n    age: : 30\n    '
    var_5 = module_0.tokenize_yaml(var_4)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: Johnathan\nage: 30'
    var_2 = 'name: John\nage: 200'
    var_3 = 'name: John\nage: thirty'
    var_4 = ''
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #3
#--------------------------


import base64 as module_1


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'key: [value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = b'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'utf-8'
    var_9 = module_1.decode(var_8)
    var_10 = '\n    key1:\n      key2: value2\n      key3:\n        - item1\n        - item2\n    '
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key1'
    var_13 = var_11.value[var_12]
    var_14 = 'key2'
    var_15 = var_11.value[var_12]
    var_16 = var_15.value[var_14]
    var_17 = 'key3'
    var_18 = var_11.value[var_12]
    var_19 = var_18.value[var_17]
    var_20 = var_11.value[var_12]
    var_21 = var_20.value[var_17]
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = '\n    int: 42\n    float: 3.14\n    bool: true\n    null: null\n    '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'int'
    var_27 = var_25.value[var_26]
    var_28 = 'float'
    var_29 = var_25.value[var_28]
    var_30 = 'bool'
    var_31 = var_25.value[var_30]
    var_32 = 'null'
    var_33 = var_25.value[var_32]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John Doe\nage: 200'
    var_3 = ''
    var_4 = 'name: John\nage: 25\ncity: New York'
    var_5 = 'name: John'
    var_6 = 'person:\n  name: John\n  age: 25'
    var_7 = '- name: Apple\n  quantity: 5\n- name: Banana\n  quantity: 10'
    var_8 = True
    var_9 = '- name: Apple\n  quantity: 0\n- name: Banana\n  quantity: 10'
    var_10 = b'name: John\nage: 25'
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #5
#--------------------------



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
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    name: John\n    age: 30\n    hobbies:\n      - reading\n      - hiking\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'hobbies'
    var_21 = var_19.value[var_20]
    var_22 = b'key: value'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [unclosed list'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: John Doe'
    var_2 = 1
    var_3 = ''
    var_4 = 'name: John\nage: 25'
    var_5 = 'person:\n  name: John'
    var_6 = 'names:\n  - John\n  - Jane'
    var_7 = 'name: 123'
    var_8 = 'age: 25'
    var_9 = module_0.String()
    var_10 = 'name: John\nage: 25'
    var_11 = 'name: John\nage:'
    var_12 = 'All test cases passed!'
    var_13 = print(var_12)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: '
    var_2 = 'name: Johnathan\nage: 25'
    var_3 = 'name: John\nage: 200'
    var_4 = 'name: Johnathan\nage: 200'
    var_5 = ''
    var_6 = b'name: John\nage: 25'
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    name: John\n    age: 30\n    '
    var_1 = '\n    name: John\n    age: thirty\n    '
    var_2 = '\n    name: John Doe\n    age: 200\n    '
    var_3 = ''
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #9
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
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'users'
    var_21 = var_19.value[var_20]
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = b'key: value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: [unclosed list'
    var_27 = module_0.tokenize_yaml(var_26)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    outer:\n      inner: value\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '\n    - item1\n    - item2\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '3.14'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'key: [value'
    var_17 = module_0.tokenize_yaml(var_16)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = '\n    name: John Doe\n    age: 30\n    city: New York\n    '
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = len(var_2)
    var_5 = 1
    var_6 = var_4 - var_5
    var_7 = '\n    name: John\n    age: 30\n    - item1\n    '
    var_8 = module_0.tokenize_yaml(var_7)
    var_9 = '\n    person:\n      name: John\n      age: 30\n      hobbies:\n        - reading\n        - swimming\n    '
    var_10 = module_0.tokenize_yaml(var_9)
    var_11 = 'person'
    var_12 = var_10.value[var_11]
    var_13 = 'hobbies'
    var_14 = var_12.value[var_13]
    var_15 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_16 = module_0.tokenize_yaml(var_15)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = var_3.value
    var_5 = b'key: value'
    var_6 = module_0.tokenize_yaml(var_5)
    var_7 = var_6.value
    var_8 = 'key: [value'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '\n    key1:\n      key2: value2\n      key3:\n        - item1\n        - item2\n    '
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = var_11.value
    var_13 = 'key1'
    var_14 = var_11.value[var_13]
    var_15 = 'key3'
    var_16 = var_11.value[var_13][var_15]
    var_17 = '\n    int: 42\n    float: 3.14\n    bool: true\n    null: null\n    '
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = var_18.value
    var_20 = '\n    - item1\n    - item2\n    '
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = var_21.value
    var_23 = '{}'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = var_24.value
    var_26 = '[]'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = var_27.value
    var_29 = '\n    ---\n    key1: value1\n    ---\n    key2: value2\n    '
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = var_30.value
    var_32 = 'All test cases passed!'
    var_33 = print(var_32)



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 25\n    '
    var_1 = '\n    name: John Doe\n    age: twenty-five\n    '
    var_2 = '\n    name: John Doe\n    age: -5\n    '
    var_3 = ''
    var_4 = '\n    name: John Doe\n    age: 25\n    city: New York\n    '
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #17
#--------------------------



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
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'key:\n  nested: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key'
    var_23 = var_21.value[var_22]
    var_24 = 'key: ['
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = ''
    var_5 = '\n    person:\n      name: John Doe\n      age: 30\n    '
    var_6 = 'person'
    var_7 = '\n    people:\n      - name: John Doe\n        age: 30\n      - name: Jane Smith\n        age: 25\n    '
    var_8 = 'people'
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '- item1\n- item2'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'scalar'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '123'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '123.45'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'key: ['
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = '\n    name: John Doe\n    age: 30\n    city: New York\n    '
    var_7 = '\n    person:\n      name: John Doe\n      age: 30\n    '
    var_8 = 'person'
    var_9 = '\n    names:\n      - John Doe\n      - Jane Smith\n    '
    var_10 = 'names'
    var_11 = '\n    active: true\n    '
    var_12 = 'active'
    var_13 = '\n    value: null\n    '
    var_14 = 'value'
    var_15 = None
    var_16 = '\n    price: 9.99\n    '
    var_17 = 'price'
    var_18 = 'All test cases passed!'
    var_19 = print(var_18)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_2
import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'name: John\nage: twenty-five'
    var_5 = 'integer'
    var_6 = module_0.Field()
    var_7 = module_1.validate_yaml(var_4, var_6)
    var_8 = ''
    var_9 = module_0.Field()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'person:\n  name: John\n  age: 25'
    var_12 = 'object'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_11, var_13)
    var_15 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_16 = 'array'
    var_17 = module_0.Field()
    var_18 = module_1.validate_yaml(var_15, var_17)
    var_19 = 'is_valid: true'
    var_20 = 'boolean'
    var_21 = module_0.Field()
    var_22 = module_1.validate_yaml(var_19, var_21)
    var_23 = 'value: null'
    var_24 = 'null'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_23, var_25)
    var_27 = 'pi: 3.14'
    var_28 = 'number'
    var_29 = module_0.Field()
    var_30 = module_1.validate_yaml(var_27, var_29)
    var_31 = 'count: 10'
    var_32 = module_0.Field()
    var_33 = module_1.validate_yaml(var_31, var_32)
    var_34 = 'name: John\nage: 25\ncity: New York'
    var_35 = 'name'
    var_36 = 'age'
    var_37 = 'city'
    var_38 = module_0.Field()
    var_39 = module_0.Field()
    var_40 = module_0.Field()
    var_41 = {var_35: var_38, var_36: var_39, var_37: var_40}
    var_42 = module_2.Schema(var_41)
    var_43 = module_1.validate_yaml(var_34, var_42)
    var_44 = 'All test cases passed!'
    var_45 = print(var_44)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: John\nage: 200'
    var_3 = 1
    var_4 = ''
    var_5 = 'name: John\nage: 30\ncity: New York'
    var_6 = 'name: John'
    var_7 = 'name: 123\nage: 30'
    var_8 = module_0.String()
    var_9 = module_0.String()
    var_10 = module_0.String()
    var_11 = 'name: John\naddress:\n  street: 123 Main St\n  city: New York'
    var_12 = module_0.String()
    var_13 = 'items:\n  - name: Apple\n    quantity: 5\n  - name: Orange\n    quantity: 3'
    var_14 = 'name: John\naddress:\n  street: 123 Main St\n  city: 123'
    var_15 = 'All test cases passed!'
    var_16 = print(var_15)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'string'
    var_3 = module_0.Field()
    var_4 = 'integer'
    var_5 = module_0.Field()
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'name: John\nage: 30'
    var_8 = 'name: John'
    var_9 = 1
    var_10 = 'name: John\nage: thirty'
    var_11 = ''
    var_12 = 'person'
    var_13 = 'object'
    var_14 = module_0.Field()
    var_15 = module_0.Field()
    var_16 = {var_0: var_14, var_1: var_15}
    var_17 = module_0.Field()
    var_18 = {var_12: var_17}
    var_19 = 'person:\n  name: Alice\n  age: 25'
    var_20 = 'numbers'
    var_21 = 'array'
    var_22 = module_0.Field()
    var_23 = module_0.Field()
    var_24 = {var_20: var_23}
    var_25 = 'numbers:\n  - 1\n  - 2\n  - 3'
    var_26 = 'numbers:\n  - 1\n  - two\n  - 3'
    var_27 = 'email'
    var_28 = module_0.Field()
    var_29 = module_0.Field()
    var_30 = module_0.Field()
    var_31 = {var_0: var_28, var_1: var_29, var_27: var_30}
    var_32 = 'name: 123\nage: thirty\nemail: invalid-email'
    var_33 = "name: John O'Connor\nage: 30"
    var_34 = 'active'
    var_35 = 'description'
    var_36 = 'boolean'
    var_37 = module_0.Field()
    var_38 = True
    var_39 = module_0.Field(allow_null=var_38)
    var_40 = {var_34: var_37, var_35: var_39}
    var_41 = 'active: true\ndescription: null'
    var_42 = 'All tests passed!'
    var_43 = print(var_42)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'key:\n  nested: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '- item1\n- item2'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '123'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '123.456'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'key: value\n  nested: value'
    var_17 = module_0.tokenize_yaml(var_16)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 25\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: twenty-five\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = True
    var_7 = '\n    name: John Doe\n    age: 25\n    city: New York\n    '
    var_8 = '\n    person:\n      name: John Doe\n      age: 25\n    '
    var_9 = 'person'
    var_10 = '\n    names:\n      - John Doe\n      - Jane Smith\n    '
    var_11 = 'names'
    var_12 = '\n    names:\n      - John Doe\n      - 25\n    '
    var_13 = '\n    active: true\n    '
    var_14 = 'active'
    var_15 = '\n    value: null\n    '
    var_16 = 'value'
    var_17 = 'All test cases passed!'
    var_18 = print(var_17)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    assert var_3 == 'John Doe'
    var_4 = '\n    name: John Doe\n    age: thirty\n    '
    var_5 = 'integer'
    var_6 = module_0.Field()
    var_7 = module_1.validate_yaml(var_4, var_6)
    var_8 = ''
    var_9 = module_0.Field()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = '\n    person:\n      name: John Doe\n      age: 30\n    '
    var_12 = 'object'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_11, var_13)
    var_15 = '\n    - item1\n    - item2\n    - item3\n    '
    var_16 = 'array'
    var_17 = module_0.Field()
    var_18 = module_1.validate_yaml(var_15, var_17)
    var_19 = '\n    enabled: true\n    '
    var_20 = 'boolean'
    var_21 = module_0.Field()
    var_22 = module_1.validate_yaml(var_19, var_21)
    assert var_22 is True
    var_23 = '\n    value: null\n    '
    var_24 = 'null'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_23, var_25)
    assert var_26 is None
    var_27 = '\n    price: 9.99\n    '
    var_28 = 'number'
    var_29 = module_0.Field()
    var_30 = module_1.validate_yaml(var_27, var_29)
    var_31 = '\n    quantity: 10\n    '
    var_32 = module_0.Field()
    var_33 = module_1.validate_yaml(var_31, var_32)
    assert var_33 == 10
    var_34 = '\n    message: Hello, World!\n    '
    var_35 = module_0.Field()
    var_36 = module_1.validate_yaml(var_34, var_35)
    assert var_36 == 'Hello, World!'
    var_37 = 'All test cases passed!'
    var_38 = print(var_37)



# Parsed testcases at query #29
#--------------------------




# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '\n            name: John Doe\n            age: 30\n            '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John Doe'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '\n            name: John Doe\n            age: thirty\n            '
    var_7 = ''
    var_8 = '\n            person:\n              name: John Doe\n              age: 30\n            '
    var_9 = 'person'
    var_10 = {var_1: var_3, var_2: var_4}
    var_11 = {var_9: var_10}
    var_12 = '\n            people:\n              - name: John Doe\n                age: 30\n              - name: Jane Smith\n                age: 25\n            '
    var_13 = 'people'
    var_14 = {var_1: var_3, var_2: var_4}
    var_15 = 'Jane Smith'
    var_16 = 25
    var_17 = {var_1: var_15, var_2: var_16}
    var_18 = [var_14, var_17]
    var_19 = {var_13: var_18}
    var_20 = 0
    var_21 = [var_2]

def test_case_0():
    var_0 = '\n            name: John Doe\n            age: 30\n            '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John Doe'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '\n            name: John Doe\n            age: thirty\n            '
    var_7 = ''
    var_8 = '\n            person:\n              name: John Doe\n              age: 30\n            '
    var_9 = 'person'
    var_10 = {var_1: var_3, var_2: var_4}
    var_11 = {var_9: var_10}
    var_12 = '\n            people:\n              - name: John Doe\n                age: 30\n              - name: Jane Smith\n                age: 25\n            '
    var_13 = 'people'
    var_14 = {var_1: var_3, var_2: var_4}
    var_15 = 'Jane Smith'
    var_16 = 25
    var_17 = {var_1: var_15, var_2: var_16}
    var_18 = [var_14, var_17]
    var_19 = {var_13: var_18}
    var_20 = 0
    var_21 = [var_2]



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = ''
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    key1:\n      key2: value2\n      key3:\n        - item1\n        - item2\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'key: value\n  invalid_indentation: error'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'special: "value with \\"quotes\\""'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '---\ndoc1: value1\n---\ndoc2: value2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '\n    anchor: &anchor\n      key: value\n    alias: *anchor\n    '
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '!!str string_value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'All test cases passed!'
    var_21 = print(var_20)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = 1
    var_5 = ''
    var_6 = b'\n    name: John Doe\n    age: 30\n    '
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: twenty-five'
    var_2 = 1
    var_3 = ''
    var_4 = 'person:\n  name: John\n  age: 25\nhobbies:\n  - reading\n  - swimming'
    var_5 = 'person:\n  name: John\n  age: twenty-five\nhobbies:\n  - reading\n  - swimming'
    var_6 = 'name: John'
    var_7 = 'name: John\nage: 25\ncity: New York'
    var_8 = 'name: John\nage: 25\ninvalid_yaml: ['
    var_9 = 'hobbies: []'
    var_10 = 'person: {}'
    var_11 = 'name: null\nage: 25'
    var_12 = 'name: John\nage: true'
    var_13 = 'name: John\nage: 25.5'
    var_14 = "name: John\nage: '25'"
    var_15 = 'name: 123\nage: 25'
    var_16 = 'numbers: [1, 2, 3]'
    var_17 = "numbers: ['1', '2', '3']"
    var_18 = "numbers: [1, '2', 3]"
    var_19 = 'numbers: [[1, 2], [3, 4]]'
    var_20 = 'data:\n  name: John\n  age: 25'
    var_21 = 'person:\n  name: John\n  age: 25\nhobbies:\n  - reading\n  - swimming'
    var_22 = 'person:\n  name: John\n  age: twenty-five\nhobbies:\n  - reading\n  - swimming'
    var_23 = 'person:\n  name: John\nhobbies:\n  - reading\n  - swimming'
    var_24 = 'person:\n  name: John\n  age: 25\n  city: New York\nhobbies:\n  - reading\n  - swimming'
    var_25 = 'person:\n  name: John\n  age: 25\nhobbies: []'
    var_26 = 'person: {}\nhobbies:\n  - reading\n  - swimming'
    var_27 = 'person:\n  name: null\n  age: 25\nhobbies:\n  - reading\n  - swimming'
    var_28 = 'person:\n  name: John\n  age: true\nhobbies:\n  - reading\n  - swimming'



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe\nage: 25'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John Doe\nage: twenty-five'
    var_4 = ''
    var_5 = 'name: John Doe'
    var_6 = True
    var_7 = 'name: John Doe\nage: 25\ncity: New York'
    var_8 = 'All test cases passed!'
    var_9 = print(var_8)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 25\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: twenty-five\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = True
    var_7 = '\n    name: John Doe\n    age: 25\n    city: New York\n    '
    var_8 = '\n    person:\n      name: John Doe\n      age: 25\n    '
    var_9 = 'person'
    var_10 = '\n    names:\n      - John Doe\n      - Jane Smith\n    '
    var_11 = 'names'
    var_12 = '\n    active: true\n    '
    var_13 = 'active'
    var_14 = '\n    price: 9.99\n    '
    var_15 = 'price'
    var_16 = '\n    value: null\n    '
    var_17 = 'value'
    var_18 = None
    var_19 = 'All tests passed!'
    var_20 = print(var_19)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'name: John\nage: twenty-five'
    var_5 = 'integer'
    var_6 = module_0.Field()
    var_7 = module_1.validate_yaml(var_4, var_6)
    var_8 = ''
    var_9 = module_0.Field()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'person:\n  name: John\n  age: 25'
    var_12 = 'object'
    var_13 = 'name'
    var_14 = 'age'
    var_15 = module_0.Field()
    var_16 = module_0.Field()
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.Field()
    var_19 = module_1.validate_yaml(var_11, var_18)
    var_20 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_21 = 'array'
    var_22 = module_0.Field()
    var_23 = module_0.Field()
    var_24 = module_1.validate_yaml(var_20, var_23)
    var_25 = 'is_active: true'
    var_26 = 'boolean'
    var_27 = module_0.Field()
    var_28 = module_1.validate_yaml(var_25, var_27)
    var_29 = 'value: null'
    var_30 = 'null'
    var_31 = module_0.Field()
    var_32 = module_1.validate_yaml(var_29, var_31)
    var_33 = 'price: 9.99'
    var_34 = 'number'
    var_35 = module_0.Field()
    var_36 = module_1.validate_yaml(var_33, var_35)
    var_37 = 'quantity: 10'
    var_38 = module_0.Field()
    var_39 = module_1.validate_yaml(var_37, var_38)
    var_40 = 'name: John\nage: 25\ncity: New York'
    var_41 = 'city'
    var_42 = module_0.Field()
    var_43 = module_0.Field()
    var_44 = module_0.Field()
    var_45 = {var_13: var_42, var_14: var_43, var_41: var_44}
    var_46 = module_0.Field()
    var_47 = module_1.validate_yaml(var_40, var_46)
    var_48 = 'name: John'
    var_49 = module_0.Field()
    var_50 = module_0.Field()
    var_51 = {var_13: var_49, var_14: var_50}
    var_52 = [var_14]
    var_53 = module_0.Field()
    var_54 = module_1.validate_yaml(var_48, var_53)
    var_55 = 'name: John\nage: 25\ncity: New York'
    var_56 = module_0.Field()
    var_57 = module_0.Field()
    var_58 = {var_13: var_56, var_14: var_57}
    var_59 = module_0.Field()
    var_60 = module_1.validate_yaml(var_55, var_59)
    var_61 = 'person:\n  name: John\n  age: 25'
    var_62 = 'person'
    var_63 = module_0.Field()
    var_64 = module_0.Field()
    var_65 = {var_13: var_63, var_14: var_64}
    var_66 = module_0.Field()
    var_67 = {var_62: var_66}
    var_68 = module_0.Field()
    var_69 = module_1.validate_yaml(var_61, var_68)
    var_70 = 'people:\n  - name: John\n    age: 25\n  - name: Jane\n    age: 30'
    var_71 = 'people'
    var_72 = module_0.Field()
    var_73 = module_0.Field()
    var_74 = {var_13: var_72, var_14: var_73}
    var_75 = module_0.Field()
    var_76 = module_0.Field()
    var_77 = {var_71: var_76}
    var_78 = module_0.Field()
    var_79 = module_1.validate_yaml(var_70, var_78)
    var_80 = 'name: John\nage: twenty-five'
    var_81 = module_0.Field()
    var_82 = module_0.Field()
    var_83 = {var_13: var_81, var_14: var_82}
    var_84 = module_0.Field()
    var_85 = module_1.validate_yaml(var_80, var_84)
    var_86 = 'person:\n  name: John\n  age: twenty-five'
    var_87 = module_0.Field()
    var_88 = module_0.Field()
    var_89 = {var_13: var_87, var_14: var_88}
    var_90 = module_0.Field()
    var_91 = {var_62: var_90}
    var_92 = module_0.Field()
    var_93 = module_1.validate_yaml(var_86, var_92)
    var_94 = 'fruits:\n  - apple\n  - 123\n  - orange'
    var_95 = 'fruits'
    var_96 = module_0.Field()
    var_97 = module_0.Field()
    var_98 = {var_95: var_97}
    var_99 = module_0.Field()
    var_100 = module_1.validate_yaml(var_94, var_99)
    var_101 = 'is_active: yes'
    var_102 = 'is_active'
    var_103 = module_0.Field()
    var_104 = {var_102: var_103}
    var_105 = module_0.Field()
    var_106 = module_1.validate_yaml(var_101, var_105)
    var_107 = 'value: none'
    var_108 = 'value'
    var_109 = module_0.Field()
    var_110 = {var_108: var_109}
    var_111 = module_0.Field()
    var_112 = module_1.validate_yaml(var_107, var_111)
    var_113 = 'price: 9.99.99'
    var_114 = 'price'
    var_115 = module_0.Field()
    var_116 = {var_114: var_115}
    var_117 = module_0.Field()
    var_118 = module_1.validate_yaml(var_113, var_117)
    var_119 = 'quantity: 10.5'
    var_120 = 'quantity'
    var_121 = module_0.Field()
    var_122 = {var_120: var_121}
    var_123 = module_0.Field()
    var_124 = module_1.validate_yaml(var_119, var_123)
    var_125 = 'name: John\nage: twenty-five'
    var_126 = module_0.Field()
    var_127 = module_0.Field()
    var_128 = {var_13: var_126, var_14: var_127}
    var_129 = [var_14]
    var_130 = module_0.Field()
    var_131 = module_1.validate_yaml(var_125, var_130)
    var_132 = 'name: John\nage: 25\ncity: 123'
    var_133 = module_0.Field()
    var_134 = module_0.Field()
    var_135 = {var_13: var_133, var_14: var_134}
    var_136 = module_0.Field()
    var_137 = module_1.validate_yaml(var_132, var_136)
    var_138 = 'person:\n  name: John\n  age: twenty-five'
    var_139 = module_0.Field()
    var_140 = module_0.Field()
    var_141 = {var_13: var_139, var_14: var_140}
    var_142 = [var_14]
    var_143 = module_0.Field()
    var_144 = {var_62: var_143}
    var_145 = module_0.Field()
    var_146 = module_1.validate_yaml(var_138, var_145)
    var_147 = 'person:\n  name: John\n  age: 25\n  city: 123'
    var_148 = module_0.Field()
    var_149 = module_0.Field()
    var_150 = {var_13: var_148, var_14: var_149}
    var_151 = module_0.Field()
    var_152 = {var_62: var_151}
    var_153 = module_0.Field()
    var_154 = module_1.validate_yaml(var_147, var_153)
    var_155 = 'people:\n  - name: John\n    age: twenty-five\n  - name: Jane\n    age: 30'



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John Doe\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John Doe\n    age: 30\n    - item1\n    '
    var_5 = module_0.tokenize_yaml(var_4)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    name: John\n    age: 25\n    '
    var_1 = '\n    name: John\n    age: twenty-five\n    '
    var_2 = 1
    var_3 = '\n    name: John Doe\n    age: -5\n    '
    var_4 = ''
    var_5 = '\n    name: John\n    age: 25\n    city: New York\n    '
    var_6 = '\n    name: John\n    '
    var_7 = 'All test cases passed!'
    var_8 = print(var_7)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'name: John Doe\nage: thirty'
    var_5 = 'integer'
    var_6 = module_0.Field()
    var_7 = module_1.validate_yaml(var_4, var_6)
    var_8 = ''
    var_9 = module_0.Field()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'person:\n  name: John Doe\n  age: 30'
    var_12 = 'object'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_11, var_13)
    var_15 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_16 = 'array'
    var_17 = module_0.Field()
    var_18 = module_1.validate_yaml(var_15, var_17)
    var_19 = 'is_active: true'
    var_20 = 'boolean'
    var_21 = module_0.Field()
    var_22 = module_1.validate_yaml(var_19, var_21)
    var_23 = 'value: null'
    var_24 = 'null'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_23, var_25)
    var_27 = 'price: 9.99'
    var_28 = 'number'
    var_29 = module_0.Field()
    var_30 = module_1.validate_yaml(var_27, var_29)
    var_31 = 'name: John Doe\nage: 30\ninvalid'
    var_32 = module_0.Field()
    var_33 = module_1.validate_yaml(var_31, var_32)
    var_34 = 'name: John Doe'
    var_35 = 'name'
    var_36 = 'age'
    var_37 = [var_35, var_36]
    var_38 = module_0.Field()
    var_39 = module_1.validate_yaml(var_34, var_38)
    var_40 = 'All test cases passed!'
    var_41 = print(var_40)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 'name: Johnathan\nage: 30'
    var_3 = ''
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = 'name: John Doe\nage: thirty'
    var_5 = 'integer'
    var_6 = module_0.Field()
    var_7 = module_1.validate_yaml(var_4, var_6)
    var_8 = ''
    var_9 = module_0.Field()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'person:\n  name: John Doe\n  age: 30'
    var_12 = 'object'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_11, var_13)
    var_15 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_16 = 'array'
    var_17 = module_0.Field()
    var_18 = module_1.validate_yaml(var_15, var_17)
    var_19 = 'is_valid: true'
    var_20 = 'boolean'
    var_21 = module_0.Field()
    var_22 = module_1.validate_yaml(var_19, var_21)
    var_23 = 'value: null'
    var_24 = 'null'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_23, var_25)
    var_27 = 'price: 9.99'
    var_28 = 'number'
    var_29 = module_0.Field()
    var_30 = module_1.validate_yaml(var_27, var_29)
    var_31 = 'quantity: 10'
    var_32 = module_0.Field()
    var_33 = module_1.validate_yaml(var_31, var_32)
    var_34 = 'name: John Doe\nage: 30\ncity: New York'
    var_35 = module_0.Field()
    var_36 = module_1.validate_yaml(var_34, var_35)
    var_37 = 'name: John Doe\nage: thirty'
    var_38 = module_0.Field()
    var_39 = module_1.validate_yaml(var_37, var_38)
    var_40 = 'name: John Doe'
    var_41 = 'age'
    var_42 = [var_41]
    var_43 = module_0.Field()
    var_44 = module_1.validate_yaml(var_40, var_43)
    var_45 = 'name: John Doe\nage: 30\ncity: New York'
    var_46 = 'name'
    var_47 = module_0.Field()
    var_48 = module_0.Field()
    var_49 = {var_46: var_47, var_41: var_48}
    var_50 = module_0.Field()
    var_51 = module_1.validate_yaml(var_45, var_50)
    var_52 = 'person:\n  name: John Doe\n  age: 30'
    var_53 = 'person'
    var_54 = module_0.Field()
    var_55 = module_0.Field()
    var_56 = {var_46: var_54, var_41: var_55}
    var_57 = module_0.Field()
    var_58 = {var_53: var_57}
    var_59 = module_0.Field()
    var_60 = module_1.validate_yaml(var_52, var_59)
    var_61 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_62 = 'fruits'
    var_63 = module_0.Field()
    var_64 = module_0.Field()
    var_65 = {var_62: var_64}
    var_66 = module_0.Field()
    var_67 = module_1.validate_yaml(var_61, var_66)
    var_68 = 'is_valid: true'
    var_69 = 'is_valid'
    var_70 = module_0.Field()
    var_71 = {var_69: var_70}
    var_72 = module_0.Field()
    var_73 = module_1.validate_yaml(var_68, var_72)
    var_74 = 'value: null'
    var_75 = 'value'
    var_76 = module_0.Field()
    var_77 = {var_75: var_76}
    var_78 = module_0.Field()
    var_79 = module_1.validate_yaml(var_74, var_78)
    var_80 = 'price: 9.99'
    var_81 = 'price'
    var_82 = module_0.Field()
    var_83 = {var_81: var_82}
    var_84 = module_0.Field()
    var_85 = module_1.validate_yaml(var_80, var_84)
    var_86 = 'quantity: 10'
    var_87 = 'quantity'
    var_88 = module_0.Field()
    var_89 = {var_87: var_88}
    var_90 = module_0.Field()
    var_91 = module_1.validate_yaml(var_86, var_90)
    var_92 = 'name: John Doe\nage: 30\ncity: New York'
    var_93 = 'city'
    var_94 = module_0.Field()
    var_95 = module_0.Field()
    var_96 = module_0.Field()
    var_97 = {var_46: var_94, var_41: var_95, var_93: var_96}
    var_98 = module_0.Field()
    var_99 = module_1.validate_yaml(var_92, var_98)
    var_100 = 'person:\n  name: John Doe\n  age: thirty'
    var_101 = module_0.Field()
    var_102 = module_0.Field()
    var_103 = {var_46: var_101, var_41: var_102}
    var_104 = module_0.Field()
    var_105 = {var_53: var_104}
    var_106 = module_0.Field()
    var_107 = module_1.validate_yaml(var_100, var_106)
    var_108 = 'fruits:\n  - apple\n  - 123\n  - orange'
    var_109 = module_0.Field()
    var_110 = module_0.Field()
    var_111 = {var_62: var_110}
    var_112 = module_0.Field()
    var_113 = module_1.validate_yaml(var_108, var_112)
    var_114 = 'is_valid: yes'
    var_115 = module_0.Field()
    var_116 = {var_69: var_115}
    var_117 = module_0.Field()
    var_118 = module_1.validate_yaml(var_114, var_117)
    var_119 = 'value: none'
    var_120 = module_0.Field()
    var_121 = {var_75: var_120}
    var_122 = module_0.Field()
    var_123 = module_1.validate_yaml(var_119, var_122)
    var_124 = 'price: 9.99 dollars'
    var_125 = module_0.Field()
    var_126 = {var_81: var_125}
    var_127 = module_0.Field()
    var_128 = module_1.validate_yaml(var_124, var_127)
    var_129 = 'quantity: 10.5'
    var_130 = module_0.Field()
    var_131 = {var_87: var_130}
    var_132 = module_0.Field()
    var_133 = module_1.validate_yaml(var_129, var_132)
    var_134 = 'name: John Doe\nage: thirty\ncity: New York'
    var_135 = module_0.Field()
    var_136 = module_0.Field()
    var_137 = module_0.Field()
    var_138 = {var_46: var_135, var_41: var_136, var_93: var_137}
    var_139 = module_0.Field()
    var_140 = module_1.validate_yaml(var_134, var_139)
    var_141 = 'person:\n  name: John Doe'
    var_142 = module_0.Field()
    var_143 = module_0.Field()
    var_144 = {var_46: var_142, var_41: var_143}
    var_145 = module_0.Field()
    var_146 = {var_53: var_145}
    var_147 = [var_53]
    var_148 = module_0.Field()
    var_149 = module_1.validate_yaml(var_141, var_148)
    var_150 = 'person:\n  name: John Doe\n  age: 30\n  city: New York'
    var_151 = module_0.Field()
    var_152 = module_0.Field()
    var_153 = {var_46: var_151, var_41: var_152}
    var_154 = module_0.Field()
    var_155 = {var_53: var_154}
    var_156 = module_0.Field()
    var_157 = module_1.validate_yaml(var_150, var_156)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'name: John Doe\nage: thirty'
    var_2 = 'name: John Doe\nage: 200'
    var_3 = ''
    var_4 = 'name: John Doe'
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'name: John\nage: twenty-five'
    var_4 = ''
    var_5 = 'name: John'
    var_6 = True
    var_7 = 'name: John\nage: 25\ncity: New York'
    var_8 = 'All unit tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 25\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: twenty-five\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = True
    var_7 = '\n    name: John Doe\n    age: 25\n    city: New York\n    '
    var_8 = 'All test cases pass'
    var_9 = print(var_8)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'name: John Doe'
    var_2 = 1
    var_3 = 'name: John Doe\nage: thirty'
    var_4 = ''
    var_5 = 'person:\n  name: John Doe\n  age: 30'
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John Doe Too Long\nage: -5'
    var_3 = ''
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'name: John Doe\nage: thirty'
    var_2 = 'name: John Doe\nage: 200'
    var_3 = ''
    var_4 = 'name: John Doe'
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: [value'
    var_7 = module_0.tokenize_yaml(var_6)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = True
    var_7 = '\n    name: John Doe\n    age: 30\n    city: New York\n    '
    var_8 = 'All test cases passed!'
    var_9 = print(var_8)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = '\n    name: John\n    age: 30\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = ''
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John\n    age: 30\n    - item1\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '\n    person:\n      name: John\n      age: 30\n      hobbies:\n        - reading\n        - swimming\n    '
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '\n    name: John\n    age: 30\n    height: 1.75\n    is_student: false\n    address: null\n    '
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John Doe\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John Doe\n    age: 30\n      extra: invalid\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = 'name: John\nage: twenty-five'
    var_4 = module_0.Field()
    var_5 = module_1.validate_yaml(var_3, var_4)
    var_6 = ''
    var_7 = module_0.Field()
    var_8 = module_1.validate_yaml(var_6, var_7)
    var_9 = 'person:\n  name: John\n  age: 25'
    var_10 = module_0.Field()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = 'is_active: true'
    var_16 = module_0.Field()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = 'value: null'
    var_19 = module_0.Field()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'price: 9.99'
    var_22 = module_0.Field()
    var_23 = module_1.validate_yaml(var_21, var_22)
    var_24 = 'quantity: 10'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_24, var_25)
    var_27 = 'name: John\nage: 25\ncity: New York'
    var_28 = module_0.Field()
    var_29 = module_1.validate_yaml(var_27, var_28)
    var_30 = 'message: "Hello, world!"'
    var_31 = module_0.Field()
    var_32 = module_1.validate_yaml(var_30, var_31)
    var_33 = 'name: 张三\nage: 30'
    var_34 = module_0.Field()
    var_35 = module_1.validate_yaml(var_33, var_34)
    var_36 = 'description: Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
    var_37 = module_0.Field()
    var_38 = module_1.validate_yaml(var_36, var_37)
    var_39 = 'name: '
    var_40 = module_0.Field()
    var_41 = module_1.validate_yaml(var_39, var_40)
    var_42 = 'age: 25'
    var_43 = module_0.Field()
    var_44 = module_1.validate_yaml(var_42, var_43)
    var_45 = 'name: John\nage: 25\ncity: New York\ncountry: USA'
    var_46 = module_0.Field()
    var_47 = module_1.validate_yaml(var_45, var_46)
    var_48 = 'name: John\nname: Jane'
    var_49 = module_0.Field()
    var_50 = module_1.validate_yaml(var_48, var_49)
    var_51 = 'name: John\nage: 25\ncity: New York\n'
    var_52 = module_0.Field()
    var_53 = module_1.validate_yaml(var_51, var_52)
    var_54 = 'name: John  \nage: 25  '
    var_55 = module_0.Field()
    var_56 = module_1.validate_yaml(var_54, var_55)
    var_57 = '  name: John\n  age: 25'
    var_58 = module_0.Field()
    var_59 = module_1.validate_yaml(var_57, var_58)
    var_60 = 'name:\tJohn\nage:\t25'
    var_61 = module_0.Field()
    var_62 = module_1.validate_yaml(var_60, var_61)
    var_63 = 'name: John\n  age: 25'
    var_64 = module_0.Field()
    var_65 = module_1.validate_yaml(var_63, var_64)
    var_66 = '# This is a comment\nname: John\nage: 25'
    var_67 = module_0.Field()
    var_68 = module_1.validate_yaml(var_66, var_67)
    var_69 = '---\nname: John\nage: 25\n---\nname: Jane\nage: 30'
    var_70 = module_0.Field()
    var_71 = module_1.validate_yaml(var_69, var_70)
    var_72 = 'person: &person\n  name: John\n  age: 25\nanother_person: *person'
    var_73 = module_0.Field()
    var_74 = module_1.validate_yaml(var_72, var_73)
    var_75 = 'people:\n  - name: John\n    age: 25\n  - name: Jane\n    age: 30'
    var_76 = module_0.Field()
    var_77 = module_1.validate_yaml(var_75, var_76)
    var_78 = 'message: "Hello \\"world\\"!"'
    var_79 = module_0.Field()
    var_80 = module_1.validate_yaml(var_78, var_79)
    var_81 = 'description: |\n  This is a\n  multiline\n  string.'
    var_82 = module_0.Field()
    var_83 = module_1.validate_yaml(var_81, var_82)
    var_84 = 'description: >\n  This is a\n  folded\n  string.'
    var_85 = module_0.Field()
    var_86 = module_1.validate_yaml(var_84, var_85)
    var_87 = 'description: |-\n  This is a\n  literal\n  string.'
    var_88 = module_0.Field()
    var_89 = module_1.validate_yaml(var_87, var_88)
    var_90 = 'description: >-\n  This is a\n  block scalar\n  string.'
    var_91 = module_0.Field()
    var_92 = module_1.validate_yaml(var_90, var_91)
    var_93 = 'description: "This is a flow scalar string."'
    var_94 = module_0.Field()
    var_95 = module_1.validate_yaml(var_93, var_94)
    var_96 = '!!str name: John\n!!int age: 25'
    var_97 = module_0.Field()
    var_98 = module_1.validate_yaml(var_96, var_97)
    var_99 = '!my_tag name: John\n!my_tag age: 25'
    var_100 = module_0.Field()
    var_101 = module_1.validate_yaml(var_99, var_100)
    var_102 = 'name: !!str John\nage: !!int 25'
    var_103 = module_0.Field()
    var_104 = module_1.validate_yaml(var_102, var_103)
    var_105 = 'name: John\nage: 25'
    var_106 = module_0.Field()
    var_107 = module_1.validate_yaml(var_105, var_106)
    var_108 = 'name: Café\nage: 25'
    var_109 = module_0.Field()
    var_110 = module_1.validate_yaml(var_108, var_109)
    var_111 = 'message: Hello 👋'
    var_112 = module_0.Field()
    var_113 = module_1.validate_yaml(var_111, var_112)
    var_114 = 'name: John\x00Doe\nage: 25'
    var_115 = module_0.Field()
    var_116 = module_1.validate_yaml(var_114, var_115)



# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'name: John\nage: 25'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = 'name: John\nage: twenty-five'
    var_4 = module_0.Field()
    var_5 = module_1.validate_yaml(var_3, var_4)
    var_6 = ''
    var_7 = module_0.Field()
    var_8 = module_1.validate_yaml(var_6, var_7)
    var_9 = 'person:\n  name: John\n  age: 25'
    var_10 = module_0.Field()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'fruits:\n  - apple\n  - banana\n  - orange'
    var_13 = module_0.Field()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = 'flag: true\nvalue: null'
    var_16 = module_0.Field()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = 'count: 10\nprice: 9.99'
    var_19 = module_0.Field()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'message: "Hello, world!"'
    var_22 = module_0.Field()
    var_23 = module_1.validate_yaml(var_21, var_22)
    var_24 = '---\nname: John\n---\nage: 25'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_24, var_25)
    var_27 = 'name: John\nage: 25\ninvalid'
    var_28 = module_0.Field()
    var_29 = module_1.validate_yaml(var_27, var_28)
    var_30 = 'All test cases passed!'
    var_31 = print(var_30)



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------




# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
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
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'key:\n  subkey: subvalue'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key'
    var_23 = var_21.value[var_22]
    var_24 = b'hello'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = ': invalid'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'All tests passed!'
    var_29 = print(var_28)



# Parsed testcases at query #30
#--------------------------




# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: '
    var_2 = 'name: John Doe Long Name\nage: 200'
    var_3 = ''
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = 'name'
    var_2 = 'age'
    var_3 = '\n    name: John Doe\n    age: thirty\n    '
    var_4 = ''
    var_5 = '\n    name: John Doe\n    '
    var_6 = True
    var_7 = '\n    name: John Doe\n    age: 30\n    city: New York\n    '
    var_8 = 'All unit tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #34
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John\n    age: 30\n    invalid\n    '
    var_5 = module_0.tokenize_yaml(var_4)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\nage: thirty'
    var_2 = 1
    var_3 = 'name: John Doe Smith\nage: 30'
    var_4 = ''
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #36
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '\n    name: John Doe\n    age: 30\n    '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '\n    name: John Doe\n    age: 30\n    invalid: [1, 2, 3\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = b'name: John Doe\nage: 30'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'Hello, World!'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'null'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #37
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: value'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '- item1\n- item2'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'scalar'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '123'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '123.45'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'key: ['
    var_17 = module_0.tokenize_yaml(var_16)



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'name: John Doe\nage: 30'
    var_1 = 'string'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_0, var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 1
    var_6 = var_3[var_5]
    var_7 = 'name: John Doe\nage: thirty'
    var_8 = module_1.validate_yaml(var_7, var_2)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_8[var_5]
    var_11 = var_8[var_5]
    var_12 = 'parse_error'
    var_13 = ''
    var_14 = module_1.validate_yaml(var_13, var_2)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_14[var_5]
    var_17 = var_14[var_5]
    var_18 = 'no_content'
    var_19 = b'name: John Doe\nage: 30'
    var_20 = module_1.validate_yaml(var_19, var_2)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_20[var_5]
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



