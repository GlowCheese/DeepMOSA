####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'parent:\n  child: value'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'count: 42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'count'
    var_15 = var_13.value[var_14]
    var_16 = 'price: 19.99'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'price'
    var_19 = var_17.value[var_18]
    var_20 = 'active: true'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'active'
    var_23 = var_21.value[var_22]
    var_24 = 'value: null'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'value'
    var_27 = var_25.value[var_26]
    var_28 = 'simple string'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '{ invalid yaml :'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '- item\n  bad_indent: value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = '\nusers:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\n'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'users'
    var_37 = var_35.value[var_36]
    var_38 = 'key1: value1\nkey2: value2\nkey3: value3'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = var_39.value
    var_41 = len(var_40)
    assert var_41 == 3
    var_42 = 'name: José'
    var_43 = 'utf-8'
    var_44 = module_1.encode(var_43)
    var_45 = module_0.tokenize_yaml(var_44)
    var_46 = 'items:\n  - 1\n  - string\n  - true\n  - null'
    var_47 = module_0.tokenize_yaml(var_46)
    var_48 = 'items'
    var_49 = var_47.value[var_48]
    var_50 = var_47.value[var_48]
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 4



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: : syntax:'
    var_5 = ''
    var_6 = 5
    var_7 = module_0.String(max_length=var_6)
    var_8 = 'toolongstring'
    var_9 = module_0.Integer()
    var_10 = '42'
    var_11 = module_0.Integer()
    var_12 = module_0.Array(var_11)
    var_13 = '- 1\n- 2\n- 3'
    var_14 = 'user:\n  name: Bob\n  age: 35'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'invalid: yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'hello'
    var_16 = module_0.String()
    var_17 = 'null'
    var_18 = True
    var_19 = module_0.String()
    var_20 = 'true'
    var_21 = module_0.Boolean()
    var_22 = '3.14'
    var_23 = module_0.Float()



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = '42'
    var_2 = module_0.Integer()
    var_3 = 'name: John\nage: 30'
    var_4 = module_0.Field()
    var_5 = module_0.Integer()
    var_6 = '- 1\n- 2\n- 3'
    var_7 = module_0.Field()
    var_8 = b'key: value'
    var_9 = module_0.Field()
    var_10 = 'invalid: [yaml: content'
    var_11 = module_0.Field()
    var_12 = ''
    var_13 = module_0.Field()
    var_14 = '   \n  \n  '
    var_15 = module_0.Field()
    var_16 = 'true'
    var_17 = module_0.Field()
    var_18 = 'null'
    var_19 = module_0.Field()
    var_20 = '3.14'
    var_21 = module_0.Field()
    var_22 = 'users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30'
    var_23 = module_0.Field()



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = b'test'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'invalid: [yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = 'name: VeryLongName'
    var_16 = '- item1\n- item2\n- item3'
    var_17 = module_0.String()
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = module_0.String()
    var_22 = module_0.String()
    var_23 = module_0.String()
    var_24 = 'name: Alice\naddress:\n  street: Main St\n  city: NYC'



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = 'invalid: yaml: content:'
    var_6 = module_0.String()
    var_7 = 'parse_error'
    var_8 = 'mapping values'
    var_9 = 'hello'
    var_10 = module_0.Integer()
    var_11 = '- item1\n- item2\n- item3'
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = 'users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30'
    var_15 = 'users'
    var_16 = 'name'
    var_17 = 'age'
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.Object(properties=var_20)
    var_22 = module_0.Array(var_21)
    var_23 = {var_15: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = b'value: 123'
    var_26 = 'value'
    var_27 = module_0.Integer()
    var_28 = {var_26: var_27}
    var_29 = module_0.Object(properties=var_28)
    var_30 = 'enabled: true\ndisabled: false'
    var_31 = 'enabled'
    var_32 = 'disabled'
    var_33 = module_0.Boolean()
    var_34 = module_0.Boolean()
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Object(properties=var_35)
    var_37 = 'value: null'
    var_38 = True
    var_39 = module_0.String()
    var_40 = {var_26: var_39}
    var_41 = module_0.Object(properties=var_40)
    var_42 = 'price: 19.99'
    var_43 = 'price'
    var_44 = module_0.Number()
    var_45 = {var_43: var_44}
    var_46 = module_0.Object(properties=var_45)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = '42'
    var_2 = module_0.Integer()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = 'invalid: yaml: content:'
    var_6 = module_0.Integer()
    var_7 = module_1.validate_yaml(var_5, var_6)
    var_8 = 'hello'
    var_9 = module_0.Integer()
    var_10 = '[1, 2, 3]'
    var_11 = module_0.Integer()
    var_12 = module_0.Array(var_11)
    var_13 = 'key1: value1\nkey2: value2'
    var_14 = {}
    var_15 = module_0.Object(properties=var_14)
    var_16 = b'name: test'
    var_17 = module_0.Field()
    var_18 = 'value: null'
    var_19 = 'flag: true'
    var_20 = module_0.Field()
    var_21 = '3.14'
    var_22 = module_0.Float()



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = b'value: test'
    var_6 = module_0.String()
    var_7 = 'invalid: [yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = '   \n  \t  '
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = 'not_a_number'
    var_17 = module_0.Integer()
    var_18 = '\n    users:\n      - name: Alice\n        age: 25\n      - name: Bob\n        age: 30\n    '
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = 'value: null'
    var_22 = 'enabled: true'
    var_23 = module_0.Boolean()
    var_24 = 'price: 19.99'
    var_25 = module_0.Float()



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: syntax: here:'
    var_10 = module_0.String()
    var_11 = b'hello'
    var_12 = module_0.String()
    var_13 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_14 = module_0.String()
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'true'
    var_22 = module_0.String()
    var_23 = 'null'
    var_24 = module_0.String()
    var_25 = '3.14'
    var_26 = module_0.String()



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'hello'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'name: John\nage: 30'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '   \n  \n  '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '{invalid: [yaml: content}'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '\n    users:\n      - name: Alice\n        age: 25\n      - name: Bob\n        age: 30\n    '
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'test: value'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'name: José'
    var_33 = 'utf-8'
    var_34 = module_1.encode(var_33)
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = '\n    description: |\n      This is a\n      multiline string\n    '
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = '- item1\n- item2\n- item3'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = var_39.value
    var_41 = len(var_40)
    assert var_41 == 3



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = module_0.String()
    var_5 = 'hello'
    var_6 = 'name: John\n  invalid: [unclosed'
    var_7 = ''
    var_8 = 'name: VeryLongName'
    var_9 = module_0.Integer()
    var_10 = '42'
    var_11 = 'items:\n  - apple\n  - banana'



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: syntax:'
    var_10 = module_0.String()
    var_11 = 'not_a_number'
    var_12 = module_0.Integer()
    var_13 = b'hello'
    var_14 = module_0.String()
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30'
    var_22 = module_0.String()
    var_23 = 'null'
    var_24 = True
    var_25 = module_0.String()
    var_26 = 'true'
    var_27 = module_0.String()



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = b'world'
    var_3 = module_0.String()
    var_4 = '42'
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = 'name: John\nage: 30'
    var_9 = '{ invalid: yaml: syntax'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \n  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '- one\n- two\n- three'
    var_19 = module_0.String()
    var_20 = 'true'
    var_21 = module_0.String()
    var_22 = 'null'
    var_23 = module_0.String()



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = 'name: John\nage: 30'
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = '- item1\n- item2'
    var_6 = module_0.String()
    var_7 = module_0.Array(var_6)
    var_8 = ''
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = 'invalid: [yaml: content'
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'test: value'
    var_15 = module_0.String()
    var_16 = '42'
    var_17 = 0
    var_18 = 'null'
    var_19 = True
    var_20 = module_0.String()
    var_21 = 'true'
    var_22 = module_0.Boolean()
    var_23 = '3.14'
    var_24 = module_0.Number()
    var_25 = 'users:\n  - name: Alice\n  - name: Bob'
    var_26 = module_0.String()
    var_27 = 'users'



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = '[1, 2, 3]'
    var_6 = module_0.Integer()
    var_7 = module_0.Array(var_6)
    var_8 = '{ invalid yaml: [}'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'hello'
    var_15 = module_0.String()
    var_16 = 'name: VeryLongName'
    var_17 = '\n    string_val: "text"\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_18 = module_0.Object()



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'key:\n  nested: value\n  items: [1, 2]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'enabled: true\ndisabled: false'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'value: null'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'pi: 3.14'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: [unclosed'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key:\n\tinvalid'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = module_0.tokenize_yaml(var_6)
    var_25 = 'first: 1\nsecond: 2\nthird: 3'
    var_26 = module_0.tokenize_yaml(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = 'users'
    var_32 = var_30.value[var_31]
    var_33 = len(var_32)
    assert var_33 == 2



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = '{ invalid yaml: [}'
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = ''
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = '   \n  \t  '
    var_17 = module_0.String()
    var_18 = module_1.validate_yaml(var_16, var_17)
    var_19 = b'test value'
    var_20 = module_0.String()
    var_21 = 'null'
    var_22 = True
    var_23 = module_0.String()
    var_24 = 'true'
    var_25 = module_0.Boolean()
    var_26 = '3.14'
    var_27 = module_0.Float()



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'hello'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'items:\n  - name: item1\n    value: 1\n  - name: item2\n    value: 2'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'items'
    var_27 = var_25.value[var_26]
    var_28 = 'key: [invalid'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '  key1: value1\nkey2: value2'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'key1: value1\nkey2: value2'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = module_0.tokenize_yaml(var_30)
    var_35 = 'key: "value with spaces"'
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = '{}'
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = '[]'
    var_40 = module_0.tokenize_yaml(var_39)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = b'test_value'
    var_6 = module_0.String()
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = '123'
    var_14 = 2
    var_15 = module_0.String(max_length=var_14)
    var_16 = '[1, 2, 3]'
    var_17 = module_0.Integer()
    var_18 = module_0.Array(var_17)
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'true'
    var_23 = module_0.Boolean()
    var_24 = '3.14'
    var_25 = module_0.Float()



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1
import base64 as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = b'test'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \t  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '[1, 2, 3]'
    var_19 = module_0.Integer()
    var_20 = module_0.Array(var_19)
    var_21 = 'null'
    var_22 = True
    var_23 = module_0.String()
    var_24 = 'true'
    var_25 = module_0.Boolean()
    var_26 = '3.14'
    var_27 = module_0.Float()
    var_28 = '名前: 太郎'
    var_29 = 'utf-8'
    var_30 = module_2.encode(var_29)
    var_31 = module_0.String()



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = '\nname: John\nage: 30\n'
    var_7 = 'invalid: yaml: syntax:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = 'not_a_number'
    var_14 = module_0.Integer()
    var_15 = '\n- item1\n- item2\n- item3\n'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = b'test_value'
    var_19 = module_0.String()
    var_20 = '   \n  \n  '
    var_21 = module_0.String()
    var_22 = module_1.validate_yaml(var_20, var_21)
    var_23 = module_0.String()
    var_24 = module_0.String()
    var_25 = module_0.String()
    var_26 = '\nname: Alice\naddress:\n  street: Main St\n  city: Springfield\n'



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'name: John\nage: 30'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2\n- item3'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'items:\n  - first\n  - second'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '42'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'false'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'null'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'hello world'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: [invalid'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '\nusers:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30\nsettings:\n  debug: true\n  timeout: 60\n'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'users'
    var_31 = var_29.value[var_30]
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'key: value'
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = 'name: José'
    var_36 = 'utf-8'
    var_37 = module_1.encode(var_36)
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = 'description: |\n  This is a\n  multiline string'
    var_40 = module_0.tokenize_yaml(var_39)
    var_41 = 'message: "Hello World"'
    var_42 = module_0.tokenize_yaml(var_41)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'count: 42'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'value: 3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'enabled: true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'value: null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = b'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '   \n  \n  '
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'key: value: invalid:'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value\n  list:\n    - item1\n    - item2'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'line1: value1\nline2: value2'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'message: "Hello: World!"'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'greeting: こんにちは'
    var_29 = 'utf-8'
    var_30 = module_1.encode(var_29)
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '"123"'
    var_33 = module_0.tokenize_yaml(var_32)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = '42'
    var_2 = module_0.Integer()
    var_3 = 'hello'
    var_4 = module_0.String()
    var_5 = 'name: John\nage: 30'
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = '[1, 2, 3]'
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = '{ invalid yaml: [unclosed'
    var_12 = module_0.String()
    var_13 = b'test_value'
    var_14 = module_0.String()
    var_15 = 'null'
    var_16 = True
    var_17 = module_0.String()
    var_18 = 'true'
    var_19 = module_0.Boolean()
    var_20 = '3.14'
    var_21 = module_0.Float()
    var_22 = 'not_a_number'
    var_23 = module_0.Integer()



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = module_0.String()
    var_8 = 'John'
    var_9 = 'name: John'
    var_10 = '\n    name: John\n    age: 30\n    items:\n      - item1\n      - item2\n    '
    var_11 = '\n    int_val: 42\n    float_val: 3.14\n    bool_val: true\n    null_val: null\n    '
    var_12 = module_0.Field()



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: [yaml: content'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'hello world'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_19 = module_0.String()
    var_20 = 'active: true'
    var_21 = module_0.String()
    var_22 = 'value: null'
    var_23 = module_0.String()



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\n  invalid syntax: [unclosed'
    var_5 = ''
    var_6 = 10
    var_7 = module_0.String(max_length=var_6)
    var_8 = 'John'
    var_9 = 'name: VeryLongName'
    var_10 = '\nitems:\n  - id: 1\n    name: Item1\n  - id: 2\n    name: Item2\n'
    var_11 = module_0.Integer()
    var_12 = module_0.String()
    var_13 = 'items'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\nage: not_a_number'
    var_3 = '42'
    var_4 = 'not_a_number'
    var_5 = 'items:\n  - item1\n  - item2'
    var_6 = 'person:\n  name: John\n  age: 30'
    var_7 = ''
    var_8 = 'invalid: yaml: content:'
    var_9 = '   \n\n  '



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = 'name: John\n  invalid: [unclosed'
    var_6 = ''
    var_7 = module_0.String()
    var_8 = b'hello'
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = 'count: not_a_number'
    var_12 = 'items:\n  - name: item1\n  - name: item2'
    var_13 = '- 1\n- 2\n- 3'
    var_14 = module_0.Integer()
    var_15 = module_0.Array(var_14)
    var_16 = 'active: true'
    var_17 = module_0.Boolean()
    var_18 = 'value: null'



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'simple string'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'null'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'parent:\n  child: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'test'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'invalid: : yaml:'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'line1: value1\nline2: value2'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'items:\n  - name: item1\n    value: 10\n  - name: item2\n    value: 20'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'items'
    var_31 = var_29.value[var_30]
    var_32 = var_29.value[var_30]
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 'message: "Hello: World!"'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'text: こんにちは'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'text: café'
    var_39 = 'utf-8'
    var_40 = module_1.encode(var_39)
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = 'key: value\n  invalid indentation:'
    var_43 = module_0.tokenize_yaml(var_42)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1
import base64 as module_2

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'not_a_number'
    var_8 = module_0.Integer()
    var_9 = b'test_value'
    var_10 = module_0.String()
    var_11 = '- 1\n- 2\n- 3'
    var_12 = module_0.Integer()
    var_13 = module_0.Array(var_12)
    var_14 = ''
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = 'invalid: yaml: content:'
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = module_0.String()
    var_21 = module_0.String()
    var_22 = module_0.String()
    var_23 = 'name: Jane\naddress:\n  street: Main St\n  city: NYC'
    var_24 = 'name: José'
    var_25 = 'utf-8'
    var_26 = module_2.encode(var_25)
    var_27 = module_0.String()



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\n  invalid: [unclosed'
    var_3 = 'name: John\nage: not_a_number'
    var_4 = ''
    var_5 = 'hello world'
    var_6 = module_0.String()
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = 'name: John\naddress:\n  street: Main St\n  city: Boston'
    var_11 = 'active: true\ninactive: false'
    var_12 = 'name: John\nmiddle_name: null'



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: [yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test string'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = 'true'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = module_0.String()
    var_21 = '3.14'
    var_22 = module_0.String()



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '[1, 2, 3]'
    var_8 = module_0.Integer()
    var_9 = module_0.Array(var_8)
    var_10 = 'invalid: yaml: content:'
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test_value'
    var_14 = module_0.String()
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  \t  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'not_a_number'
    var_22 = module_0.Integer()
    var_23 = 'null'
    var_24 = True
    var_25 = module_0.String()
    var_26 = module_0.String()
    var_27 = module_0.String()
    var_28 = '\nname: Alice\naddress:\n  street: Main St\n  city: Boston\n'
    var_29 = 'true'
    var_30 = module_0.Boolean()
    var_31 = '3.14'
    var_32 = module_0.Float()



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value\nother: 123'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2\n- 3'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = '42'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '3.14'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'parent:\n  child: value\n  list:\n    - a\n    - b'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'parent'
    var_15 = var_13.value[var_14]
    var_16 = b'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '   \n  \n  '
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: value\n  invalid:\n invalid'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '[invalid: yaml:'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'line1: value\nline2: value2\nline3: value3'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = "key: 'value with: colon'"
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '# This is a comment\nkey: value'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '- string\n- 42\n- 3.14\n- true\n- null'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = var_33.value
    var_35 = len(var_34)
    assert var_35 == 5



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: [yaml: syntax:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test_string'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = 'flag: true\nempty: null'
    var_19 = module_0.Boolean()
    var_20 = 'not_a_number'
    var_21 = module_0.Integer()
    var_22 = '3.14'
    var_23 = module_0.Float()



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test the validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = 'name: John\nage: 30'
    var_6 = b'test_value'
    var_7 = module_0.String()
    var_8 = 'invalid: yaml: content:'
    var_9 = module_0.String()
    var_10 = '42'
    var_11 = module_0.Integer()
    var_12 = 'null'
    var_13 = True
    var_14 = module_0.String()
    var_15 = 'true'
    var_16 = module_0.Boolean()
    var_17 = '3.14'
    var_18 = module_0.Float()
    var_19 = 'not_a_number'
    var_20 = module_0.Integer()
    var_21 = '- item1\n- item2\n- item3'
    var_22 = module_0.String()
    var_23 = module_0.Array(var_22)



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John'
    var_1 = b'age: 25'
    var_2 = module_0.Field()
    var_3 = 'invalid: [unclosed'
    var_4 = module_0.Field()
    var_5 = ''
    var_6 = '   \n  \t  '
    var_7 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_8 = module_0.Field()
    var_9 = 'users'
    var_10 = 'hello world'
    var_11 = 20
    var_12 = module_0.Field()
    var_13 = 'a'
    var_14 = 100
    var_15 = var_13 * var_14
    var_16 = 10
    var_17 = module_0.Field()
    var_18 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_19 = module_0.Field()
    var_20 = module_0.Field()
    var_21 = module_0.Field()
    var_22 = module_0.Field()
    var_23 = module_0.Field()



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = 'invalid: yaml: content:'
    var_6 = module_0.String()
    var_7 = module_1.validate_yaml(var_5, var_6)
    var_8 = ''
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = b'test_string'
    var_12 = module_0.String()
    var_13 = '42'
    var_14 = module_0.Integer()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = 'user:\n  name: Alice\n  age: 25'
    var_19 = 'not_a_number'
    var_20 = module_0.Integer()
    var_21 = 'null'
    var_22 = True
    var_23 = module_0.String()



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '[1, 2, 3]'
    var_8 = module_0.Integer()
    var_9 = '{ invalid yaml: [}'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test content'
    var_16 = module_0.String()
    var_17 = '   \n  \t  '
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = 'name: VeryLongName'
    var_21 = 'null'
    var_22 = True
    var_23 = module_0.String()



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'hello'
    var_8 = module_0.Integer()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'name: Alice\nage: 25'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = '[1, 2, 3]'
    var_19 = module_0.Integer()
    var_20 = module_0.Array(var_19)
    var_21 = 'null'
    var_22 = True
    var_23 = module_0.String()
    var_24 = 'true'
    var_25 = module_0.Boolean()



# Parsed testcases at query #42
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'parent:\n  child: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'key: value'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '   \n  \n  '
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'users:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'users'
    var_29 = var_27.value[var_28]
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = module_0.tokenize_yaml(var_24)
    var_32 = 'start_position'
    var_33 = hasattr(var_31, var_32)
    var_34 = 'end_position'
    var_35 = hasattr(var_31, var_34)
    var_36 = 'text: |\n  line1\n  line2'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'key: café'
    var_39 = 'utf-8'
    var_40 = module_1.encode(var_39)
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = var_41.value
    var_43 = str(var_42)



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = b'test_value'
    var_6 = module_0.String()
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = '42'
    var_11 = module_0.Integer()
    var_12 = 'true'
    var_13 = module_0.Boolean()
    var_14 = 'null'
    var_15 = True
    var_16 = 'invalid: yaml: content:'
    var_17 = module_0.String()
    var_18 = module_1.validate_yaml(var_16, var_17)
    var_19 = ''
    var_20 = module_0.String()
    var_21 = module_1.validate_yaml(var_19, var_20)
    var_22 = 'not_a_number'
    var_23 = module_0.Integer()



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'parent:\n  child: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = b'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = '   \n  \t  '
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key: value\n  invalid:\n   : bad'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'line1: value1\nline2: value2'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'users:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'users'
    var_29 = var_27.value[var_28]
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = '- 1\n- hello\n- true\n- null'
    var_32 = module_0.tokenize_yaml(var_31)
    var_33 = '"quoted: string"'
    var_34 = module_0.tokenize_yaml(var_33)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = 'name: 123\nage: not_a_number'
    var_8 = module_0.String()
    var_9 = 'hello'
    var_10 = 'items:\n  - item1\n  - item2'
    var_11 = 'string: test\ninteger: 42\nfloat: 3.14\nboolean: true\nnull_value: null'



# Parsed testcases at query #46
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'hello world'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = ':\n  key: value'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'key1: value1\nkey2: value2'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '\nusers:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\n'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'users'
    var_33 = var_31.value[var_32]
    var_34 = var_31.value[var_32]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'test: data'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'café: français'
    var_39 = 'utf-8'
    var_40 = module_1.encode(var_39)
    var_41 = module_0.tokenize_yaml(var_40)



# Parsed testcases at query #47
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'parent:\n  child: value'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'hello'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '42'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'false'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'null'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: value\n  invalid:'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '\nusers:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\n'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'users'
    var_29 = var_27.value[var_28]
    var_30 = var_27.value[var_28]
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 'key: value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'message: |\n  line1\n  line2'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'name: José'
    var_37 = 'utf-8'
    var_38 = module_1.encode(var_37)
    var_39 = module_0.tokenize_yaml(var_38)



# Parsed testcases at query #48
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \n  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'hello'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'key: value\nkey2: value2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value\n  items:\n    - a\n    - b'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: : invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: [unclosed'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'line1: value1\nline2: value2'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'message: hello world'
    var_31 = 'utf-8'
    var_32 = module_1.encode(var_31)
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'key: value'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'start_pos'
    var_37 = hasattr(var_35, var_36)
    var_38 = 'end_pos'
    var_39 = hasattr(var_35, var_38)
    var_40 = '\nusers:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\nsettings:\n  debug: true\n  timeout: 5.5\n'
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = 'users'
    var_43 = var_41.value[var_42]
    var_44 = len(var_43)
    assert var_44 == 2



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = 'name: John\nage: 30'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = '- item1\n- item2\n- item3'
    var_6 = module_0.String()
    var_7 = module_0.Array(var_6)
    var_8 = 'invalid: yaml: content:'
    var_9 = module_0.String()
    var_10 = module_1.validate_yaml(var_8, var_9)
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'test_value'
    var_15 = module_0.String()
    var_16 = 'not_a_number'
    var_17 = module_0.Integer()
    var_18 = 'user:\n  name: Alice\n  age: 25'
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = 'enabled: true\ndisabled: false'
    var_22 = module_0.Boolean()
    var_23 = module_0.Boolean()
    var_24 = 'value: null'
    var_25 = True
    var_26 = module_0.String()
    var_27 = 'integer: 42\nfloat: 3.14'
    var_28 = module_0.Integer()
    var_29 = module_0.Integer()



# Parsed testcases at query #50
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = 'name: John\nage: 30'
    var_4 = b'name: Jane\nage: 25'
    var_5 = 'name: John\n  invalid: [unclosed'
    var_6 = ''
    var_7 = 'name: 123\nage: not_a_number'
    var_8 = module_0.String()
    var_9 = 'John'
    var_10 = '   \n  \n   '



# Parsed testcases at query #51
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'hello'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value\n  items:\n    - a\n    - b'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key1: value1\n  key2: value2'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'key1: value1\nkey2: value2'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '\nusers:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\nsettings:\n  debug: true\n  timeout: 30\n'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'users'
    var_33 = var_31.value[var_32]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'test: value'
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = 'content'
    var_38 = hasattr(var_36, var_37)
    var_39 = 'message: "Hello: World!"'
    var_40 = module_0.tokenize_yaml(var_39)
    var_41 = '{}'
    var_42 = module_0.tokenize_yaml(var_41)
    var_43 = '[]'
    var_44 = module_0.tokenize_yaml(var_43)



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = '{ invalid yaml: ['
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test_value'
    var_16 = module_0.String()
    var_17 = 'null'
    var_18 = True
    var_19 = module_0.String()
    var_20 = 'true'
    var_21 = module_0.Boolean()
    var_22 = '3.14'
    var_23 = module_0.Float()



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = 'not_a_number'
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = 'name: John\nage: 30'
    var_9 = '- item1\n- item2\n- item3'
    var_10 = module_0.String()
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = 'invalid: yaml: content:'
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = b'test_value'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'true'
    var_23 = module_0.Boolean()



# Parsed testcases at query #54
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '[1, 2, 3]'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'invalid: yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test string'
    var_16 = module_0.String()
    var_17 = module_0.String()
    var_18 = 'items: value'
    var_19 = '   \n  \t  '
    var_20 = module_0.String()
    var_21 = module_1.validate_yaml(var_19, var_20)
    var_22 = 'true'
    var_23 = module_0.String()



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test value'
    var_16 = module_0.String()
    var_17 = 'true'
    var_18 = module_0.Boolean()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'not a number'
    var_23 = module_0.Integer()



# Parsed testcases at query #56
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = 'name: John\nage: 30'
    var_6 = 'invalid: yaml: content:'
    var_7 = module_0.String()
    var_8 = module_1.validate_yaml(var_6, var_7)
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = b'test_value'
    var_13 = module_0.String()
    var_14 = '- item1\n- item2'
    var_15 = module_0.String()
    var_16 = '42'
    var_17 = module_0.Integer()
    var_18 = 'true'
    var_19 = module_0.String()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = '   \n  \n  '
    var_24 = module_0.String()
    var_25 = module_1.validate_yaml(var_23, var_24)



# Parsed testcases at query #57
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = 'name: John\nage: 30'
    var_6 = 'invalid: [yaml: syntax'
    var_7 = module_0.String()
    var_8 = module_1.validate_yaml(var_6, var_7)
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = '   \n  \n  '
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test_value'
    var_16 = module_0.String()
    var_17 = '- item1\n- item2\n- item3'
    var_18 = module_0.String()
    var_19 = module_0.Array(var_18)
    var_20 = '42'
    var_21 = module_0.Integer()
    var_22 = '3.14'
    var_23 = module_0.Float()
    var_24 = 'true'
    var_25 = module_0.Boolean()
    var_26 = 'null'
    var_27 = True
    var_28 = module_0.String()



# Parsed testcases at query #58
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = 'name: John\nage: 30'
    var_6 = 'invalid: [yaml: content:'
    var_7 = module_0.String()
    var_8 = 'not_a_number'
    var_9 = module_0.Integer()
    var_10 = b'hello'
    var_11 = module_0.String()
    var_12 = '- item1\n- item2'
    var_13 = module_0.String()
    var_14 = 'key1: value1\nkey2: value2'
    var_15 = module_0.String()
    var_16 = '42'
    var_17 = module_0.Integer()
    var_18 = 'true'
    var_19 = module_0.String()
    var_20 = 'null'
    var_21 = module_0.String()



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = '\nname: John\nage: 30\n'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\n  invalid: : syntax'
    var_3 = 'name: John\nage: not_a_number'
    var_4 = '42'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = '\nusers:\n  - name: Alice\n    age: 28\n  - name: Bob\n    age: 35\n'
    var_8 = 'users'
    var_9 = 'name: Charlie'
    var_10 = 'name: David\nage: 40\nemail: david@example.com'



# Parsed testcases at query #60
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'name: John\n  invalid: [unclosed'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test_value'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = '   \n\n  '
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = 'true'
    var_21 = module_0.String()
    var_22 = 'null'
    var_23 = module_0.String()



# Parsed testcases at query #61
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: : syntax'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = 5
    var_8 = module_0.String(max_length=var_7)
    var_9 = 'hello_world_too_long'
    var_10 = '42'
    var_11 = module_0.Integer()
    var_12 = '- item1\n- item2\n- item3'
    var_13 = module_0.String()
    var_14 = module_0.Array(var_13)
    var_15 = 'flag: true\nnumber: 3.14\nempty: null'
    var_16 = module_0.Boolean()
    var_17 = module_0.Float()



# Parsed testcases at query #62
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = 'name: John\n  invalid: : syntax'
    var_4 = b'name: Alice\nage: 25'
    var_5 = ''
    var_6 = module_0.String()
    var_7 = 'hello'
    var_8 = module_0.Integer()
    var_9 = 'count: not_a_number'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = 'person:\n  name: Bob\n  details:\n    age: 40'



# Parsed testcases at query #63
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = '\n    name: John\n    age: 30\n    '
    var_7 = '{ invalid yaml'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = b'test'
    var_11 = module_0.String()
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \n  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = 'name: VeryLongName'
    var_19 = '[1, 2, 3]'
    var_20 = module_0.String()
    var_21 = 'null'
    var_22 = module_0.String()



# Parsed testcases at query #64
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '[1, 2, 3]'
    var_8 = module_0.String()
    var_9 = '{ invalid: yaml: content }'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test_value'
    var_16 = module_0.String()
    var_17 = 'true'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = module_0.String()
    var_21 = '3.14'
    var_22 = module_0.String()



# Parsed testcases at query #65
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'hello'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'key: value\nother: data'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'parent:\n  child: value\n  list:\n    - a\n    - b'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '42'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'false'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'null'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'key: [invalid'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = ': invalid'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'test: value'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'line1: value1\nline2: value2'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '\n    root:\n      string: hello\n      number: 123\n      float: 45.67\n      bool: true\n      null_val: null\n      list:\n        - item1\n        - item2\n      nested:\n        deep: value\n    '
    var_33 = module_0.tokenize_yaml(var_32)



# Parsed testcases at query #66
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: [yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = '   \n  \t  '
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test_value'
    var_17 = module_0.String()
    var_18 = '- item1\n- item2\n- item3'
    var_19 = module_0.String()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = 'true'
    var_24 = module_0.String()
    var_25 = '3.14'
    var_26 = module_0.String()



# Parsed testcases at query #67
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = 'invalid: yaml: content:'
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test content'
    var_17 = module_0.String()
    var_18 = 'null'
    var_19 = True
    var_20 = module_0.String()
    var_21 = 'true'
    var_22 = module_0.Boolean()
    var_23 = '3.14'
    var_24 = module_0.Float()
    var_25 = 3.14



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\n  invalid: [unclosed'
    var_3 = ''
    var_4 = '   \n  \n  '
    var_5 = 'test_string'
    var_6 = '- item1\n- item2\n- item3'
    var_7 = 'items:\n  - name: first\n    value: 1\n  - name: second\n    value: 2'
    var_8 = 'items'
    var_9 = 'string: hello\nnumber: 42\nfloat: 3.14\nbool: true\nnull_val: null'



# Parsed testcases at query #69
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: [yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test content'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = 'null'
    var_19 = True
    var_20 = module_0.String()
    var_21 = 'true'
    var_22 = module_0.Boolean()
    var_23 = '3.14'
    var_24 = module_0.Float()



# Parsed testcases at query #70
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '{ invalid yaml: [unclosed'
    var_8 = module_0.String()
    var_9 = 'not_a_number'
    var_10 = module_0.Integer()
    var_11 = '- item1\n- item2'
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = ''
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = b'test_value'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'true'
    var_23 = module_0.Boolean()



# Parsed testcases at query #71
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{a: 1, b: 2}'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'key1: value1\nkey2: value2'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '- item1\n- item2\n- item3'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = b'hello'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '   \n  \n  '
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '{ invalid yaml: ['
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'test_value'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'line1\nline2'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = '\n    config:\n      database:\n        host: localhost\n        port: 5432\n        credentials:\n          username: admin\n          password: secret\n    '
    var_41 = module_0.tokenize_yaml(var_40)



# Parsed testcases at query #72
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test_value'
    var_14 = module_0.String()
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = 'user:\n  name: Alice\n  age: 25'
    var_19 = 'not_a_number'
    var_20 = module_0.Integer()
    var_21 = 'true'
    var_22 = module_0.Boolean()



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n\n  '
    var_7 = 'name: John\nage: not_a_number'
    var_8 = module_0.String()
    var_9 = 'hello'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = 'flag: true\nvalue: null\npi: 3.14'
    var_14 = module_0.Float()



# Parsed testcases at query #2
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'items:\n  - name: test\n    value: 123'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'hello'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '{ invalid: yaml: content:'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'test'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'line1\nline2: value'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '  value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = '- item1\n- item2\n- item3'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 3
    var_38 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_39 = module_0.tokenize_yaml(var_38)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: syntax:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \n  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = b'test content'
    var_19 = module_0.String()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = 'true'
    var_24 = module_0.Boolean()



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import base64 as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n\n   '
    var_7 = module_0.String()
    var_8 = 'hello'
    var_9 = 'name: 123\nage: invalid_age'
    var_10 = 'name: José\nage: 28'
    var_11 = 'utf-8'
    var_12 = module_1.encode(var_11)
    var_13 = 'items:\n  - apple\n  - banana'
    var_14 = 'user:\n  name: Bob\n  age: 35'
    var_15 = None



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: [yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \n  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = b'test_value'
    var_19 = module_0.String()
    var_20 = module_0.String()
    var_21 = module_0.String()
    var_22 = module_0.String()
    var_23 = 'name: Alice\naddress:\n  street: Main St\n  city: Boston'
    var_24 = 'value: null'
    var_25 = True
    var_26 = module_0.String()



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = b'test_value'
    var_6 = module_0.String()
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = '   \n  \t  '
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = '- item1\n- item2\n- item3'
    var_17 = module_0.String()
    var_18 = module_0.Array(var_17)
    var_19 = 'count: 42\nactive: true\nratio: 3.14'
    var_20 = module_0.Integer()
    var_21 = module_0.Boolean()
    var_22 = module_0.Float()
    var_23 = 'value: null'



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'hello'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'null'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'parent:\n  child: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '{ invalid: yaml: syntax:'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = module_0.tokenize_yaml(var_6)
    var_25 = 'greeting: こんにちは'
    var_26 = module_0.tokenize_yaml(var_25)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = 3
    var_7 = module_0.String(max_length=var_6)
    var_8 = 'toolongname'
    var_9 = '42'
    var_10 = module_0.Integer()
    var_11 = None
    var_12 = '- item1\n- item2\n- item3'
    var_13 = '\n    user:\n      name: Alice\n      age: 28\n      email: alice@example.com\n    '
    var_14 = None
    var_15 = '\n    active: true\n    inactive: false\n    empty: null\n    '
    var_16 = None
    var_17 = None
    var_18 = None
    var_19 = '\n    pi: 3.14159\n    ratio: 0.5\n    '
    var_20 = None
    var_21 = None
    var_22 = 'pi'
    var_23 = 'ratio'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: : syntax'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = module_0.String()
    var_8 = 'test_string'
    var_9 = module_0.Integer()
    var_10 = 'not_an_integer'
    var_11 = '\n    users:\n      - name: Alice\n        age: 28\n      - name: Bob\n        age: 35\n    '
    var_12 = module_0.Array()



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = b'test string'
    var_10 = module_0.String()
    var_11 = '{ invalid: yaml: content'
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = ''
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = '   \n  \n  '
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = 'true'
    var_21 = module_0.Boolean()
    var_22 = 'null'
    var_23 = True
    var_24 = module_0.String()



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = '\nname: John\nage: 30\n'
    var_3 = b'name: Jane\nage: 25'
    var_4 = module_0.String()
    var_5 = 'hello'
    var_6 = module_0.Integer()
    var_7 = '42'
    var_8 = module_0.Integer()
    var_9 = module_0.Array(var_8)
    var_10 = '\n- 1\n- 2\n- 3\n'
    var_11 = '\nname: John\n  age: 30\n    invalid indent\n'
    var_12 = ''
    var_13 = '   \n  \n  '
    var_14 = '\nname: John\nage: not_a_number\n'
    var_15 = module_0.String()
    var_16 = module_0.String()
    var_17 = module_0.String()
    var_18 = '\nname: Alice\naddress:\n  street: Main St\n  city: New York\n'



# Parsed testcases at query #12
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'items:\n  - name: test\n    value: 123'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'hello'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '{ invalid yaml :'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'test'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'start_position'
    var_31 = hasattr(var_29, var_30)
    var_32 = 'end_position'
    var_33 = hasattr(var_29, var_32)
    var_34 = 'key1: value1\nkey2: value2'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = 'users'
    var_41 = var_39.value[var_40]
    var_42 = 'invalid: [yaml: content'
    var_43 = module_0.tokenize_yaml(var_42)
    var_44 = 'message: Hello, 世界'
    var_45 = module_0.tokenize_yaml(var_44)
    var_46 = "'special: chars!'"
    var_47 = module_0.tokenize_yaml(var_46)
    var_48 = var_47.value
    var_49 = str(var_48)
    var_50 = "[1, 'two', 3.0, true, null]"
    var_51 = module_0.tokenize_yaml(var_50)
    var_52 = var_51.value
    var_53 = len(var_52)
    assert var_53 == 5



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = '42'
    var_5 = module_0.Integer()
    var_6 = 'name: John\n  age: [invalid'
    var_7 = ''
    var_8 = '   \n\t  '
    var_9 = 'name: John\nage: not_a_number'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = 'person:\n  name: Bob\n  age: 35'



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_1.validate_yaml(var_0, var_1)
    assert var_2 == 'hello'
    var_3 = '42'
    var_4 = module_0.Integer()
    var_5 = module_1.validate_yaml(var_3, var_4)
    assert var_5 == 42
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = '\nname: John\nage: 30\n'
    var_9 = b'test_string'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    assert var_11 == 'test_string'
    var_12 = 'invalid: : yaml:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'not_a_number'
    var_22 = module_0.Integer()
    var_23 = module_1.validate_yaml(var_21, var_22)
    var_24 = '\nusers:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30\n'
    var_25 = module_0.Field()
    var_26 = module_1.validate_yaml(var_24, var_25)
    var_27 = '\nstring_val: hello\nint_val: 123\nfloat_val: 45.67\nbool_val: true\nnull_val: null\nlist_val:\n  - item1\n  - item2\n'
    var_28 = module_0.Field()
    var_29 = module_1.validate_yaml(var_27, var_28)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = '42'
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = 'name: John\nage: 30'
    var_8 = '[1, 2, 3]'
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = 'invalid: [yaml: syntax:'
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = b'test_value'
    var_15 = module_0.String()
    var_16 = 'null'
    var_17 = True
    var_18 = module_0.String()
    var_19 = 'true'
    var_20 = module_0.Boolean()
    var_21 = '3.14'
    var_22 = module_0.Float()
    var_23 = 'not_a_number'
    var_24 = module_0.Integer()



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'Test validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = b'world'
    var_4 = module_0.String()
    var_5 = '42'
    var_6 = module_0.Integer()
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = 'name: John\nage: 30'
    var_10 = '- item1\n- item2'
    var_11 = module_0.String()
    var_12 = 'invalid: yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = module_0.String()
    var_22 = 'items: test'
    var_23 = module_0.Integer()
    var_24 = 'count: not_a_number'



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = 'invalid: yaml: content:'
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = ''
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test: value'
    var_17 = module_0.String()
    var_18 = 'true'
    var_19 = module_0.Boolean()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = '3.14'
    var_24 = module_0.Float()



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- apple\n- banana\n- cherry'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test_value'
    var_16 = module_0.String()
    var_17 = 'name: VeryLongName'
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = module_0.String()
    var_22 = module_0.String()
    var_23 = module_0.String()
    var_24 = 'name: Alice\naddress:\n  street: Main St\n  city: NYC'



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content: :'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = b'test_value'
    var_13 = module_0.String()
    var_14 = ''
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = '   \n   '
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = module_0.String()
    var_21 = 'data: nested_value'
    var_22 = 'active: true\nvalue: null'
    var_23 = module_0.String()



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 'key: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'test'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'name: José'
    var_21 = 'utf-8'
    var_22 = module_1.encode(var_21)
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = '   \n  \t  '
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '{ invalid yaml :'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '\n    users:\n      - name: John\n        age: 30\n      - name: Jane\n        age: 25\n    '
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'key: value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'line1: value1\nline2: value2'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = "message: 'Hello, World!'"
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = '0x10'
    var_39 = module_0.tokenize_yaml(var_38)
    var_40 = '1.5e3'
    var_41 = module_0.tokenize_yaml(var_40)
    var_42 = '- key1: val1\n- key2: val2'
    var_43 = module_0.tokenize_yaml(var_42)
    var_44 = var_43.value
    var_45 = len(var_44)
    assert var_45 == 2



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\n  invalid: : syntax'
    var_5 = 'name: John\nage: not_a_number'
    var_6 = module_0.String()
    var_7 = 'hello'
    var_8 = module_0.Integer()
    var_9 = '42'
    var_10 = 'items:\n  - item1\n  - item2'
    var_11 = ''
    var_12 = '   \n  \n  '



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: [yaml: content:'
    var_10 = module_0.String()
    var_11 = ''
    var_12 = module_0.String()
    var_13 = module_1.validate_yaml(var_11, var_12)
    var_14 = '   \n  \t  '
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = b'test_value'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'true'
    var_23 = module_0.Boolean()



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = b'test_value'
    var_13 = module_0.String()
    var_14 = '[1, 2, 3]'
    var_15 = module_0.Integer()
    var_16 = module_0.Array(var_15)
    var_17 = 'true'
    var_18 = module_0.Boolean()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()
    var_22 = 'not_a_number'
    var_23 = module_0.Integer()



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = b'test_value'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \t  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '- item1\n- item2\n- item3'
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19)
    var_21 = 'value: null'
    var_22 = 'enabled: true\ndisabled: false'
    var_23 = module_0.Boolean()
    var_24 = module_0.Boolean()



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = b'name: Jane\nage: 25'
    var_2 = 'name: John\n  invalid: [unclosed'
    var_3 = ''
    var_4 = '   \n  \n  '
    var_5 = 'name: John'
    var_6 = '42'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = 'users:\n  - name: Alice\n    age: 28\n  - name: Bob\n    age: 32'
    var_9 = 'users'



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = module_0.Integer()
    var_4 = '[1, 2, 3]'
    var_5 = module_0.Integer()
    var_6 = module_0.Array(var_5)
    var_7 = '{ invalid: yaml: syntax'
    var_8 = module_0.Integer()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.Integer()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'name: Jane\nage: 25'
    var_14 = module_0.Field()
    var_15 = module_0.Integer()
    var_16 = 'age: not_a_number'
    var_17 = module_0.Integer()
    var_18 = 'value: null'
    var_19 = 'active: true\ninactive: false'
    var_20 = module_0.Boolean()
    var_21 = module_0.Boolean()
    var_22 = 'price: 19.99'
    var_23 = module_0.Float()
    var_24 = 'user:\n  name: Bob\n  scores: [1, 2, 3]'
    var_25 = module_0.Field()
    var_26 = '   \n  \n  '
    var_27 = module_0.Integer()
    var_28 = module_1.validate_yaml(var_26, var_27)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello world'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = ''
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = 'invalid: yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test value'
    var_16 = module_0.String()
    var_17 = b'invalid: yaml: content:'
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = 'true'
    var_24 = module_0.Boolean()



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\nage: [invalid'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = 'name: 123\nage: not_a_number'
    var_8 = 5
    var_9 = module_0.String(max_length=var_8)
    var_10 = 'hello'
    var_11 = 'this_is_too_long'
    var_12 = '- item1\n- item2\n- item3'
    var_13 = module_0.String()
    var_14 = module_0.Array(var_13)
    var_15 = 'count: 42\nratio: 3.14\nactive: true\nempty: null'
    var_16 = module_0.Integer()
    var_17 = module_0.Integer()
    var_18 = module_0.String()
    var_19 = module_0.String()



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name: John\n  invalid: [unclosed'
    var_2 = ''
    var_3 = b'name: Jane\nage: 25'
    var_4 = '   \n\n   '
    var_5 = 'hello world'
    var_6 = module_0.String()
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = 'person:\n  name: Bob\n  details:\n    age: 40'



# Parsed testcases at query #30
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = 'key: value'
    var_18 = module_0.tokenize_yaml(var_17)
    var_19 = 'key'
    var_20 = var_18.value[var_19]
    var_21 = 'items:\n  - name: item1\n  - name: item2'
    var_22 = module_0.tokenize_yaml(var_21)
    var_23 = b'hello: world'
    var_24 = module_0.tokenize_yaml(var_23)
    var_25 = module_0.tokenize_yaml(var_17)
    var_26 = ''
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '   \n  \t  '
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = '{invalid: [yaml: syntax}'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = '# comment\nkey: value'
    var_33 = module_0.tokenize_yaml(var_32)
    var_34 = 'text: |\n  line1\n  line2'
    var_35 = module_0.tokenize_yaml(var_34)
    var_36 = 'a: 1'
    var_37 = module_0.tokenize_yaml(var_36)
    var_38 = 'start_index'
    var_39 = hasattr(var_37, var_38)
    var_40 = 'end_index'
    var_41 = hasattr(var_37, var_40)
    var_42 = '\n    users:\n      - name: Alice\n        age: 30\n      - name: Bob\n        age: 25\n    '
    var_43 = module_0.tokenize_yaml(var_42)
    var_44 = '- item1\n- item2\n- item3'
    var_45 = module_0.tokenize_yaml(var_44)
    var_46 = var_45.value
    var_47 = len(var_46)
    assert var_47 == 3



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = 'invalid: yaml: content:'
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = ''
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test_value'
    var_17 = module_0.String()
    var_18 = 'true'
    var_19 = module_0.Boolean()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = 'not_a_number'
    var_24 = module_0.Integer()



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'hello'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'key: value\nfoo: bar'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'parent:\n  child: value\n  list:\n    - a\n    - b'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'invalid: ['
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: value\ninvalid: {'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'test: data'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'text: |\n  line1\n  line2'
    var_31 = module_0.tokenize_yaml(var_30)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = '42'
    var_4 = module_0.Integer()
    var_5 = module_1.validate_yaml(var_3, var_4)
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = 'name: John\nage: 30'
    var_9 = 'hello'
    var_10 = module_0.Integer()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = b'test_string'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '- item1\n- item2\n- item3'
    var_16 = module_0.String()
    var_17 = module_0.Array(var_16)
    var_18 = module_1.validate_yaml(var_15, var_17)
    var_19 = 'true'
    var_20 = module_0.Boolean()
    var_21 = module_1.validate_yaml(var_19, var_20)
    var_22 = 'null'
    var_23 = True
    var_24 = module_0.String()
    var_25 = module_1.validate_yaml(var_22, var_24)
    var_26 = 'invalid: yaml: content:'
    var_27 = module_0.String()
    var_28 = module_1.validate_yaml(var_26, var_27)
    var_29 = ''
    var_30 = module_0.String()
    var_31 = module_1.validate_yaml(var_29, var_30)



# Parsed testcases at query #34
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'parent:\n  child: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = b'test: data'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '   \n  \t  '
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '{invalid: [yaml: content]'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: value\nsecond: line'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '\n    users:\n      - name: John\n        age: 30\n      - name: Jane\n        age: 25\n    '
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'users'
    var_31 = var_29.value[var_30]
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = '- item1\n- item2\n- item3'
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = 'message: "Hello, World!"'
    var_36 = module_0.tokenize_yaml(var_35)
    var_37 = 'text: |\n  Line 1\n  Line 2'
    var_38 = module_0.tokenize_yaml(var_37)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'simple string'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'parent:\n  child: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'number: 42'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'decimal: 3.14'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'flag: true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'empty: null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'key1: value1\nkey2: value2'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'invalid: [unclosed'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'message: "hello world"'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = '- name: Alice\n  age: 30\n- name: Bob\n  age: 25'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 'test: data'
    var_33 = module_0.tokenize_yaml(var_32)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = b'test'
    var_16 = module_0.String()
    var_17 = 'true'
    var_18 = module_0.String()
    var_19 = 'null'
    var_20 = module_0.String()
    var_21 = '3.14'
    var_22 = module_0.String()



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = b'test_string'
    var_14 = module_0.String()
    var_15 = '[1, 2, 3]'
    var_16 = module_0.Integer()
    var_17 = module_0.Array(var_16)
    var_18 = 'key1: value1\nkey2: value2'
    var_19 = module_0.String()
    var_20 = '   \n  \t  '
    var_21 = module_0.String()
    var_22 = module_1.validate_yaml(var_20, var_21)
    var_23 = 'null'
    var_24 = True
    var_25 = module_0.String()



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = module_0.Array(var_8)
    var_10 = 'invalid: yaml: syntax:'
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = ''
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test_value'
    var_17 = module_0.String()
    var_18 = 'key: null'
    var_19 = 'enabled: true'
    var_20 = module_0.Boolean()
    var_21 = '3.14'
    var_22 = module_0.Float()



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = b'test_value'
    var_8 = module_0.String()
    var_9 = 'invalid: yaml: content:'
    var_10 = module_0.String()
    var_11 = module_1.validate_yaml(var_9, var_10)
    var_12 = ''
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = '   \n  \t  '
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '[1, 2, 3]'
    var_19 = module_0.String()
    var_20 = 'null'
    var_21 = True
    var_22 = module_0.String()
    var_23 = 'true'
    var_24 = module_0.String()



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test the validate_yaml function with various inputs.'
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = '42'
    var_4 = module_0.Integer()
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = 'name: John\nage: 30'
    var_8 = '{ invalid yaml: [unclosed'
    var_9 = module_0.String()
    var_10 = 'not_a_number'
    var_11 = module_0.Integer()
    var_12 = '- item1\n- item2\n- item3'
    var_13 = module_0.String()
    var_14 = module_0.Array(var_13)
    var_15 = 'true'
    var_16 = module_0.Boolean()
    var_17 = 'null'
    var_18 = True
    var_19 = module_0.String()
    var_20 = b'hello'
    var_21 = module_0.String()
    var_22 = '3.14'
    var_23 = module_0.Float()



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = 'name: John\nage: 30'
    var_5 = 'invalid: yaml: content:'
    var_6 = module_0.String()
    var_7 = b'test_value'
    var_8 = module_0.String()
    var_9 = '- item1\n- item2\n- item3'
    var_10 = module_0.String()
    var_11 = module_0.Array(var_10)
    var_12 = 'user:\n  name: Alice\n  email: alice@example.com'
    var_13 = module_0.Object()
    var_14 = '12345'
    var_15 = 100
    var_16 = module_0.Integer(maximum=var_15)
    var_17 = 'enabled: true\nnullable: null\ndisabled: false'
    var_18 = module_0.Object()



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = '\nname: John\nage: 30\n'
    var_3 = '\nname: John\nage: invalid:syntax:\n'
    var_4 = b'name: Jane\nage: 25'
    var_5 = module_0.String()
    var_6 = 'hello'
    var_7 = ''
    var_8 = '   \n  \t  '
    var_9 = '\nname: John\nage: not_an_integer\n'
    var_10 = '\n- item1\n- item2\n- item3\n'
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'hello'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = '- item1\n- item2'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'key:\n  nested: value'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'false'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = '3.14'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'value'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = module_0.tokenize_yaml(var_6)
    var_27 = 'key: [invalid'
    var_28 = module_0.tokenize_yaml(var_27)
    var_29 = 'root:\n  - item1\n  - item2\n  key: value'
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = 'key: |\n  line1\n  line2'
    var_32 = module_0.tokenize_yaml(var_31)
    var_33 = b'test: \xc3\xa9'
    var_34 = module_0.tokenize_yaml(var_33)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0
import base64 as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '   \n  \t  '
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = b'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = 'key: value'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'outer:\n  inner: value'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'simple string'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = '42'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = 'true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = 'false'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = 'null'
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = 'users:\n  - name: John\n    age: 30\n  - name: Jane\n    age: 25'
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'users'
    var_27 = var_25.value[var_26]
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'key: "unclosed'
    var_30 = module_0.tokenize_yaml(var_29)
    var_31 = 'key: value\n  bad: indent\n    worse: indent'
    var_32 = module_0.tokenize_yaml(var_31)
    var_33 = 'test: 123'
    var_34 = module_0.tokenize_yaml(var_33)
    var_35 = 'message: Hello 世界'
    var_36 = 'utf-8'
    var_37 = module_1.encode(var_36)
    var_38 = module_0.tokenize_yaml(var_37)
    var_39 = 'text: |\n  line1\n  line2'
    var_40 = module_0.tokenize_yaml(var_39)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = module_0.String()
    var_8 = 'John'
    var_9 = module_0.Integer()
    var_10 = '42'
    var_11 = '- item1\n- item2\n- item3'
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = 'users:\n  - name: John\n    age: 30\n  - name: Jane\n    age: 28'
    var_15 = 'users'



# Parsed testcases at query #46
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n\n  '
    var_7 = 'name: Bob'
    var_8 = module_0.String()
    var_9 = 'John'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.String()
    var_12 = 'user:\n  name: Alice\n  age: 28'
    var_13 = 'count: 42\nratio: 3.14\nactive: true\nempty: null'
    var_14 = module_0.Integer()
    var_15 = module_0.Float()
    var_16 = module_0.Boolean()



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: Bob\nage: invalid'
    var_5 = module_0.String()
    var_6 = 'hello'
    var_7 = module_0.Integer()
    var_8 = '42'
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = '[1, 2, 3]'
    var_12 = module_0.String()
    var_13 = module_0.String()
    var_14 = module_0.String()
    var_15 = 'name: Charlie\naddress:\n  street: Main St\n  city: NYC'
    var_16 = ''
    var_17 = 'invalid: yaml: content:'



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = '   \n  \n  '
    var_7 = 5
    var_8 = module_0.String(max_length=var_7)
    var_9 = 'this_is_a_very_long_string'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = 'hello world'
    var_14 = module_0.String()
    var_15 = '42'
    var_16 = module_0.Integer()
    var_17 = 'true'
    var_18 = module_0.Boolean()
    var_19 = 'null'
    var_20 = True
    var_21 = module_0.String()



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = '\nname: John\nage: 30\n'
    var_7 = 'not_a_number'
    var_8 = module_0.Integer()
    var_9 = '\n- item1\n- item2\n- item3\n'
    var_10 = module_0.String()
    var_11 = module_0.Array(var_10)
    var_12 = b'test_string'
    var_13 = module_0.String()
    var_14 = ''
    var_15 = module_0.String()
    var_16 = module_1.validate_yaml(var_14, var_15)
    var_17 = '{invalid: yaml: content:'
    var_18 = module_0.String()
    var_19 = module_1.validate_yaml(var_17, var_18)
    var_20 = 'other_field: value'
    var_21 = module_0.String()
    var_22 = module_0.String()
    var_23 = module_0.String()
    var_24 = '\nname: Alice\naddress:\n  street: Main St\n  city: New York\n'



# Parsed testcases at query #50
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = '- item1\n- item2'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'hello'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_yaml(var_14)
    var_16 = 'outer:\n  inner: value'
    var_17 = module_0.tokenize_yaml(var_16)
    var_18 = '- 1\n- hello\n- true'
    var_19 = module_0.tokenize_yaml(var_18)
    var_20 = b'key: value'
    var_21 = module_0.tokenize_yaml(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_yaml(var_22)
    var_24 = '   \n  \n  '
    var_25 = module_0.tokenize_yaml(var_24)
    var_26 = 'key: [invalid'
    var_27 = module_0.tokenize_yaml(var_26)
    var_28 = 'key1: value1\nkey2: value2'
    var_29 = module_0.tokenize_yaml(var_28)
    var_30 = 'test: value'
    var_31 = module_0.tokenize_yaml(var_30)
    var_32 = 'start_position'
    var_33 = hasattr(var_31, var_32)
    var_34 = 'end_position'
    var_35 = hasattr(var_31, var_34)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Alice\nage: 25'
    var_4 = 'name: John\nage: [invalid'
    var_5 = ''
    var_6 = '   \n  \t  '
    var_7 = 'name: John\nage: not_a_number'
    var_8 = module_0.String()
    var_9 = 'hello'
    var_10 = '- item1\n- item2\n- item3'
    var_11 = 'name: John\nage: 30\naddress:\n  city: NYC\n  zip: 10001'
    var_12 = 'active: true\nempty: null\nrating: 4.5'



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name: John\nage: 30'
    var_3 = b'name: Jane\nage: 25'
    var_4 = 'name: John\n  invalid: [unclosed'
    var_5 = ''
    var_6 = module_0.String()
    var_7 = 'hello'
    var_8 = 'name: Christopher'
    var_9 = '- item1\n- item2\n- item3'
    var_10 = module_0.String()
    var_11 = module_0.Array(var_10)
    var_12 = '\n    users:\n      - name: Alice\n        age: 28\n      - name: Bob\n        age: 35\n    '
    var_13 = 'users'
    var_14 = '\n    string: hello\n    integer: 42\n    float: 3.14\n    boolean: true\n    null_value: null\n    '
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = module_0.String()
    var_18 = module_0.String()



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = b'world'
    var_3 = module_0.String()
    var_4 = '42'
    var_5 = module_0.Integer()
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = 'name: John\nage: 30'
    var_9 = '- item1\n- item2\n- item3'
    var_10 = module_0.String()
    var_11 = module_0.Array(var_10)
    var_12 = 'invalid: yaml: content:'
    var_13 = module_0.String()
    var_14 = module_1.validate_yaml(var_12, var_13)
    var_15 = ''
    var_16 = module_0.String()
    var_17 = module_1.validate_yaml(var_15, var_16)
    var_18 = '   \n  \n  '
    var_19 = module_0.String()
    var_20 = module_1.validate_yaml(var_18, var_19)
    var_21 = 'not_a_number'
    var_22 = module_0.Integer()
    var_23 = 'other_field: value'
    var_24 = 'users:\n  - name: Alice\n    age: 25\n  - name: Bob\n    age: 30'
    var_25 = module_0.Object()
    var_26 = 'flag: true\nempty: null'
    var_27 = module_0.Object()



# Parsed testcases at query #54
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = 'invalid: yaml: content:'
    var_8 = module_0.String()
    var_9 = module_1.validate_yaml(var_7, var_8)
    var_10 = ''
    var_11 = module_0.String()
    var_12 = module_1.validate_yaml(var_10, var_11)
    var_13 = '   \n  \t  '
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test_value'
    var_17 = module_0.String()
    var_18 = '- item1\n- item2\n- item3'
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19)
    var_21 = 'outer:\n  inner: value'
    var_22 = module_0.String()
    var_23 = 'not_a_number'
    var_24 = module_0.Integer()



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = '42'
    var_3 = module_0.Integer()
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = 'name: John\nage: 30'
    var_7 = '- item1\n- item2\n- item3'
    var_8 = module_0.String()
    var_9 = '{ invalid: yaml: content'
    var_10 = module_0.String()
    var_11 = 'not_a_number'
    var_12 = module_0.Integer()
    var_13 = ''
    var_14 = module_0.String()
    var_15 = module_1.validate_yaml(var_13, var_14)
    var_16 = b'test_value'
    var_17 = module_0.String()
    var_18 = 'null'
    var_19 = True
    var_20 = module_0.String()
    var_21 = 'true'
    var_22 = module_0.Boolean()



