####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 'key'
    var_5 = var_1.value[var_4]
    var_6 = '{"outer": {"inner": "value"}}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'outer'
    var_9 = var_7.value[var_8]
    var_10 = '[1, 2, "three"]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = '{"null": null, "bool": true, "number": 42.5}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"invalid": json}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '  {  "key"  :  "value"  }  '
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"text": "Hello\\nWorld", "value": 123.45}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"data": null, "flag": false}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "NYC"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'person'
    var_9 = var_5.value[var_8]
    var_10 = '{"tags": ["python", "json"], "count": 2}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'tags'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = '{"is_active": true, "data": null, "ratio": 1.5}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = ''
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John", "age": 30,}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John" "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{name: "John"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'{"name": "John", "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = b'{"name": "J\xffhn", "age": 30}'
    var_32 = module_0.tokenize_json(var_31)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'invalid json'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '[1, 2, "three"]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '"string"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"a": {"b": [1, 2, {"c": 3}]}}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"int": 42, "float": 3.14, "exp": 1e5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"null": null, "bool": true}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'\xff\xfe{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '{"data": {"nested": [1, 2, 3]}, "flag": true}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'data'
    var_11 = var_9.value[var_10]
    var_12 = 'nested'
    var_13 = var_9.value[var_10]
    var_14 = var_13.value[var_12]
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{name: "John"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   \n  \t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 2



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '[1, 2, 3, "four"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 4
    var_8 = '{"data": {"nested": [1, 2, 3]}, "flag": true}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'data'
    var_11 = var_9.value[var_10]
    var_12 = 'nested'
    var_13 = var_9.value[var_10]
    var_14 = var_13.value[var_12]
    var_15 = '{"value": null}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"null": null, "bool": true, "float": 1.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = "{'name': 'John'}"
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = '[]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = '  {  "key"  :  "value"  }  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John", "age": 30'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"name": "John", age: 30}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '{"name": "John "Doe"", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": null}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"active": true, "deleted": false}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"price": 19.99, "quantity": 5}'
    var_34 = module_0.tokenize_json(var_33)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age":}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{name: 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": }'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"text": "Hello\\nWorld"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{'name': 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John" /* comment */}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John "Doe"", "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "Jöhn", "age": 30}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name": null}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"active": true, "deleted": false}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"price": 19.99, "quantity": 5}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"value": 1.23e+4}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value
    var_44 = len(var_43)
    assert var_44 == 0
    var_45 = '{"items": []}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = var_46.value[var_10]
    var_48 = var_46.value[var_10]
    var_49 = var_48.value
    var_50 = len(var_49)
    assert var_50 == 0
    var_51 = '  {  "name"  :  "John"  }  '
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{\n\t"name": "John"\n}'
    var_54 = module_0.tokenize_json(var_53)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John"'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"text": "Hello\\nWorld"}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = '{"a": null, "b": true, "c": false}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = ''
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", "age": 30'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{name: "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"id": 1, "name": "Alice"}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"valid": true}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  \n  {"key": "value"}  \t  '
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"text": "Hello\\nWorld"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30,}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"name": "John", "age": 30}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = b'{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json"], "counts": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3]}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John\\u0020Doe", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"name": "John", "hobbies": ["reading", "swimming"]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'hobbies'
    var_11 = var_9.value[var_10]
    var_12 = '{"is_active": true, "balance": null, "score": 98.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = "{name: 'John'}"
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John", "age": 30,}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = b'\xff\xfe{"name": "John"}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John" "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = "{name: 'John'}"
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John" /* comment */}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"items": [1, 2, 3,]}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "John}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name": "John"'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"items": [1, 2, 3'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"name": "John\\x"}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"age": +30}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"age": 01}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"age": 0x14}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"age": Infinity}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"age": NaN}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = b'{"name": "John", "age": 30}'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = var_54.value
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = '  \n  \t  {"name": "John", "age": 30}  \n  \t  '
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = '{"name": "Jöhn", "emoji": "😀"}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = var_62.value
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = '{"name": "John\\nDoe", "quote": "He said \\"Hello\\""}'
    var_66 = module_0.tokenize_json(var_65)
    var_67 = var_66.value
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = '{"name": null}'
    var_70 = module_0.tokenize_json(var_69)
    var_71 = var_70.value
    var_72 = len(var_71)
    assert var_72 == 1
    var_73 = '{"is_active": true, "is_admin": false}'
    var_74 = module_0.tokenize_json(var_73)
    var_75 = var_74.value
    var_76 = len(var_75)
    assert var_76 == 2
    var_77 = '{"price": 19.99, "discount": 0.25}'
    var_78 = module_0.tokenize_json(var_77)
    var_79 = var_78.value
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = '{"value": 1.23e+4, "small": 5.67e-8}'
    var_82 = module_0.tokenize_json(var_81)
    var_83 = var_82.value
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = '{}'
    var_86 = module_0.tokenize_json(var_85)
    var_87 = var_86.value
    var_88 = len(var_87)
    assert var_88 == 0
    var_89 = '{"items": []}'
    var_90 = module_0.tokenize_json(var_89)
    var_91 = var_90.value
    var_92 = len(var_91)
    assert var_92 == 1
    var_93 = var_90.value[var_14]
    var_94 = var_90.value[var_14]
    var_95 = var_94.value
    var_96 = len(var_95)
    assert var_96 == 0
    var_97 = '{"matrix": [[1, 2], [3, 4]]}'
    var_98 = module_0.tokenize_json(var_97)
    var_99 = var_98.value
    var_100 = len(var_99)
    assert var_100 == 1
    var_101 = 'matrix'
    var_102 = var_98.value[var_101]
    var_103 = var_98.value[var_101]
    var_104 = var_103.value
    var_105 = len(var_104)
    assert var_105 == 2
    var_106 = 0
    var_107 = var_98.value[var_101]
    var_108 = var_107.value[var_106]
    var_109 = var_98.value[var_101]
    var_110 = var_109.value[var_106]
    var_111 = var_110.value
    var_112 = len(var_111)
    assert var_112 == 2



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": }'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '[1, "two", true]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"a": {"b": [1, 2, 3]}, "c": null}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"key": "\xff"}'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"name": "John", "age": 30,}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"text": "Hello\\nWorld", "value": 123.45}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"a": null, "b": true, "c": false}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"id": 1, "name": "Alice"}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  \n  { "name" : "John" , "age" : 30 }  \t  '
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  {  "name"  :  "John"  }  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John\\nDoe", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"int": 42, "float": 3.14, "exp": 1e3}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"bool1": true, "bool2": false, "null": null}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = b'{"name": "John", "age": 30}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'user'
    var_17 = var_13.value[var_16]
    var_18 = '[1, 2, 3, "four"]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = '{"a": null, "b": true, "c": false}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"data": {"nested": [1, 2, 3]}, "flag": true}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'data'
    var_13 = var_11.value[var_12]
    var_14 = 'nested'
    var_15 = var_11.value[var_12]
    var_16 = var_15.value[var_14]
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"special": "é"}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #42
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{name: 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", /* comment */ "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John"} extra'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "Jöhn", "emoji": "😀"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name": "John\\nDoe", "quote": "He said \\"Hello\\""}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"null": null, "bool_true": true, "bool_false": false}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '[1, "two", {"three": 3}]'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 3
    var_47 = 2
    var_48 = var_44.value[var_47]



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"null": null, "bool": true, "float": 1.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 'key'
    var_5 = var_1.value[var_4]
    var_6 = '{"outer": {"inner": [1, 2, 3]}}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'outer'
    var_9 = var_7.value[var_8]
    var_10 = 'inner'
    var_11 = var_7.value[var_8]
    var_12 = var_11.value[var_10]
    var_13 = var_7.value[var_8]
    var_14 = var_13.value[var_10]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"key": "value"'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"key": "value"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_22.value[var_4]
    var_26 = '  \n  {"key": "value"}  \t  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = var_27.value[var_4]



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"text": "Hello\\nWorld"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{'name': 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = var_3.value
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = '[1, 2, "three"]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = '"hello"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30,}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = '  \n  \t  {"name": "John", "age": 30}  \n  \t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '[1, 2, 3, "test"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"value": null}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"active": true, "deleted": false}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"scientific": 1.23e+4}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John", "age": 30,}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '   \n  \t  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = b'{"name": "John", "age": 30}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "J\xffhn", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John" /* comment */}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = "{name: 'John'}"
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name" "John"}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John", "age": 30}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '{"data": [{"id": 1}, {"id": 2}]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'data'
    var_13 = var_9.value[var_12]
    var_14 = var_9.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = '{}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = '{"text": "Hello\\nWorld", "price": 19.99}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{name: "John"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John", "age": 30}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"text": "Hello\\nWorld"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '  \n  {  "name"  :  "John"  }  \n  '
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  \n  {"name": "John", "age": 30}  \n  '
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"null": null, "bool": true, "float": 3.14}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = "{name: 'John'}"
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '3.14'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'true'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'false'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'null'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = '[]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{invalid}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"a": 1 "b": 2}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"a" 1}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = b'{"name": "John", "age": 30}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = b'{"name": "J\xffhn", "age": 30}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = var_41.value
    var_43 = len(var_42)
    assert var_43 == 2



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "NYC"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'person'
    var_9 = var_5.value[var_8]
    var_10 = '{"names": ["John", "Jane"], "count": 2}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'names'
    var_15 = var_11.value[var_14]
    var_16 = '{"is_active": true, "balance": null, "score": 98.5}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John", "age": 30,}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = "{name: 'John'}"
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John", "age": 30,}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = b'{"name": "John", "age": 30}'
    var_27 = module_0.tokenize_json(var_26)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[{"name": "John"}, {"name": "Jane"}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = '[]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = '{"a": null, "b": true, "c": false}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"int": 42, "float": 3.14, "exp": 1e5}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John", "age": 30,}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{name: "John"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John"}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 13
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = 0
    var_9 = 14
    var_10 = '{}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = {}
    var_13 = 1
    var_14 = '{"a": {"b": "c"}}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = 11
    var_20 = module_1.ScalarToken(var_18, var_19, var_5, var_14)
    var_21 = {var_17: var_20}
    var_22 = 5
    var_23 = 15
    var_24 = '[1, 2, 3]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = module_1.ScalarToken(var_13, var_13, var_13, var_24)
    var_27 = 2
    var_28 = 4
    var_29 = module_1.ScalarToken(var_27, var_28, var_28, var_24)
    var_30 = 3
    var_31 = module_1.ScalarToken(var_30, var_4, var_4, var_24)
    var_32 = [var_26, var_29, var_31]
    var_33 = 8
    var_34 = module_1.ListToken(var_32, var_8, var_33, var_24)
    var_35 = '{"a": [1, "2"], "b": null}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = 6
    var_38 = module_1.ScalarToken(var_13, var_37, var_37, var_35)
    var_39 = '2'
    var_40 = 9
    var_41 = 10
    var_42 = module_1.ScalarToken(var_39, var_40, var_41, var_35)
    var_43 = [var_38, var_42]
    var_44 = module_1.ListToken(var_43, var_22, var_19, var_35)
    var_45 = None
    var_46 = 18
    var_47 = 21
    var_48 = module_1.ScalarToken(var_45, var_46, var_47, var_35)
    var_49 = {var_16: var_44, var_17: var_48}
    var_50 = 22
    var_51 = ''
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": "value"'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = b'{"key": "value"}'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = module_1.ScalarToken(var_3, var_4, var_5, var_53)
    var_58 = {var_2: var_57}
    var_59 = '  {  "key"  :  "value"  }  '
    var_60 = module_0.tokenize_json(var_59)
    var_61 = 17
    var_62 = module_1.ScalarToken(var_3, var_19, var_61, var_59)
    var_63 = {var_2: var_62}
    var_64 = 20



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json"], "count": 2}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{name: "John"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = "{'name': 'John'}"
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = b'{"name": "\xff\xfe", "age": 30}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"name": "John", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '  \n  {  "name"  :  "John"  }  \n  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John "The Boss" Doe", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"null": null, "true": true, "false": false}'
    var_27 = module_0.tokenize_json(var_26)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "Jöhn", "age": 30, "city": "New York"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": null}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"active": true, "deleted": false}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"price": 19.99, "quantity": 5}'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John" "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '  \n  {"name": "John", "age": 30}  \n  '
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "is_deleted": false}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John", "age": 30,}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{name: "John"}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "John" "age": 30}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = b'{"name": "John", "age": 30}'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = b'{"name": "John", "age": 30}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"person": {"name": "John", "age": 30}}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'person'
    var_15 = var_13.value[var_14]
    var_16 = '{"tags": ["python", "json"]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'tags'
    var_19 = var_17.value[var_18]
    var_20 = '{"price": 19.99, "quantity": 5}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"is_active": true, "data": null}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"id": 1, "name": "Alice"}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = '[]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = '  {  "key"  :  "value"  }  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": value}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"test": 123}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"escaped": "\\"quote\\"", "newline": "line1\\nline2"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"null": null, "bool_true": true, "bool_false": false}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John" /* comment */}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{name: "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = '  \n  \t  {"name": "John", "age": 30}  \n  \t  '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 2



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '  \n  {  "name"  :  "John"  }  \n  '
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  \t  {"name": "John", "age": 30}  \n  \t  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "is_student": false, "grades": [1, 2, 3]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 'grades'
    var_5 = var_1.value[var_4]
    var_6 = var_1.value[var_4]
    var_7 = var_6.value
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = '{"person": {"name": "Alice", "address": {"city": "NYC"}}}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 'person'
    var_12 = var_10.value[var_11]
    var_13 = 'address'
    var_14 = var_10.value[var_11]
    var_15 = var_14.value[var_13]
    var_16 = '[{"id": 1}, {"id": 2}]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = '{}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = '[]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = '{"text": "Hello\\nWorld", "tab": "Hello\\tWorld"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"name": "John"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"name": "John" "age": 30}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"name": "John",}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '   \n  \t  '
    var_39 = module_0.tokenize_json(var_38)
    var_40 = b'{"name": "John", "age": 30}'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = '{"null_value": null, "bool_value": true}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = '[]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = '{"value": null}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John", "age": 30'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John" "age": 30}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   \n  \t  '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = b'{"name": "John", "age": 30}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"name": "Jöhn", "emoji": "😀"}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"items": [1, 2, 3]}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"name": "John", "age": 30}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{name: "John"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John",}'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  \n  {"name": "John", "age": 30}  \t  '
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "is_admin": false}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"text": "Hello\\nWorld\\t!"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name": "John", "age": 30,}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{name: "John"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name" "John"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = b'{"name": "John", "age": 30}'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #42
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  {  "name"  :  "John"  }  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John\\nDoe", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"price": 19.99, "quantity": 5}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"is_active": true, "data": null}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #45
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #46
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #47
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John\\nDoe", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\u0040Doe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #48
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John", "age": 30,}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{name: "John"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name": "John", "emoji": "😀"}'
    var_34 = module_0.tokenize_json(var_33)



# Parsed testcases at query #49
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"null": null, "bool": true, "float": 1.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John",}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John"}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #50
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #51
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"text": "Hello\\nWorld", "price": 19.99}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #52
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 12
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = 0
    var_9 = 13
    var_10 = '{"outer": {"inner": "value"}}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'outer'
    var_13 = 'inner'
    var_14 = 18
    var_15 = 23
    var_16 = module_1.ScalarToken(var_3, var_14, var_15, var_10)
    var_17 = {var_13: var_16}
    var_18 = 8
    var_19 = 24
    var_20 = 25
    var_21 = '{"key": [1, 2, 3]}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = 1
    var_24 = 9
    var_25 = module_1.ScalarToken(var_23, var_24, var_24, var_21)
    var_26 = 2
    var_27 = 11
    var_28 = module_1.ScalarToken(var_26, var_27, var_27, var_21)
    var_29 = 3
    var_30 = module_1.ScalarToken(var_29, var_9, var_9, var_21)
    var_31 = [var_25, var_28, var_30]
    var_32 = 14
    var_33 = module_1.ListToken(var_31, var_18, var_32, var_21)
    var_34 = {var_2: var_33}
    var_35 = 15
    var_36 = '{"bool": true, "null": null, "num": 123}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'bool'
    var_39 = 'null'
    var_40 = 'num'
    var_41 = True
    var_42 = module_1.ScalarToken(var_41, var_24, var_5, var_36)
    var_43 = None
    var_44 = 21
    var_45 = module_1.ScalarToken(var_43, var_44, var_19, var_36)
    var_46 = 123
    var_47 = 36
    var_48 = 38
    var_49 = module_1.ScalarToken(var_46, var_47, var_48, var_36)
    var_50 = {var_38: var_42, var_39: var_45, var_40: var_49}
    var_51 = 39
    var_52 = ''
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{"key": "value"'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{key: "value"}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '{"key1": "value1" "key2": "value2"}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = b'{"key": "value"}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = module_1.ScalarToken(var_3, var_4, var_5, var_58)
    var_63 = {var_2: var_62}



# Parsed testcases at query #53
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  {  "name"  :  "John"  }  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John\\nDoe", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"price": 19.99, "quantity": 5}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"is_active": true, "data": null}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #54
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)



# Parsed testcases at query #55
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \t  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #56
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #57
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John\\nDoe", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\u0040Doe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"float": 3.14, "int": 42, "exp": 1e10}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"bool": true, "null": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #58
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = b'{"name": "John", "age": 30}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"name": "John", "age": 30'
    var_19 = module_0.tokenize_json(var_18)



# Parsed testcases at query #59
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #60
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json", "test"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30, "city": "New York"'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #61
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"text": "Hello\\nWorld", "tab": "Hello\\tWorld"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"unicode": "Hello\\u0020World"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{name: "John"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "John",}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = "{'name': 'John'}"
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"name": "John", /* comment */ "age": 30}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = b'{"name": "John", "age": 30}'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #62
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"name": "John", "age": 30}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'invalid json'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #63
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age":}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \t  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "The Boss" Doe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #64
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #65
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"null": null, "bool": true, "float": 1.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": }'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John"}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #66
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json", "test"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"data": null, "enabled": false, "version": 1.5}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = '  {  "key"  :  "value"  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{name: "John"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John", "age": 30,}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '   \n  \t  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = b'{"name": "John", "age": 30}'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #67
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three"]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #68
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = '{}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = '{"value": null}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{name: "John"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John", "age": 30}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name": "John "Doe"", "email": "john@example.com"}'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #69
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "NY"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John "Doe"", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John \\u00d1", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #70
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "city": "New York"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json"], "count": 2}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = '{"is_active": true, "balance": null, "score": 98.6}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": John, "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "J\xffhn", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #71
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, "three", {"four": 4}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = 3
    var_13 = var_9.value[var_12]
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = "{name: 'John'}"
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John", "age": 30,}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #72
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)



# Parsed testcases at query #73
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = var_3.value
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = 'key'
    var_8 = var_3.value[var_7]
    var_9 = '[1, "two", true]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_10.value[var_6]
    var_14 = 1
    var_15 = var_10.value[var_14]
    var_16 = 2
    var_17 = var_10.value[var_16]
    var_18 = '{"key": "value"'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = var_21.value[var_7]
    var_25 = '{"outer": {"inner": [1, 2, 3]}}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'outer'
    var_30 = var_26.value[var_29]
    var_31 = var_26.value[var_29]
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = var_26.value[var_29]
    var_35 = var_26.value[var_29]
    var_36 = 'inner'
    var_37 = var_26.value[var_29]
    var_38 = var_37.value[var_36]
    var_39 = var_26.value[var_29]
    var_40 = var_39.value[var_36]
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 3



# Parsed testcases at query #74
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "NY"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'person'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = '{"text": "Hello\\nWorld", "value": 123.45}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": John, "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #75
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{name: 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = "{name: 'John'}"
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John", /* comment */ "age": 30}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name": "John"} extra'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John", "age": 30}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #76
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"text": "Hello\\nWorld"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"float": 3.14, "int": 42, "exp": 1e5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"bool": true, "null": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #77
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = '[]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = '{"text": "Hello\\nWorld"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John"'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John",}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '   \n  \t  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #78
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3, "four"]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{name: "John"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"name": "John", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '  \n  { "name" : "John" , "age" : 30 }  \n  '
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json", "test"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"value": null}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John", "age": 30,}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = "{'name': 'John'}"
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John" "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John", age: 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John", "age": 30,}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = "{'name': 'John', 'age': 30}"
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John", /* comment */ "age": 30}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = b'{"name": "John", "age": 30}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"name": "Jöhn", "city": "New York"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"name": "John "The Boss" Doe", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"is_active": true, "is_admin": false}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"value": null}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = var_39.value
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = '{}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '[]'
    var_45 = module_0.tokenize_json(var_44)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John" "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John", "age": invalid}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '  \n  {"name": "John", "age": 30}  \n  '
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John"'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{name: "John"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   \n  \t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"text": "Hello\\nWorld"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{name: "John"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John" "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John",}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John"}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '[1, 2, 3, "four"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"text": "Hello\\nWorld", "price": 19.99}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{name: "John", age: 30}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '   \n  \t  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30}\xff'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = '[]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = '  {  "name"  :  "John"  }  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{name: "John"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John" "age": 30}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John",}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   \n  \t  '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = b'{"name": "John"}'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"null": null, "bool": true, "float": 1.5}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name" "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John",}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{name: "John"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'{"name": "John"}'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = '{"name": "John\\nDoe", "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John\\u0040Doe", "age": 30}'
    var_32 = module_0.tokenize_json(var_31)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json", "test"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"message": "Hello\\nWorld", "value": 123.45}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"data": null, "enabled": true, "disabled": false}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '"hello"'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '42'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = 'true'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = 'null'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '[1, 2, 3]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 3
    var_35 = '  { "name" : "John" , "age" : 30 }  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = b'{"name": "John", "age": 30}'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '[1, 2, "three"]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '  { "key" : "value" }  '
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"outer": {"inner": [1, 2, 3]}}'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{name: 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", /* comment */ "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John"} extra'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"name": "John \\u00f6", "age": 30}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '\n    {\n        "name": "John",\n        "age": 30\n    }\n    '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_48 = module_0.tokenize_json(var_47)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John\\nDoe", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": John, "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"text": "Hello\\nWorld", "value": 123.45}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"value": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "NY"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'person'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John", "age": 30,}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{'name': 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John" /* comment */}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '  \n  \r  \t  {"name": "John", "age": 30}  \n  \r  \t  '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John "The Boss" Doe", "path": "/home/user"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"integer": 42, "float": 3.14, "negative": -10, "scientific": 1.23e-4}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'name'
    var_5 = var_1.value[var_4]
    var_6 = 'age'
    var_7 = var_1.value[var_6]
    var_8 = '{"person": {"name": "John", "age": 30}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'person'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = '{"tags": ["python", "json"]}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 'tags'
    var_18 = var_16.value[var_17]
    var_19 = var_16.value[var_17]
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = '{"is_active": true, "balance": null}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John", "age": 30,}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{name: "John"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"text": "Hello\\nWorld"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"text": "Hello\\u0020World"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{name: "John"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "John", "age": 30,}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = "{'name': 'John'}"
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"name": "John", /* comment */ "age": 30}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = b'{"name": "John", "age": 30}'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"name": "John"'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{name: "John"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = ''
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '   '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = b'{"name": "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"text": "Hello\\nWorld"}'
    var_34 = module_0.tokenize_json(var_33)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3, "four"]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  \n  \t  {"name": "John"}  \r  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = b'{"name": "John", "age": 30}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '"hello"'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '42'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '3.14'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'true'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 'null'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, "three"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"data": {"nested": [1, 2, 3]}, "flag": true}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'data'
    var_13 = var_11.value[var_12]
    var_14 = 'nested'
    var_15 = var_11.value[var_12]
    var_16 = var_15.value[var_14]
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age":}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json", "test"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"price": 19.99, "quantity": 5, "discount": 0.15}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"data": null, "enabled": false, "verified": true}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John", "age": 30,}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 13
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = 0
    var_9 = 14
    var_10 = ''
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"key": "value"'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = module_1.ScalarToken(var_3, var_4, var_5, var_12)
    var_17 = {var_2: var_16}
    var_18 = '{"outer": {"inner": "value"}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'outer'
    var_21 = 'inner'
    var_22 = 17
    var_23 = 23
    var_24 = module_1.ScalarToken(var_3, var_22, var_23, var_18)
    var_25 = {var_21: var_24}
    var_26 = 8
    var_27 = 24
    var_28 = 25
    var_29 = '{"key": [1, 2, 3]}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = 1
    var_32 = 9
    var_33 = module_1.ScalarToken(var_31, var_32, var_32, var_29)
    var_34 = 2
    var_35 = 11
    var_36 = module_1.ScalarToken(var_34, var_35, var_35, var_29)
    var_37 = 3
    var_38 = module_1.ScalarToken(var_37, var_5, var_5, var_29)
    var_39 = [var_33, var_36, var_38]
    var_40 = module_1.ListToken(var_39, var_26, var_9, var_29)
    var_41 = {var_2: var_40}
    var_42 = 15
    var_43 = '{"key": "value\\nwith\\ttabs"}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = 'value\nwith\ttabs'
    var_46 = module_1.ScalarToken(var_45, var_4, var_23, var_43)
    var_47 = {var_2: var_46}



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "John", "age": 30,}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John"}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{invalid}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John"}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John Doe", "age": 30, "city": "New York"}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{name: "John"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"names": ["John", "Jane"], "age": 30}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'names'
    var_11 = var_9.value[var_10]
    var_12 = b'{"name": "John", "age": 30}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = "{name: 'John', age: 30}"
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John", "age": 30,}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #42
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "The Boss" Doe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"null": null, "bool_true": true, "bool_false": false}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = "{name: 'John'}"
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"name": "John", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '  \n  \t  {"name": "John", "age": 30}  \n  '
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #45
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3]}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'items'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  \n  {"name": "John", "age": 30}  \t  '
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #46
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "is_admin": false}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", "age": 30'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{name: "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '   \n  \t  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name": "John", "age": 30,}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = "{'name': 'John'}"
    var_40 = module_0.tokenize_json(var_39)
    var_41 = b'{"name": "John", "age": 30}'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #47
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John Doe", "age": 30, "city": "New York"}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #48
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = b'{"name": "John", "age": 30}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30,}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "address": {"city": "New York"}}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'address'
    var_17 = var_13.value[var_16]
    var_18 = '[{"name": "John"}, {"name": "Jane"}]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2



# Parsed testcases at query #49
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 42, "bool": true, "null": null}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"outer": {"inner": [1, 2, 3]}, "array": [{"a": 1}, {"b": 2}]}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"key": "value",}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = b'{"key": "value"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '  \n  \t  {"key": "value"}  \n  '
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"key": "value with spaces and \\"quotes\\""}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #50
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'user'
    var_9 = var_5.value[var_8]
    var_10 = '{"items": [1, 2, 3], "name": "test"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'items'
    var_15 = var_11.value[var_14]
    var_16 = var_11.value[var_14]
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{name: "John"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"name": "John", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2



# Parsed testcases at query #51
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"null": null, "bool": true, "float": 1.5}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = "{name: 'John'}"
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John"}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #52
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, "three"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30,}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"name": "John", "age": 30}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2



# Parsed testcases at query #53
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #54
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = "{'name': 'John'}"
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{name: 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John" /* comment */}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '  \n  {\n    "name": "John"\n  }  \n'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #55
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3, "four", {"five": 5}]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = 4
    var_13 = var_9.value[var_12]
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = '[]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = '{"special": "line1\\nline2\\t\\r", "unicode": "日本語"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"name": "John", "age": 30,}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{name: "John"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #56
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, "three"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30,}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"name": "John", "age": 30}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = b'\x80\x81'
    var_19 = module_0.tokenize_json(var_18)



# Parsed testcases at query #57
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"text": "Hello\\nWorld", "price": 19.99}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John"'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = "{name: 'John'}"
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'\xff\xfe{"name": "John"}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #58
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, "three"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"'
    var_9 = var_5.value
    var_10 = var_9 == var_2
    var_11 = ''
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '{"name": "John", "age": 30,}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = '{"name": "John", "age": 30'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = b'{"name": "John", "age": 30}'
    var_18 = module_0.tokenize_json(var_17)



# Parsed testcases at query #59
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  \n  {"name": "John", "age": 30}  \n  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"integer": 42, "float": 3.14, "scientific": 1.23e+4}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #60
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = '{"null_value": null, "bool_value": true}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = "{name: 'John'}"
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '  {  "name"  :  "John"  }  '
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #61
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = '[]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = '  {  "name"  :  "John"  }  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John"'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{name: "John"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '   \n  \t  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John"}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #62
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John "Doe"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #63
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": "thirty"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John",}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = "{'name': 'John'}"
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John" /* comment */}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #64
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"text": "Hello\\nWorld"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"int": 42, "float": 3.14}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"bool": true, "null": null}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #65
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "name"  :  "John"  }  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John\\nDoe", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"price": 19.99, "quantity": 5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "data": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #66
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json"], "counts": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '  \n  {"name": "John", "age": 30}  \t  '
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #67
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 'key'
    var_5 = var_1.value[var_4]
    var_6 = '{"outer": {"inner": "value"}}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'outer'
    var_9 = var_7.value[var_8]
    var_10 = '[1, 2, "three"]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 3
    var_14 = '{"bool": true, "null": null, "num": 42.5}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"key": "value"'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"key": "value",}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key" "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{key: "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = "{'key': 'value'}"
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value" /* comment */}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"key": "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = b'{"key": "\xff\xfe"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '  \n  \r  \t  {"key": "value"}  \n  '
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #68
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #69
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "is_student": false}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"id": 1, "tags": ["a", "b"]}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = 'tags'
    var_9 = var_5.value[var_6]
    var_10 = var_9.value[var_8]
    var_11 = ''
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '{"name": "John", "age": 30,}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = b'{"name": "John", "age": 30}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '  \n  { "key" : "value" }  \t  '
    var_18 = module_0.tokenize_json(var_17)



# Parsed testcases at query #70
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 0
    var_19 = '[]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '{"value": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "is_admin": false}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"scientific": 1.23e+4}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"name": "John", "age": 30,}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{name: "John"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name" "John"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = b'{"name": "John", "age": 30}'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #71
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  \n  {"name": "John", "age": 30}  \n  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #72
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 42, "bool": true, "null": null}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = '{"outer": {"inner": [1, 2, 3]}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'outer'
    var_7 = var_5.value[var_6]
    var_8 = 'inner'
    var_9 = var_5.value[var_6]
    var_10 = var_9.value[var_8]
    var_11 = var_5.value[var_6]
    var_12 = var_11.value[var_8]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"invalid": json}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"key": "value"}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '  {  "key"  :  "value"  }  '
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #73
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"user": {"name": "John", "age": 30}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": John, "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John\\nDoe", "age": 30}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"null": null, "true": true, "false": false}'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #74
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '[1, 2, "three"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 3
    var_8 = '"hello"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = ''
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "John", "age": 30,}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"name": "John", "age": 30}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = '{"person": {"name": "John", "age": 30}, "numbers": [1, 2, 3]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'person'
    var_23 = var_19.value[var_22]
    var_24 = 'numbers'
    var_25 = var_19.value[var_24]
    var_26 = '{"special": "new\\nline", "tab": "new\\ttab"}'
    var_27 = module_0.tokenize_json(var_26)



# Parsed testcases at query #75
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"person": {"name": "John", "age": 30}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John "Doe"", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"int": 42, "float": 3.14, "exp": 1.23e-4}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #76
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"user": {"name": "Alice", "age": 25}, "active": true}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"items": [1, 2, 3], "name": "Test"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'items'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": John, "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #77
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '[1, 2, 3, "four"]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 4
    var_8 = '{"outer": {"inner": [1, 2, 3]}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'outer'
    var_11 = var_9.value[var_10]
    var_12 = 'inner'
    var_13 = var_9.value[var_10]
    var_14 = var_13.value[var_12]
    var_15 = '{"null": null, "bool": true, "float": 3.14}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = "{name: 'John'}"
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



