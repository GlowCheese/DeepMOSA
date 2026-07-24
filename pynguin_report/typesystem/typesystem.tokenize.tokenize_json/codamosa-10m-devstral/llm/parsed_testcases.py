####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_15 = '{"active": true, "data": null}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age":}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)



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
    var_23 = '{"name": "John\\x"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"name": "John}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"items": [1, 2, 3'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"age": 30.0.0}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"active": tr}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"active": fal}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"value": nu}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = b'{"name": "John", "age": 30}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '  \n  \t  {"name": "John", "age": 30}  \n  \t  '
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #3
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



# Parsed testcases at query #4
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
    var_8 = '{"names": ["John", "Jane", "Doe"], "count": 3}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'names'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = '{"is_active": true, "balance": 123.45, "data": null}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = ''
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"name": "John", "age": 30,}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{name: "John"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "John", "age": 30,}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{name: "John"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = "{'name': 'John'}"
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John" /* comment */}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"name": "John"} extra'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)



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
    var_27 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_28 = module_0.tokenize_json(var_27)



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



# Parsed testcases at query #7
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
    var_12 = 'counts'
    var_13 = var_9.value[var_12]
    var_14 = var_9.value[var_10]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_9.value[var_12]
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John", "age": 30,}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = "{name: 'John'}"
    var_25 = module_0.tokenize_json(var_24)
    var_26 = b'{"name": "John", "age": 30}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"name": "John "Doe"", "age": 30}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"value": null, "flag": true, "status": false}'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #8
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
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #9
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
    var_8 = '{}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = '[]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 0
    var_16 = '{"nested": {"array": [1, 2, {"key": "value"}]}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'nested'
    var_19 = var_17.value[var_18]
    var_20 = 'array'
    var_21 = var_17.value[var_18]
    var_22 = var_21.value[var_20]
    var_23 = var_17.value[var_18]
    var_24 = var_23.value[var_20]
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 3
    var_27 = b'{"test": "bytes"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"invalid": }'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"key" "value"}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": "value",}'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #10
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
    var_23 = '{"price": 19.99, "quantity": 5}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"active": true, "data": null}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #11
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
    var_19 = '{"name": "John"'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = "{name: 'John'}"
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"name": "John", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"integer": 42, "float": 3.14, "scientific": 1.23e+5}'
    var_28 = module_0.tokenize_json(var_27)



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



# Parsed testcases at query #13
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
    var_18 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2



# Parsed testcases at query #14
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
    var_8 = '[1, 2, "three"]'
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
    var_20 = '{"value": null}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John", "age": 30'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{name: "John"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '   \n  \t  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)



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
    var_20 = '{name: "John"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"name": "John"}'
    var_23 = module_0.tokenize_json(var_22)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "is_student": false, "grades": [90, 85, 88]}'
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
    var_9 = '{"person": {"name": "Alice", "age": 25}, "city": "New York"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 'person'
    var_12 = var_10.value[var_11]
    var_13 = '[1, 2, 3, "four", {"five": 5}]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 5
    var_17 = 4
    var_18 = var_14.value[var_17]
    var_19 = '{}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = '[]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 0
    var_27 = '  {  "key"  :  "value"  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", "age": 30'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{name: "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'{"name": "John", "age": 30}'
    var_36 = module_0.tokenize_json(var_35)



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
    var_22 = '{"special": "new\\nline", "tab": "a\\tb"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John", "age": 30'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{name: "John"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   \n  \t  '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"name": "John", "age": 30,}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = b'{"name": "John", "age": 30}'
    var_35 = module_0.tokenize_json(var_34)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_23 = '{"a": null, "b": true, "c": false}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"int": 42, "float": 3.14, "exp": 1e5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = ''
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John", "age": 30'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{name: "John"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"name": "John", "age": 30}'
    var_34 = module_0.tokenize_json(var_33)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30, "is_student": false}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 3
    var_4 = '{"person": {"name": "John", "age": 30}, "is_student": false}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'person'
    var_7 = var_5.value[var_6]
    var_8 = '{"names": ["John", "Jane"], "age": 30}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'names'
    var_11 = var_9.value[var_10]
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30,}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": 30}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #3
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
    var_12 = '{"text": "Hello\\nWorld"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30,}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{name: "John"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "John",}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"name": "John\\x"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = "{'name': 'John'}"
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"name": "John" /* comment */}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"name": "John", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '  {  "name"  :  "John"  }  '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{\n\t"name": "John"\n}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"is_active": true, "is_admin": false}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"value": null}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"value": 1.23e+4}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = var_45.value
    var_47 = len(var_46)
    assert var_47 == 0
    var_48 = '[]'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = var_49.value
    var_51 = len(var_50)
    assert var_51 == 0
    var_52 = '{"matrix": [[1, 2], [3, 4]]}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'matrix'
    var_55 = var_53.value[var_54]
    var_56 = var_53.value[var_54]
    var_57 = var_56.value
    var_58 = len(var_57)
    assert var_58 == 2
    var_59 = 0
    var_60 = var_53.value[var_54]
    var_61 = var_60.value[var_59]



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
    var_18 = b'{"name": "John", "age": 30}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"name": "John", "age": \x80}'
    var_21 = module_0.tokenize_json(var_20)



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
    var_20 = b'{"name": "John"}'
    var_21 = module_0.tokenize_json(var_20)



# Parsed testcases at query #7
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
    var_32 = '{"name": "John \\"The Boss\\"", "age": 30}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"name": null}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"active": true, "deleted": false}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"scientific": 1.23e+10}'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_8 = '[1, 2, 3, "four"]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 4
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
    var_20 = '  {  "key"  :  "value"  }  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"name": "John", "age": 30'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{name: "John"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = b'{"name": "John", "age": 30}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"name": "Jöhn", "age": 30}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"is_active": true, "data": null}'
    var_35 = module_0.tokenize_json(var_34)



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
    var_10 = '{"tags": ["python", "json"], "count": 2}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'tags'
    var_13 = var_11.value[var_12]
    var_14 = var_11.value[var_12]
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
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
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #12
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
    var_22 = '{"value": null}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"is_active": true, "is_admin": false}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"name": "John", "age": 30,}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{name: "John"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"name" "John"}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = b'{"name": "John", "age": 30}'
    var_37 = module_0.tokenize_json(var_36)



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
    var_17 = '{"name": "John", "age": 30'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John\\nDoe", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"name": "Jöhn", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"float": 3.14, "int": 42, "exp": 1e5}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"is_active": true, "is_admin": false, "middle_name": null}'
    var_28 = module_0.tokenize_json(var_27)



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = '{"data": {"id": 1, "items": [1, 2, 3]}}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'data'
    var_7 = var_5.value[var_6]
    var_8 = 'items'
    var_9 = var_5.value[var_6]
    var_10 = var_9.value[var_8]
    var_11 = var_5.value[var_6]
    var_12 = var_11.value[var_8]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '{"name": "John "The Boss" Doe", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = '[{"name": "John"}, {"name": "Jane"}]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '{"float": 3.14, "int": 42, "exp": 1.23e-4}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 3
    var_39 = '{"bool": true, "null": null, "false": false}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 3



# Parsed testcases at query #16
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
    var_7 = '[1, 2, "three"]'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = var_8.value
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = '"hello"'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '{"key": "value"'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = b'{"key": "value"}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = '  {  "key"  :  "value"  }  '
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = '{"outer": {"inner": [1, 2, 3]}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = '{"key": "value\\nwith\\tescapes"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 1



# Parsed testcases at query #17
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
    var_23 = '{"name": "John \\"Doe\\"", "age": 30}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"float": 3.14, "int": 42, "exp": 1.23e-4}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"null": null, "true": true, "false": false}'
    var_28 = module_0.tokenize_json(var_27)



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
    var_6 = 'user'
    var_7 = var_5.value[var_6]
    var_8 = '{"tags": ["python", "json"], "count": 2}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'tags'
    var_11 = var_9.value[var_10]
    var_12 = var_9.value[var_10]
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = ''
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = b'{"name": "John", "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"message": "Hello\\nWorld", "value": 123.45}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"data": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"is_active": true, "is_admin": false}'
    var_26 = module_0.tokenize_json(var_25)



# Parsed testcases at query #19
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
    var_17 = '{"name": "John", "age": 30,}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = "{name: 'John'}"
    var_20 = module_0.tokenize_json(var_19)
    var_21 = b'{"name": "John", "age": 30}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = '  \n  \t  {"name": "John", "age": 30}  \n  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2



# Parsed testcases at query #20
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
    var_19 = '{"name": "John" "age": 30}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"name": "John", "age": 30,}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = "{'name': 'John'}"
    var_24 = module_0.tokenize_json(var_23)
    var_25 = "{name: 'John'}"
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", /* comment */ "age": 30}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"name": "John"} extra content'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = b'{"name": "John", "age": 30}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '{"name": "Jöhn", "emoji": "😀"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"text": "Line 1\\nLine 2\\tTab"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"integer": 42, "float": 3.14, "negative": -10, "scientific": 1.23e+5}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"null_value": null, "bool_true": true, "bool_false": false}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"empty_object": {}, "empty_array": []}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = 'empty_object'
    var_46 = var_44.value[var_45]
    var_47 = var_44.value[var_45]
    var_48 = var_47.value
    var_49 = len(var_48)
    assert var_49 == 0
    var_50 = 'empty_array'
    var_51 = var_44.value[var_50]
    var_52 = var_44.value[var_50]
    var_53 = var_52.value
    var_54 = len(var_53)
    assert var_54 == 0



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

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
    var_10 = '{"key": {"nested": [1, 2, 3]}}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'nested'
    var_13 = 1
    var_14 = 19
    var_15 = module_1.ScalarToken(var_13, var_14, var_14, var_10)
    var_16 = 2
    var_17 = 21
    var_18 = module_1.ScalarToken(var_16, var_17, var_17, var_10)
    var_19 = 3
    var_20 = 23
    var_21 = module_1.ScalarToken(var_19, var_20, var_20, var_10)
    var_22 = [var_15, var_18, var_21]
    var_23 = 18
    var_24 = 24
    var_25 = module_1.ListToken(var_22, var_23, var_24, var_10)
    var_26 = {var_12: var_25}
    var_27 = 25
    var_28 = 26
    var_29 = b'{"key": "value"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_32 = {var_2: var_31}
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = module_2.Position(var_13, var_13, var_8)
    var_36 = '{"key": "value"'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = module_2.Position(var_13, var_9, var_5)
    var_39 = b'{"key": "value"'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = module_2.Position(var_13, var_9, var_5)



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
    var_8 = '{"items": [1, 2, 3]}'
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
    var_19 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"value": null}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{"is_active": true, "is_admin": false}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"integer": 42, "float": 3.14, "scientific": 1.23e-4}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"name": "John", "age": 30}'
    var_28 = module_0.tokenize_json(var_27)



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



