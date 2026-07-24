####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": [1, {"inner": "value"}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'nested'
    var_19 = var_17.value[var_18]
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "value"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.5e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"emoji": "😀"}'
    var_39 = 'utf-8'



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "unterminated'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{key: "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "café"}'
    var_19 = 'utf-8'
    var_20 = '{"outer": {"inner": [1, 2, 3]}}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key" "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3,]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'users'
    var_35 = var_33.value[var_34]
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = '1e10'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '-42'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"array": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n  \t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid json}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"} extra'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = module_0.tokenize_json(var_28)
    var_35 = len(var_28)
    var_36 = 1
    var_37 = var_35 - var_36



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key" "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"arr": [1, 2, {"nested": true}], "val": null}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 'arr'
    var_29 = var_27.value[var_28]
    var_30 = '1.5e-10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = module_0.tokenize_json(var_24)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"outer": {"inner": "value"}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'outer'
    var_27 = var_25.value[var_26]
    var_28 = '[[1, 2], [3, 4]]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 0
    var_33 = var_29.value[var_32]
    var_34 = '{"key": "value with \\"quotes\\""}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"name": "José"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '1.23e-4'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '-42'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": "value"'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '[1, 2, ]'
    var_45 = module_0.tokenize_json(var_44)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": "value"}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = '1.5e-10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = module_0.tokenize_json(var_22)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2, {"nested": true}], "str": "test"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"emoji": "😀"}'
    var_29 = 'utf-8'
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e10'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": [1, 2, {"deep": "value"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 'nested'
    var_29 = var_27.value[var_28]
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '[]'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"nested": [1, 2, {"deep": "value"}]}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"key": "café"}'
    var_19 = 'utf-8'
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key" "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '3.14'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1e10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key" "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e-10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"string": "value", "number": 123, "float": 45.67, "bool": true, "null": null, "array": [1, 2], "object": {"nested": "data"}}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '3.14'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '1e5'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'utf-8'
    var_19 = '{"outer": {"inner": [1, 2, 3]}}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = ''
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '   \n\t  '
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{invalid}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"key": "value"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '[1, 2, 3,]'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = 'users'
    var_34 = var_32.value[var_33]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '-3.14'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = 'false'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e5'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": [1, {"inner": true}], "value": null}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-3.14'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '[1, 2,]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"outer": {"inner": [1, 2, {"nested": true}]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"unicode": "café"}'
    var_33 = 'utf-8'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '3.14'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '1e10'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = '{"a": {"b": {"c": "d"}}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'false'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2, {"nested": true}], "val": null}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"invalid": }'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{key: "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = b'{"key": "value with \xc3\xa9"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '[]'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": }'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1e5'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "café"}'
    var_35 = 'utf-8'



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '"hello"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '3.14'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'false'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'null'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"array": [1, 2, {"nested": true}], "value": null}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value",}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key" "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "café"}'
    var_31 = 'utf-8'
    var_32 = module_0.tokenize_json(var_2)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"invalid json}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": true}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = '1.5e-10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"a": [1, 2, {"b": "c"}], "d": null}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = len(var_34)
    assert var_35 == 2



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '1.5e-10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"string": "value", "number": 123, "array": [true, false, null], "nested": {"key": "val"}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = len(var_34)
    assert var_35 == 4



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": {"key": [1, 2, 3]}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value",}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"arr": [1, "two", 3.0, true, false, null], "obj": {"nested": "value"}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '1.5e10'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"key": "café"}'
    var_41 = 'utf-8'



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '[1, 2,'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": {"key": [1, 2, 3]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 'nested'
    var_29 = var_27.value[var_28]
    var_30 = 'key'
    var_31 = var_27.value[var_28]
    var_32 = var_31.value[var_30]
    var_33 = module_0.tokenize_json(var_24)
    var_34 = '1e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"array": [1, 2], "nested": {"key": "val"}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": }'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '123 456'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"data": [{"id": 1, "name": "test"}, {"id": 2}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.5e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "café"}'
    var_39 = 'utf-8'



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"nested": [1, 2, {"key": "value"}]}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e10'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n  \t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = b'{"key": "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value with unicode: é"}'
    var_27 = 'utf-8'
    var_28 = '{"key" "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3,]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"string": "value", "number": 42, "float": 3.14, "bool": true, "null": null, "array": [1, 2, 3], "nested": {"key": "val"}}'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "café"}'
    var_19 = 'utf-8'
    var_20 = '{"outer": {"inner": 42}}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'outer'
    var_23 = var_21.value[var_22]
    var_24 = '  {  "key"  :  "value"  }  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '   \n\t  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{invalid}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": }'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "unterminated'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[1, 2, 3,]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '1e10'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '-42'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '-3.14'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"array": [1, "two", {"three": true}], "null_val": null}'
    var_45 = module_0.tokenize_json(var_44)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e10'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = None
    var_21 = ''
    var_22 = module_0.tokenize_json(var_21)
    var_23 = None
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = None
    var_27 = '{invalid}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"array": [1, "two", {"three": 3}], "null": null}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '-42'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '-3.14'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"name": "José"}'
    var_36 = 'utf-8'
    var_37 = '{}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '[]'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, 2, {"deep": true}]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '1.23e-4'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = module_0.tokenize_json(var_22)
    var_31 = '{"emoji": "😀"}'
    var_32 = 'utf-8'



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n  \t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": "value"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": [1, {"inner": "value"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '{"key": "value with \\"quotes\\""}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"name": "café"}'
    var_41 = 'utf-8'



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '   \n\t  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"key": }'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{key: "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": "value"}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = '3.14'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": "value"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"outer": {"inner": [1, 2, {"nested": true}]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, 2, {"inner": "value"}]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = 'array'
    var_29 = var_25.value[var_26]
    var_30 = var_29.value[var_28]
    var_31 = '1.5e10'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '-42'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = module_0.tokenize_json(var_22)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"arr": [1, 2], "obj": {"nested": true}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"name": "José"}'
    var_21 = 'utf-8'
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": }'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2,]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "unclosed'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"users": [{"id": 1, "name": "Alice", "active": true}, {"id": 2, "name": "Bob", "active": false}]}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '1.23e-4'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '-42'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"array": [1, "two", null], "nested": {"key": true}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '1.5e-10'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'"hello\\u00e9"'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, "two", true, null]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '1.5e-10'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "value"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '[1, 2, 3,]'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "café"}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e10'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, 2,'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": {"key": [1, 2, 3]}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'nested'
    var_31 = var_29.value[var_30]
    var_32 = 'key'
    var_33 = var_29.value[var_30]
    var_34 = var_33.value[var_32]
    var_35 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = 'users'
    var_38 = var_36.value[var_37]
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = '-42'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"key": "café"}'
    var_44 = 'utf-8'



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key":'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"outer": {"inner": [1, 2, {"deep": "value"}]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e10'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": {"inner": [1, 2, 3]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"str": "value", "num": 42, "bool": true, "null": null, "arr": [1, 2]}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e-10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid json}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '1.5e10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = module_0.tokenize_json(var_26)



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "José"}'
    var_19 = 'utf-8'
    var_20 = '{"array": [1, 2, {"nested": true}]}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{invalid}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key" "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '[1, 2,]'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, 2, ]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": [1, {"inner": "value"}], "bool": true}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e-10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"emoji": "😀"}'
    var_35 = 'utf-8'



# Parsed testcases at query #42
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '123'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e5'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"a": [1, 2, {"b": "c"}]}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"invalid": }'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{invalid}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "unterminated'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '[1, 2, 3,]'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"emoji": "😀"}'
    var_35 = 'utf-8'



# Parsed testcases at query #43
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"array": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'array'
    var_21 = var_19.value[var_20]
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value" "key2": "value2"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3,]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '  {  "key"  :  "value"  }  '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{\n  "key": "value"\n}'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #44
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"outer": {"inner": [1, 2, {"deep": "value"}]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '[1, "string", true, null, 3.14]'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = len(var_34)
    assert var_35 == 5



# Parsed testcases at query #45
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"key": "café"}'
    var_19 = 'utf-8'
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": {"key": [1, 2, 3]}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = module_0.tokenize_json(var_26)
    var_35 = 'start_position'
    var_36 = hasattr(var_34, var_35)
    var_37 = 'end_position'
    var_38 = hasattr(var_34, var_37)



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"key": "café"}'
    var_19 = 'utf-8'
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, 2,'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": {"array": [1, 2, {"key": "value"}]}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = module_0.tokenize_json(var_26)
    var_31 = '1.5e-10'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '-42'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '[]'
    var_38 = module_0.tokenize_json(var_37)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"array": [1, 2, {"nested": true}], "value": null}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = '-123'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '"hello\\"world"'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, 2, {"deep": "value"}]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = module_0.tokenize_json(var_22)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": {"key": [1, 2, 3]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"str": "value", "num": 42, "bool": true, "null": null, "arr": [1, "two"]}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "café"}'
    var_35 = 'utf-8'



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n  \t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key" "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-3.14'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": }'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key" "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, 2, 3,]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"array": [1, 2, {"nested": true}], "string": "test"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e2'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"emoji": "😀"}'
    var_35 = 'utf-8'



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"name": "John", "age": 30, "items": [1, 2, 3]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key":'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key": "café"}'
    var_37 = 'utf-8'



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": {"array": [1, 2, {"deep": true}]}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{invalid}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"invalid": }'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": "value"}], "number": 42}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = module_0.tokenize_json(var_22)
    var_29 = '1.5e10'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '-42'
    var_32 = module_0.tokenize_json(var_31)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '  { "key" : "value" }  '
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '1e5'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "value",}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"outer": {"inner": [1, 2, {"deep": "value"}]}}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key": "value\\"with\\"quotes"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '-42'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '-3.14'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"array": [1, 2, {"nested": true}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '1.5e-10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '[]'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '  {"key": "value"}  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key" "value"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.5e10'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-42'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = 'users'
    var_41 = var_39.value[var_40]
    var_42 = len(var_41)
    assert var_42 == 2



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": {"array": [1, 2, {"inner": "value"}]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1e10'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = ''
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '   \n  \t  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{invalid}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '[1, 2,]'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"outer": {"inner": [1, 2, 3]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'outer'
    var_27 = var_25.value[var_26]
    var_28 = 'inner'
    var_29 = var_25.value[var_26]
    var_30 = var_29.value[var_28]
    var_31 = '3.14'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '1e10'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '-42'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"name": "café"}'
    var_38 = 'utf-8'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e-10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, 2, 3]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = 'array'
    var_29 = var_25.value[var_26]
    var_30 = var_29.value[var_28]
    var_31 = '{"name": "John", "age": 30, "active": true, "balance": null}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '1.5e10'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '-42'
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": [1, {"inner": "value"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 'nested'
    var_29 = var_27.value[var_28]
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e10'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"arr": [1, 2], "nested": {"key": "val"}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": }'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"key": "value\xff"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"nested": [1, {"key": "value"}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid: "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e-10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "café"}'
    var_35 = 'utf-8'



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, "two", true, null]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = module_0.tokenize_json(var_22)
    var_27 = '1e10'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '-42'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "café"}'
    var_19 = 'utf-8'
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2, 3'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '1e10'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = module_0.tokenize_json(var_30)
    var_39 = len(var_30)
    var_40 = 1
    var_41 = var_39 - var_40



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n  \t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": "value"} extra'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"outer": {"inner": 123}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[[1, 2], [3, 4]]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"arr": [1, "two", null, true], "num": -42.5e2}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-100'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.5e-3'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"emoji": "🎉"}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"outer": {"inner": [1, 2, 3]}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.23e-4'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = b'{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"arr": [1, 2], "obj": {"nested": true}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '3.14'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '1e10'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "unterminated'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key" "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "value"} extra'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-3.14'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, "two", true, null]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = 'array'
    var_29 = var_25.value[var_26]
    var_30 = var_29.value[var_28]
    var_31 = '{"a": 1}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '1.5e10'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '-42'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = 'utf-8'



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, "two", true, null]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = module_0.tokenize_json(var_22)
    var_27 = '1.5e-10'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '-42'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"key": "value with unicode: 你好"}'
    var_32 = 'utf-8'



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '3.14'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '1e10'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = '{"str": "text", "num": 123, "bool": true, "null": null, "arr": [1, 2], "obj": {}}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 6
    var_38 = '-42'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '-3.14'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '[]'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"key": "café"}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = var_47.value
    var_49 = str(var_48)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"invalid": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "value"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{invalid}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e-10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = module_0.tokenize_json(var_28)
    var_37 = 'start_position'
    var_38 = hasattr(var_36, var_37)
    var_39 = 'end_position'
    var_40 = hasattr(var_36, var_39)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"array": [1, "two", null]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = module_0.tokenize_json(var_22)
    var_27 = '1.5e-10'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '-42'
    var_30 = module_0.tokenize_json(var_29)



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"outer": {"inner": [1, 2]}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key" "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'users'
    var_33 = var_31.value[var_32]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '1.5e10'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '-42'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '[]'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": "value"}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '1.5e10'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '-42'
    var_29 = module_0.tokenize_json(var_28)



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": invalid}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key":'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.5e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = module_0.tokenize_json(var_28)
    var_35 = '{"key": "café"}'
    var_36 = 'utf-8'



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"nested": {"key": [1, 2, 3]}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[{"a": 1}, {"b": 2}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = '1e10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key": "café"}'
    var_37 = 'utf-8'



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": {"key": [1, 2, 3]}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = 'key'
    var_29 = var_25.value[var_26]
    var_30 = var_29.value[var_28]
    var_31 = '{"string": "value", "number": 42, "bool": true, "null": null, "array": [1, 2], "object": {}}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 6
    var_35 = '1e10'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '-42'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": "café"}'
    var_40 = 'utf-8'



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": "value"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key" "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"array": [1, 2, {"nested": true}], "string": "test"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '1.5e-10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "café"}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": }'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, 2,]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"nested": [1, {"inner": true}], "value": null}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'nested'
    var_31 = var_29.value[var_30]
    var_32 = '1.5e10'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key": "café"}'
    var_37 = 'utf-8'



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"nested": [1, {"inner": "value"}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'nested'
    var_21 = var_19.value[var_20]
    var_22 = '  { "key" : "value" }  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '   \n\t  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{invalid}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": }'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "value"'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-123'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '1.5e10'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '[]'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"nested": [1, 2, {"inner": "value"}]}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '1e10'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2, 3,]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "value"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{key: "value"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '-42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '-3.14'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"a": [1, 2, {"b": "c"}], "d": null}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '[]'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"key": "こんにちは"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'utf-8'



# Parsed testcases at query #40
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'true'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = ''
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '   \n\t  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{invalid}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"nested": [1, {"inner": "value"}]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = '1.5e10'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '-42'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #41
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"hello"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"nested": [1, {"inner": "value"}]}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '1.5e10'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"key": "value"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n  \t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2,]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "value",}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"users": [{"id": 1, "name": "Alice", "active": true}, {"id": 2, "name": "Bob", "active": false}]}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = 'users'
    var_37 = var_35.value[var_36]
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = '{"message": "Hello\\nWorld"}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '-42'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '[-1, -3.14, -1e5]'
    var_44 = module_0.tokenize_json(var_43)



