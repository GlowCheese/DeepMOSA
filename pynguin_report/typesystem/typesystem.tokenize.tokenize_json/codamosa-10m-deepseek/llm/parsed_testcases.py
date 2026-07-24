####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 13
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 13
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 13
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = var_3.value[var_4]
    var_6 = '{"key": "value"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"key": [1, 2, 3]}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value[var_4]



# Parsed testcases at query #5
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
    var_6 = '{"key": "value"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = b'{"key": "value"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"string"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '-123.45'
    var_15 = module_0.tokenize_json(var_14)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '{"nested": {"key": "value"}}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '[1, 2, "three"]'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = var_12.value[var_8]
    var_16 = 1
    var_17 = var_12.value[var_16]
    var_18 = 2
    var_19 = var_12.value[var_18]
    var_20 = b'{"bytes": "input"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'All tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'key'
    var_9 = var_5.value[var_8]
    var_10 = b'{"key": "value"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_11.value[var_8]
    var_15 = '[1, 2, "three"]'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = 0
    var_20 = var_16.value[var_19]
    var_21 = 1
    var_22 = var_16.value[var_21]
    var_23 = 2
    var_24 = var_16.value[var_23]
    var_25 = 'All tests passed!'
    var_26 = print(var_25)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"name": "Alice", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'name'
    var_5 = var_1.value[var_4]
    var_6 = 'age'
    var_7 = var_1.value[var_6]
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "Alice", "age": 30'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"name": "Alice", "details": {"age": 30, "city": "New York"}}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'details'
    var_15 = var_13.value[var_14]
    var_16 = var_13.value[var_14]
    var_17 = var_16.value[var_6]
    var_18 = 'city'
    var_19 = var_13.value[var_14]
    var_20 = var_19.value[var_18]
    var_21 = '{"name": "Alice", "scores": [85, 90, 78]}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = 'scores'
    var_24 = var_22.value[var_23]
    var_25 = 0
    var_26 = var_22.value[var_23]
    var_27 = var_26.value[var_25]
    var_28 = 1
    var_29 = var_22.value[var_23]
    var_30 = var_29.value[var_28]
    var_31 = 2
    var_32 = var_22.value[var_23]
    var_33 = var_32.value[var_31]
    var_34 = 'All test cases passed!'
    var_35 = print(var_34)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"string"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '123'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = ''
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"key": "value"'
    var_17 = module_0.tokenize_json(var_16)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 14
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = var_5.value[var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 0
    var_11 = var_9.value[var_10]
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'true'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'null'
    var_17 = module_0.tokenize_json(var_16)



# Parsed testcases at query #14
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
    var_8 = len(var_0)
    var_9 = 1
    var_10 = var_8 - var_9
    var_11 = ''
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '{invalid}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = b'{"key": "value"}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = '[1, 2, "three"]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = var_14.value[var_8]
    var_18 = 1
    var_19 = var_14.value[var_18]
    var_20 = 2
    var_21 = var_14.value[var_20]
    var_22 = '{"nested": {"key": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = '{"number": 123.45}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '{"true": true, "false": false, "null": null}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 3



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '[1, "two", false]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_10.value[var_8]
    var_14 = 1
    var_15 = var_10.value[var_14]
    var_16 = 2
    var_17 = var_10.value[var_16]
    var_18 = b'{"bytes": "test"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'All tokenize_json tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 8
    var_7 = 14
    var_8 = module_1.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = {var_4: var_8}
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)



# Parsed testcases at query #18
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
    var_6 = '[1, 2, 3]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_7.value
    var_11 = '"hello"'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '42'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = 'true'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 'null'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = ''
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"key": "value"'
    var_22 = module_0.tokenize_json(var_21)



# Parsed testcases at query #19
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
    var_7 = var_3.value[var_6][var_6]
    var_8 = 1
    var_9 = var_3.value[var_6][var_8]
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = '[1, 2, 3]'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_3
    var_9 = '{"key": true}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_3
    var_13 = '{"key": null}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_3
    var_17 = '{"key": 123.456}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = len(var_17)
    var_20 = var_19 - var_3
    var_21 = 'invalid_json'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = ''
    var_24 = module_0.tokenize_json(var_23)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = '{"key": {"nested": "value"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = '[1, 2, "three"]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = 0
    var_17 = var_13.value[var_16]
    var_18 = 1
    var_19 = var_13.value[var_18]
    var_20 = 2
    var_21 = var_13.value[var_20]
    var_22 = b'{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"number": 123}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"bool": true}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"null": null}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '[1, 2, 3]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'key'
    var_9 = var_5.value[var_8]
    var_10 = b'{"key": "value"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_11.value[var_8]
    var_15 = '[1, 2, 3]'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = var_16.value
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'true'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 'false'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '123'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '123.456'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '1.23e4'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '"hello"'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '[1, 2, 3]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '"string"'
    var_9 = module_0.tokenize_json(var_8)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '[1, 2, "three"]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_10.value[var_8]
    var_14 = 1
    var_15 = var_10.value[var_14]
    var_16 = 2
    var_17 = var_10.value[var_16]
    var_18 = b'{"bytes": "input"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '42'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = 'true'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = 'null'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '[1, "two", false]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 3



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_1.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = {var_4: var_8}
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_5, var_6, var_7, var_10)
    var_15 = {var_4: var_14}



# Parsed testcases at query #29
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
    var_7 = '{"key": "value"'
    var_8 = module_0.tokenize_json(var_7)



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '[1, 2, "three"]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_10.value[var_8]
    var_14 = 1
    var_15 = var_10.value[var_14]
    var_16 = 2
    var_17 = var_10.value[var_16]
    var_18 = b'{"bytes": true}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #31
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)



# Parsed testcases at query #32
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = b'{"key": "value"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"key": null}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"true": true, "false": false}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'true'
    var_15 = 2
    var_16 = 6
    var_17 = module_1.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = var_13.value[var_17]
    var_19 = 'false'
    var_20 = 16
    var_21 = 20
    var_22 = module_1.ScalarToken(var_19, var_20, var_21, var_12)
    var_23 = var_13.value[var_22]
    var_24 = '{"int": 123, "float": 123.456}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'int'
    var_27 = 4
    var_28 = module_1.ScalarToken(var_26, var_15, var_27, var_24)
    var_29 = var_25.value[var_28]
    var_30 = 'float'
    var_31 = 15
    var_32 = 19
    var_33 = module_1.ScalarToken(var_30, var_31, var_32, var_24)
    var_34 = var_25.value[var_33]
    var_35 = '[1, "two", false]'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 3
    var_39 = '{"nested": {"key": "value"}, "array": [1, 2, 3]}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = 'nested'
    var_42 = 7
    var_43 = module_1.ScalarToken(var_41, var_15, var_42, var_39)
    var_44 = var_40.value[var_43]
    var_45 = 'array'
    var_46 = 30
    var_47 = 34
    var_48 = module_1.ScalarToken(var_45, var_46, var_47, var_39)
    var_49 = var_40.value[var_48]
    var_50 = var_49.value
    var_51 = len(var_50)
    assert var_51 == 3



# Parsed testcases at query #33
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'Expected DictToken'
    var_5 = '{"key": "value"'
    var_6 = module_0.tokenize_json(var_5)



# Parsed testcases at query #34
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)



# Parsed testcases at query #35
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 0
    var_5 = var_1.value[var_4][var_4]
    var_6 = 1
    var_7 = var_1.value[var_4][var_6]
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value[var_4]
    var_13 = var_9.value[var_6]
    var_14 = 2
    var_15 = var_9.value[var_14]
    var_16 = '{"key": {"nested_key": "nested_value"}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = var_17.value[var_4][var_4]
    var_21 = var_17.value[var_4][var_6]
    var_22 = var_17.value[var_4][var_6]
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_17.value[var_4][var_6]
    var_26 = var_25.value[var_4][var_4]
    var_27 = var_17.value[var_4][var_6]
    var_28 = var_27.value[var_4][var_6]
    var_29 = '{}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = '[]'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = '{key: "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #36
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 8
    var_7 = 14
    var_8 = module_1.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = {var_4: var_8}
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_5, var_6, var_7, var_10)
    var_15 = {var_4: var_14}



# Parsed testcases at query #37
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)



# Parsed testcases at query #38
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #39
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"key": {"nested": "value"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'key'
    var_11 = var_9.value[var_10]
    var_12 = '{"key": [1, 2, 3]}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value[var_10]
    var_15 = '{"key": true, "key2": false, "key3": null}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '{"key": 42, "key2": 3.14, "key3": -1}'
    var_18 = module_0.tokenize_json(var_17)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '{"key": [1, 2, 3]}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '{"true": true, "false": false}'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 1
    var_16 = '{"null": null}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"int": 123, "float": 123.456}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"escaped": "\\"\\\\\\/\\b\\f\\n\\r\\t"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"unicode": "\\u00E9"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = b'{"bytes": "value"}'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 9
    var_5 = 15
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = 0
    var_9 = 16
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = 7
    var_19 = module_1.ScalarToken(var_17, var_18, var_18, var_10)
    var_20 = [var_13, var_16, var_19]
    var_21 = 8
    var_22 = module_1.ListToken(var_20, var_8, var_21, var_10)
    var_23 = 'true'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = True
    var_26 = module_1.ScalarToken(var_25, var_8, var_17, var_23)
    var_27 = 'null'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = None
    var_30 = module_1.ScalarToken(var_29, var_8, var_17, var_27)
    var_31 = '{"numbers": [1.23, 4.56]}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = 'numbers'
    var_34 = 1.23
    var_35 = 12
    var_36 = module_1.ScalarToken(var_34, var_35, var_5, var_31)
    var_37 = 4.56
    var_38 = 17
    var_39 = 20
    var_40 = module_1.ScalarToken(var_37, var_38, var_39, var_31)
    var_41 = [var_36, var_40]
    var_42 = 11
    var_43 = 21
    var_44 = module_1.ListToken(var_41, var_42, var_43, var_31)
    var_45 = {var_33: var_44}
    var_46 = 22



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '{"key": {"nested": 123}}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '[1, "two", false]'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = b'{"bytes": "input"}'
    var_16 = module_0.tokenize_json(var_15)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 0
    var_5 = var_1.value[var_4][var_4]
    var_6 = 1
    var_7 = var_1.value[var_4][var_6]
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)



# Parsed testcases at query #5
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
    var_6 = '{"key": 42}'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_7.value[var_4]
    var_11 = '{"key": true}'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = var_12.value[var_4]
    var_16 = '{"key": false}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = var_17.value[var_4]
    var_21 = '{"key": null}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = var_22.value[var_4]
    var_26 = '{"key": [1, 2, 3]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = var_27.value[var_4]
    var_31 = var_27.value[var_4]
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 3
    var_34 = var_27.value[var_4]
    var_35 = var_34.value
    var_36 = '{"key": {"nested": "value"}}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = var_37.value[var_4]
    var_41 = var_37.value[var_4]
    var_42 = var_41.value
    var_43 = len(var_42)
    assert var_43 == 1
    var_44 = 'nested'
    var_45 = var_37.value[var_4]
    var_46 = var_45.value[var_44]



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = b'{"key": "value"}'
    var_7 = module_0.tokenize_json(var_6)



# Parsed testcases at query #7
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
    var_6 = '{"key": "value"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"key": {"nested_key": "nested_value"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'key'
    var_13 = var_9.value[var_12]
    var_14 = '{"key": [1, 2, 3]}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = var_15.value[var_12]
    var_19 = var_15.value[var_12]
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = '{"key": true}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = '{"key": false}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '{"key": null}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = '{"key": 123}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = '{"key": 123.456}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = var_39.value
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = '{"key": 1.23e4}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '{"key": "こんにちは"}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = var_47.value
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = '{"key": "\\"escaped\\""}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = var_51.value
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = '{"key1": "value", "key2": 123, "key3": true, "key4": null}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = var_55.value
    var_57 = len(var_56)
    assert var_57 == 4
    var_58 = '{"key": [[1, 2], [3, 4]]}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = var_59.value
    var_61 = len(var_60)
    assert var_61 == 1
    var_62 = var_59.value[var_12]
    var_63 = var_59.value[var_12]
    var_64 = var_63.value
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = 0
    var_67 = var_59.value[var_12]
    var_68 = var_67.value[var_66]
    var_69 = var_59.value[var_12]
    var_70 = var_69.value[var_66]
    var_71 = var_70.value
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = '{"key": {"nested_key": [1, 2, 3]}}'
    var_74 = module_0.tokenize_json(var_73)
    var_75 = var_74.value
    var_76 = len(var_75)
    assert var_76 == 1
    var_77 = var_74.value[var_12]
    var_78 = 'nested_key'
    var_79 = var_74.value[var_12]
    var_80 = var_79.value[var_78]
    var_81 = var_74.value[var_12]
    var_82 = var_81.value[var_78]
    var_83 = var_82.value
    var_84 = len(var_83)
    assert var_84 == 3
    var_85 = '{"key1": {"key2": [1, {"key3": "value"}]}}'
    var_86 = module_0.tokenize_json(var_85)
    var_87 = var_86.value
    var_88 = len(var_87)
    assert var_88 == 1
    var_89 = 'key1'
    var_90 = var_86.value[var_89]
    var_91 = 'key2'
    var_92 = var_86.value[var_89]
    var_93 = var_92.value[var_91]
    var_94 = var_86.value[var_89]
    var_95 = var_94.value[var_91]
    var_96 = var_95.value
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = 1
    var_99 = var_86.value[var_89]
    var_100 = var_99.value[var_91]
    var_101 = var_100.value[var_98]
    var_102 = '{"key1": {"key2": {"key3": {"key4": "value"}}}}'
    var_103 = module_0.tokenize_json(var_102)
    var_104 = var_103.value
    var_105 = len(var_104)
    assert var_105 == 1
    var_106 = var_103.value[var_89]
    var_107 = var_103.value[var_89]
    var_108 = var_107.value[var_91]
    var_109 = 'key3'
    var_110 = var_103.value[var_89]
    var_111 = var_110.value[var_91]
    var_112 = var_111.value[var_109]
    var_113 = '{"key": 123456789012345678901234567890}'
    var_114 = module_0.tokenize_json(var_113)
    var_115 = var_114.value
    var_116 = len(var_115)
    assert var_116 == 1
    var_117 = '{"key": -123}'
    var_118 = module_0.tokenize_json(var_117)
    var_119 = var_118.value
    var_120 = len(var_119)
    assert var_120 == 1
    var_121 = '{"key": -123.456}'
    var_122 = module_0.tokenize_json(var_121)
    var_123 = var_122.value
    var_124 = len(var_123)
    assert var_124 == 1
    var_125 = '{"key": -1.23e4}'
    var_126 = module_0.tokenize_json(var_125)
    var_127 = var_126.value
    var_128 = len(var_127)
    assert var_128 == 1
    var_129 = '{"key": 0}'
    var_130 = module_0.tokenize_json(var_129)
    var_131 = var_130.value
    var_132 = len(var_131)
    assert var_132 == 1
    var_133 = '{"key": 0.0}'
    var_134 = module_0.tokenize_json(var_133)
    var_135 = var_134.value
    var_136 = len(var_135)
    assert var_136 == 1
    var_137 = '{"key": -0}'
    var_138 = module_0.tokenize_json(var_137)
    var_139 = var_138.value
    var_140 = len(var_139)
    assert var_140 == 1
    var_141 = '{"key": -0.0}'
    var_142 = module_0.tokenize_json(var_141)
    var_143 = var_142.value
    var_144 = len(var_143)
    assert var_144 == 1
    var_145 = '{"key": -123456789012345678901234567890}'
    var_146 = module_0.tokenize_json(var_145)
    var_147 = var_146.value
    var_148 = len(var_147)
    assert var_148 == 1



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 0
    var_5 = '[1, 2, 3]'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = var_6.value
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'false'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '123.45'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '"string"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{ invalid }'
    var_25 = module_0.tokenize_json(var_24)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 14
    var_6 = '{"key": "value"}'
    var_7 = module_1.ScalarToken(var_3, var_4, var_5, var_6)
    var_8 = {var_2: var_7}
    var_9 = 0
    var_10 = 16
    var_11 = module_0.tokenize_json(var_6)
    var_12 = module_1.ScalarToken(var_3, var_4, var_5, var_6)
    var_13 = {var_2: var_12}
    var_14 = '{"key": 123}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 123
    var_17 = 10
    var_18 = module_1.ScalarToken(var_16, var_4, var_17, var_14)
    var_19 = {var_2: var_18}
    var_20 = 12
    var_21 = '{"key": true}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = True
    var_24 = 11
    var_25 = module_1.ScalarToken(var_23, var_4, var_24, var_21)
    var_26 = {var_2: var_25}
    var_27 = 13
    var_28 = '{"key": false}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = False
    var_31 = module_1.ScalarToken(var_30, var_4, var_20, var_28)
    var_32 = {var_2: var_31}
    var_33 = '{"key": null}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = None
    var_36 = module_1.ScalarToken(var_35, var_4, var_24, var_33)
    var_37 = {var_2: var_36}
    var_38 = '{"key": [1, 2, 3]}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = 9
    var_41 = module_1.ScalarToken(var_23, var_40, var_17, var_38)
    var_42 = 2
    var_43 = module_1.ScalarToken(var_42, var_20, var_27, var_38)
    var_44 = 3
    var_45 = 15
    var_46 = module_1.ScalarToken(var_44, var_45, var_10, var_38)
    var_47 = [var_41, var_43, var_46]
    var_48 = 18
    var_49 = module_1.ListToken(var_47, var_4, var_48, var_38)
    var_50 = {var_2: var_49}
    var_51 = 20
    var_52 = '{"key": {"nested": "value"}}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'nested'
    var_55 = 17
    var_56 = 24
    var_57 = module_1.ScalarToken(var_3, var_55, var_56, var_52)
    var_58 = {var_54: var_57}
    var_59 = 26
    var_60 = 28
    var_61 = '{"key": {"nested": 123}}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = module_1.ScalarToken(var_16, var_55, var_51, var_61)
    var_64 = {var_54: var_63}
    var_65 = 22
    var_66 = '{"key": {"nested": true}}'
    var_67 = module_0.tokenize_json(var_66)
    var_68 = 21
    var_69 = module_1.ScalarToken(var_23, var_55, var_68, var_66)
    var_70 = {var_54: var_69}
    var_71 = 23
    var_72 = 25
    var_73 = '{"key": {"nested": false}}'
    var_74 = module_0.tokenize_json(var_73)
    var_75 = False
    var_76 = module_1.ScalarToken(var_75, var_55, var_65, var_73)
    var_77 = {var_54: var_76}
    var_78 = '{"key": {"nested": null}}'
    var_79 = module_0.tokenize_json(var_78)
    var_80 = module_1.ScalarToken(var_35, var_55, var_68, var_78)
    var_81 = {var_54: var_80}
    var_82 = '{"key": {"nested": [1, 2, 3]}}'
    var_83 = module_0.tokenize_json(var_82)
    var_84 = 19
    var_85 = module_1.ScalarToken(var_23, var_84, var_51, var_82)
    var_86 = module_1.ScalarToken(var_42, var_65, var_71, var_82)
    var_87 = module_1.ScalarToken(var_44, var_72, var_59, var_82)
    var_88 = [var_85, var_86, var_87]
    var_89 = module_1.ListToken(var_88, var_55, var_60, var_82)
    var_90 = {var_54: var_89}
    var_91 = 30
    var_92 = 32
    var_93 = '{"key": {"nested": {"deep": "value"}}}'
    var_94 = module_0.tokenize_json(var_93)
    var_95 = 'deep'
    var_96 = module_1.ScalarToken(var_3, var_71, var_91, var_93)
    var_97 = {var_95: var_96}
    var_98 = 34
    var_99 = 36
    var_100 = '{"key": {"nested": {"deep": 123}}}'
    var_101 = module_0.tokenize_json(var_100)
    var_102 = module_1.ScalarToken(var_16, var_71, var_59, var_100)
    var_103 = {var_95: var_102}
    var_104 = '{"key": {"nested": {"deep": true}}}'
    var_105 = module_0.tokenize_json(var_104)
    var_106 = 27
    var_107 = module_1.ScalarToken(var_23, var_71, var_106, var_104)
    var_108 = {var_95: var_107}
    var_109 = 29
    var_110 = 31
    var_111 = 33
    var_112 = '{"key": {"nested": {"deep": false}}}'
    var_113 = module_0.tokenize_json(var_112)
    var_114 = False
    var_115 = module_1.ScalarToken(var_114, var_71, var_60, var_112)
    var_116 = {var_95: var_115}
    var_117 = '{"key": {"nested": {"deep": null}}}'
    var_118 = module_0.tokenize_json(var_117)
    var_119 = module_1.ScalarToken(var_35, var_71, var_106, var_117)
    var_120 = {var_95: var_119}
    var_121 = '{"key": {"nested": {"deep": [1, 2, 3]}}}'
    var_122 = module_0.tokenize_json(var_121)
    var_123 = module_1.ScalarToken(var_23, var_72, var_59, var_121)
    var_124 = module_1.ScalarToken(var_42, var_60, var_109, var_121)
    var_125 = module_1.ScalarToken(var_44, var_110, var_92, var_121)
    var_126 = [var_123, var_124, var_125]
    var_127 = module_1.ListToken(var_126, var_71, var_98, var_121)
    var_128 = {var_95: var_127}
    var_129 = 38
    var_130 = 40
    var_131 = '{"key": {"nested": {"deep": {"deeper": "value"}}}}'
    var_132 = module_0.tokenize_json(var_131)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_1.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = {var_4: var_8}
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_5, var_6, var_7, var_10)
    var_15 = {var_4: var_14}



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'invalid'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Expected DictToken'
    var_7 = '{"key": {"nested": "value"}}'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = '[1, 2, "three"]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 'Expected ListToken'
    var_12 = '"scalar"'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Expected ScalarToken'



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '123'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = 'true'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = 'null'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '[1, "two", false]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = 'All test_tokenize_json tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #13
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
    var_6 = '[1, 2, 3]'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    var_9 = len(var_8)
    assert var_9 == 3
    var_10 = var_7.value
    var_11 = 'true'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = 'null'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = '42'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '3.14'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = '{"nested": {"key": "value"}}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = 'nested'
    var_22 = var_20.value[var_21]
    var_23 = var_20.value[var_21]
    var_24 = var_23.value[var_4]



# Parsed testcases at query #14
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
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 1
    var_11 = module_1.ScalarToken(var_10, var_10, var_10, var_8)
    var_12 = 2
    var_13 = 4
    var_14 = module_1.ScalarToken(var_12, var_13, var_13, var_8)
    var_15 = 3
    var_16 = module_1.ScalarToken(var_15, var_4, var_4, var_8)
    var_17 = [var_11, var_14, var_16]
    var_18 = 'null'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'true'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'false'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '123'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '12.3'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value", "nested": {"key2": "value2"}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'nested'
    var_31 = module_1.ScalarToken(var_3, var_4, var_5, var_28)
    var_32 = 'key2'
    var_33 = 'value2'
    var_34 = 29
    var_35 = 36
    var_36 = module_1.ScalarToken(var_33, var_34, var_35, var_28)
    var_37 = {var_32: var_36}
    var_38 = 22
    var_39 = 38
    var_40 = '[{"key": "value"}]'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 8
    var_43 = 14
    var_44 = module_1.ScalarToken(var_3, var_42, var_43, var_40)
    var_45 = {var_2: var_44}
    var_46 = 16
    var_47 = '{"key": "value", "key2": [1, 2, 3]}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = module_1.ScalarToken(var_3, var_4, var_5, var_47)
    var_50 = module_1.ScalarToken(var_10, var_38, var_38, var_47)
    var_51 = 25
    var_52 = module_1.ScalarToken(var_12, var_51, var_51, var_47)
    var_53 = 28
    var_54 = module_1.ScalarToken(var_15, var_53, var_53, var_47)
    var_55 = [var_50, var_52, var_54]
    var_56 = 20
    var_57 = 30
    var_58 = module_1.ListToken(var_55, var_56, var_57, var_47)
    var_59 = {var_2: var_49, var_32: var_58}
    var_60 = '{"key": "value", "key2": {"key3": "value3"}}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = module_1.ScalarToken(var_3, var_4, var_5, var_60)
    var_63 = 'key3'
    var_64 = 'value3'
    var_65 = 35
    var_66 = module_1.ScalarToken(var_64, var_53, var_65, var_60)
    var_67 = {var_63: var_66}
    var_68 = 37
    var_69 = '{"key": "value", "key2": {"key3": "value3", "key4": "value4"}}'
    var_70 = module_0.tokenize_json(var_69)
    var_71 = module_1.ScalarToken(var_3, var_4, var_5, var_69)
    var_72 = 'key4'
    var_73 = module_1.ScalarToken(var_64, var_53, var_65, var_69)
    var_74 = 'value4'
    var_75 = 46
    var_76 = 53
    var_77 = module_1.ScalarToken(var_74, var_75, var_76, var_69)
    var_78 = {var_63: var_73, var_72: var_77}
    var_79 = 55
    var_80 = '{"key": "value", "key2": {"key3": "value3", "key4": "value4", "key5": "value5"}}'
    var_81 = module_0.tokenize_json(var_80)
    var_82 = module_1.ScalarToken(var_3, var_4, var_5, var_80)
    var_83 = 'key5'
    var_84 = module_1.ScalarToken(var_64, var_53, var_65, var_80)
    var_85 = module_1.ScalarToken(var_74, var_75, var_76, var_80)
    var_86 = 'value5'
    var_87 = 64
    var_88 = 71
    var_89 = module_1.ScalarToken(var_86, var_87, var_88, var_80)
    var_90 = {var_63: var_84, var_72: var_85, var_83: var_89}
    var_91 = 73
    var_92 = '{"key": "value", "key2": {"key3": "value3", "key4": "value4", "key5": "value5", "key6": "value6"}}'
    var_93 = module_0.tokenize_json(var_92)
    var_94 = module_1.ScalarToken(var_3, var_4, var_5, var_92)
    var_95 = 'key6'
    var_96 = module_1.ScalarToken(var_64, var_53, var_65, var_92)
    var_97 = module_1.ScalarToken(var_74, var_75, var_76, var_92)
    var_98 = module_1.ScalarToken(var_86, var_87, var_88, var_92)
    var_99 = 'value6'
    var_100 = 82
    var_101 = 89
    var_102 = module_1.ScalarToken(var_99, var_100, var_101, var_92)
    var_103 = {var_63: var_96, var_72: var_97, var_83: var_98, var_95: var_102}
    var_104 = 91
    var_105 = '{"key": "value", "key2": {"key3": "value3", "key4": "value4", "key5": "value5", "key6": "value6", "key7": "value7"}}'
    var_106 = module_0.tokenize_json(var_105)
    var_107 = module_1.ScalarToken(var_3, var_4, var_5, var_105)
    var_108 = 'key7'
    var_109 = module_1.ScalarToken(var_64, var_53, var_65, var_105)
    var_110 = module_1.ScalarToken(var_74, var_75, var_76, var_105)
    var_111 = module_1.ScalarToken(var_86, var_87, var_88, var_105)
    var_112 = module_1.ScalarToken(var_99, var_100, var_101, var_105)
    var_113 = 'value7'
    var_114 = 100
    var_115 = 107
    var_116 = module_1.ScalarToken(var_113, var_114, var_115, var_105)
    var_117 = {var_63: var_109, var_72: var_110, var_83: var_111, var_95: var_112, var_108: var_116}
    var_118 = 109
    var_119 = '{"key": "value", "key2": {"key3": "value3", "key4": "value4", "key5": "value5", "key6": "value6", "key7": "value7", "key8": "value8"}}'
    var_120 = module_0.tokenize_json(var_119)
    var_121 = module_1.ScalarToken(var_3, var_4, var_5, var_119)
    var_122 = 'key8'
    var_123 = module_1.ScalarToken(var_64, var_53, var_65, var_119)
    var_124 = module_1.ScalarToken(var_74, var_75, var_76, var_119)
    var_125 = module_1.ScalarToken(var_86, var_87, var_88, var_119)
    var_126 = module_1.ScalarToken(var_99, var_100, var_101, var_119)
    var_127 = module_1.ScalarToken(var_113, var_114, var_115, var_119)
    var_128 = 'value8'
    var_129 = 118
    var_130 = 125
    var_131 = module_1.ScalarToken(var_128, var_129, var_130, var_119)
    var_132 = {var_63: var_123, var_72: var_124, var_83: var_125, var_95: var_126, var_108: var_127, var_122: var_131}
    var_133 = 127



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 13
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_15 = {var_6: var_14}
    var_16 = '{"key": {"nested": "value"}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value[var_6]
    var_19 = 'nested'
    var_20 = 17
    var_21 = 23
    var_22 = module_1.ScalarToken(var_7, var_20, var_21, var_16)
    var_23 = {var_19: var_22}
    var_24 = 'All tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = '{"key": {"nested_key": "nested_value"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = '{"key": [1, 2, 3]}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = '{"key": 123}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = '{"key": true}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = '{"key": null}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 1



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'key'
    var_9 = var_5.value[var_8]
    var_10 = b'{"key": "value"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_11.value[var_8]
    var_15 = '{"number": 123}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 'number'
    var_18 = var_16.value[var_17]
    var_19 = '{"bool": true}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = 'bool'
    var_22 = var_20.value[var_21]
    var_23 = '{"null": null}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = 'null'
    var_26 = var_24.value[var_25]
    var_27 = '{"array": [1, 2, 3]}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = 'array'
    var_30 = var_28.value[var_29]
    var_31 = var_28.value[var_29]
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 3
    var_34 = var_28.value[var_29]
    var_35 = var_34.value
    var_36 = '{"nested": {"key": "value"}}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'nested'
    var_39 = var_37.value[var_38]
    var_40 = var_37.value[var_38]
    var_41 = var_40.value[var_8]
    var_42 = 'All tests passed!'
    var_43 = print(var_42)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 8
    var_5 = 14
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = 0
    var_9 = 16
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = 7
    var_19 = module_1.ScalarToken(var_17, var_18, var_18, var_10)
    var_20 = [var_13, var_16, var_19]
    var_21 = module_1.ListToken(var_20, var_8, var_4, var_10)
    var_22 = 'true'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = True
    var_25 = module_1.ScalarToken(var_24, var_8, var_17, var_22)
    var_26 = 'false'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = False
    var_29 = module_1.ScalarToken(var_28, var_28, var_15, var_26)
    var_30 = 'null'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = None
    var_33 = module_1.ScalarToken(var_32, var_28, var_17, var_30)
    var_34 = '42'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = 42
    var_37 = module_1.ScalarToken(var_36, var_28, var_24, var_34)
    var_38 = '3.14'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = 3.14
    var_41 = module_1.ScalarToken(var_40, var_28, var_17, var_38)
    var_42 = '"string"'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = 'string'
    var_45 = module_1.ScalarToken(var_44, var_28, var_18, var_42)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 14
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = b'{"key": "value"}'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '{"key": [1, 2, 3]}'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = 'All tokenize_json tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '123'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"key": null}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"key": true}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": false}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"key": 123}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)



# Parsed testcases at query #23
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{invalid}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = '[1, 2, "three"]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_10.value[var_8]
    var_14 = 1
    var_15 = var_10.value[var_14]
    var_16 = 2
    var_17 = var_10.value[var_16]
    var_18 = b'{"bytes": true}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'key'
    var_9 = var_5.value[var_8]
    var_10 = b'{"key": "value"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = var_11.value[var_8]
    var_15 = '{"key": {"nested": 123}}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = var_16.value[var_8]
    var_20 = 'nested'
    var_21 = var_16.value[var_8]
    var_22 = var_21.value[var_20]
    var_23 = '[1, "two", false]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 3
    var_27 = 0
    var_28 = var_24.value[var_27]
    var_29 = 1
    var_30 = var_24.value[var_29]
    var_31 = 2
    var_32 = var_24.value[var_31]
    var_33 = 'All tests passed.'
    var_34 = print(var_33)



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '{"key": "value"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_1.ScalarToken(var_5, var_6, var_7, var_2)
    var_9 = {var_4: var_8}
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)



