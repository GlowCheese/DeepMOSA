####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'true'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = var_3.value
    assert var_4 is True
    var_5 = 'false'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is False
    var_8 = 'null'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is None
    var_11 = '123'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 == 123
    var_14 = '-456'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == -456
    var_17 = '12.34'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = '1e10'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '{"key": "value"}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = ''
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"key": "value'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"key": "value",}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = b'"bytes"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    assert var_33 == 'bytes'
    var_34 = '{"a": 1, "b": true}'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = '123'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 == 123
    var_8 = '-123'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 == -123
    var_11 = '123.456'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = '1e10'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = 'true'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 is True



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"string"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'string'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = '123'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    assert var_6 == 123
    var_7 = '-123'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = var_8.value
    assert var_9 == -123
    var_10 = '123.456'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = '1e10'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 is True
    var_19 = 'false'
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    assert var_22 is None
    var_23 = '{"key": "value", "num": 123, "nested": {"a": true}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '[1, "two", {"three": 3}]'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = 'a'
    var_28 = b'{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value[var_27]
    assert var_30 == 1
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"key": "missing_bracket"'
    var_34 = '{"key" "value"}'
    var_35 = '[1, 2, ]'
    var_36 = 'true extra'
    var_37 = '"unclosed string'
    var_38 = [var_33, var_34, var_35, var_36, var_37]
    var_39 = '{123: "value"}'
    var_40 = module_0.tokenize_json(var_39)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = module_0.tokenize_json(var_4)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = module_0.tokenize_json(var_8)
    var_11 = var_10.value
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_0.tokenize_json(var_12)
    var_15 = var_14.value
    assert var_15 is None
    var_16 = '123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 == 123
    var_19 = '-123'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    assert var_21 == -123
    var_22 = '123.456'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = '1e10'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = '{"key": "value", "num": 123}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, "two", {"three": 3}]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'a'
    var_33 = b'{"a": 1}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value[var_32]
    assert var_35 == 1
    var_36 = '   '
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "unclosed quote}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '[1, 2, ]'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"list": [true, null], "nested": {"a": 1}}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{\n  "space" : \t 123 \r\n}'
    var_45 = module_0.tokenize_json(var_44)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '{"key": "value", "number": 123, "bool": true}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, "two", {"three": 3}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'"bytes"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    assert var_32 == 'bytes'
    var_33 = '   '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": "unclosed}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key" "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": "value",}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '[[[[[1]]]]]'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '[1, "two", true]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = '{"key": "value", "num": 42}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"a": [1, {"b": 2}]}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 0
    var_23 = 'a'
    var_24 = var_21.value[var_23]
    var_25 = var_24.value[var_22]
    var_26 = b'{"test": 1}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '   '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "missing_quote}'
    var_31 = '[1, 2, ]'
    var_32 = '{"key": }'
    var_33 = 'not a json'
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = '  {  "spaced"  :   123  }  '
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -1
    var_18 = '123.456'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = '1e10'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '[1, "two", null]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = '{"key": "value", "num": 42}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"a": [1, {"b": true}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 1
    var_33 = 'a'
    var_34 = var_31.value[var_33]
    var_35 = var_34.value[var_32]
    var_36 = var_35.value
    var_37 = b'{"a": 1}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value[var_33]
    var_40 = var_39.value
    assert var_40 == 1
    var_41 = '   '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"unclosed": "string"'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{unquoted_key: 1}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[1, 2, ]'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{1: "value"}'
    var_50 = module_0.tokenize_json(var_49)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = '123'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 == 123
    var_8 = '123.45'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = '-0.5e2'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    var_14 = 'true'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 is True
    var_17 = 'false'
    var_18 = 'null'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    assert var_20 is None
    var_21 = '[1, "a", true]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 3
    var_25 = '{"key": "value", "num": 1}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"a": [1, {"b": 2}]}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = 'x'
    var_30 = '  \n  {  "x" : 10  }  \t '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value[var_29]
    assert var_32 == 10
    var_33 = 'key'
    var_34 = b'{"key": "val"}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value[var_33]
    assert var_36 == 'val'
    var_37 = '   '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": "unclosed}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '12.34.56'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{unquoted_key: 1}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"a" "b"}'
    var_46 = module_0.tokenize_json(var_45)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"string"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'string'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -124
    var_18 = module_0.tokenize_json(var_15)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '1.23'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '{"key": "value", "number": 42, "bool": true}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, "two", {"three": 3}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '  {  "a"  :  [ 1 , 2 ]  }  '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = b'"bytes"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    assert var_35 == 'bytes'
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"unclosed": "string"'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"key" "no_colon"}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": 123'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"outer": {"inner": [1, 2]}}'
    var_45 = module_0.tokenize_json(var_44)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"key": "value", "num": 10, "nested": {"a": true}}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '[1, "two", {"three": 3}]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = b'{"a": 1}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = ''
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   \n\t  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key": "value"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{key: "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"a": 1,}'
    var_29 = module_0.tokenize_json(var_28)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '[1, "two", true]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 3
    var_30 = '{"key": "value", "num": 1}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"outer": [1, {"inner": "deep"}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = 1
    var_36 = 'outer'
    var_37 = var_33.value[var_36]
    var_38 = var_37.value[var_35]
    var_39 = var_38.value
    var_40 = 'a'
    var_41 = '  {"a"  :  1}  '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value[var_40]
    var_44 = var_43.value
    assert var_44 == 1
    var_45 = b'{"a": 1}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = var_46.value[var_40]
    var_48 = var_47.value
    assert var_48 == 1
    var_49 = ''
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": unquoted_value}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key" "value"}'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{"key": "unclosed'
    var_56 = module_0.tokenize_json(var_55)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '{"key": "value", "number": 1, "bool": true, "list": [1, 2, {"inner": "obj"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = '[]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = b'"bytes"'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    assert var_36 == 'bytes'
    var_37 = '   {"a" : 1}   '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = ''
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": "unclosed quote}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"key" "value"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '[1, 2, ]'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = 'invalid'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"key": [1, 2'
    var_51 = module_0.tokenize_json(var_50)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = module_0.tokenize_json(var_4)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = module_0.tokenize_json(var_8)
    var_11 = var_10.value
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_0.tokenize_json(var_12)
    var_15 = var_14.value
    assert var_15 is None
    var_16 = '123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 == 123
    var_19 = '-123'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    assert var_21 == -123
    var_22 = '123.456'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = '1e10'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = '[1, "a", {"b": 2}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = '{"key": "value", "num": 1}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'a'
    var_35 = b'{"a": 1}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value[var_34]
    var_38 = var_37.value
    assert var_38 == 1
    var_39 = '{}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = module_0.tokenize_json(var_39)
    var_42 = var_41.value
    var_43 = len(var_42)
    assert var_43 == 0
    var_44 = '[]'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = module_0.tokenize_json(var_44)
    var_47 = var_46.value
    var_48 = len(var_47)
    assert var_48 == 0
    var_49 = '  \n  "spaced"  \t '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    assert var_51 == 'spaced'
    var_52 = ''
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{unquoted_key: 1}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{"a": 1 "b": 2}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '[1, 2,]'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = '{"a": [1, 2'
    var_61 = module_0.tokenize_json(var_60)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '{"key": "value", "num": 1, "bool": true}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, "two", {"three": 3}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = '[]'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = b'{"a": 1}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"key": "unclosed}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = "Expecting ',' delimiter"
    var_44 = 'Expecting property name'
    var_45 = '[1, 2, ]'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '1.2.3'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{123: "value"}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": 123'
    var_52 = module_0.tokenize_json(var_51)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '42'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '3.14'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '[1, "two", true]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = '{"key": "value", "num": 123}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = '{"a": [1, {"b": 2}]}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = 'a'
    var_26 = var_23.value[var_25]
    var_27 = var_26.value
    var_28 = 1
    var_29 = var_27[var_28]
    var_30 = var_29.value
    var_31 = b'{"byte": true}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"unclosed": "string'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key" "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = 'utf-8'
    var_40 = '!'
    var_41 = module_0.tokenize_json(var_40)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -12
    var_20 = module_0.tokenize_json(var_17)
    var_21 = var_20.value
    assert var_21 == -123
    var_22 = '123.456'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = '1e10'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = '[1, "two", true]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = '{"key": "value", "num": 1}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"a": [1, {"b": 2}]}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = 1
    var_37 = 'a'
    var_38 = var_35.value[var_37]
    var_39 = var_38.value[var_36]
    var_40 = '  \n  "space"  \t '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = var_41.value
    assert var_42 == 'space'
    var_43 = 'byte'
    var_44 = b'{"byte": true}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = var_45.value[var_43]
    var_47 = var_46.value
    assert var_47 is True
    var_48 = ''
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"unclosed": "string"'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '[1, 2, ]'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{key: "no quotes"}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{123: "number as key"}'
    var_57 = module_0.tokenize_json(var_56)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -126
    var_18 = '123.45'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = '1e10'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '{"key": "value", "number": 1, "bool": true}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, "two", {"three": 3}]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"a": [1, {"b": 2}]}'
    var_29 = var_28
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '  {  "key"  :  "val"  }  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = b'{"byte": true}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "missing_quote}'
    var_39 = '{"key" "missing_colon"}'
    var_40 = '[1, 2, ]'
    var_41 = '{"key": val}'
    var_42 = '{'
    var_43 = [var_38, var_39, var_40, var_41, var_42]
    var_44 = '{"key": 123'
    var_45 = module_0.tokenize_json(var_44)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = '[]'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    var_6 = '"string with \\"quotes\\""'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 == 'string with "quotes"'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = '{"name": "test"}'
    var_2 = '{"name": 123}'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    var_8 = module_0.tokenize_json(var_5)
    var_9 = var_8.value
    assert var_9 is True
    var_10 = 'false'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = module_0.tokenize_json(var_10)
    var_14 = var_13.value
    assert var_14 is False
    var_15 = 'null'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = None
    var_19 = module_0.tokenize_json(var_15)
    var_20 = var_19.value
    assert var_20 is None
    var_21 = '123'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    assert var_23 == 123
    var_24 = '-123'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    assert var_26 == -123
    var_27 = '0.456'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = '1e10'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = '[1, "two", true]'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": "value", "num": 42}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"a": [1, {"b": 2}], "c": null}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = b'{"bytes": true}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = ''
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '   '
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"key": "unclosed quote}'
    var_47 = '[1, 2, '
    var_48 = '{"key" "missing colon"}'
    var_49 = '{"key": ,}'
    var_50 = 'not json at all'
    var_51 = [var_46, var_47, var_48, var_49, var_50]
    var_52 = '"test"'
    var_53 = module_0.tokenize_json(var_52)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is True
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 is None
    var_15 = '[1, "two", true]'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 1
    var_18 = module_1.ScalarToken(var_17, var_17, var_17, var_15)
    var_19 = 'two'
    var_20 = 5
    var_21 = 8
    var_22 = module_1.ScalarToken(var_19, var_20, var_21, var_15)
    var_23 = True
    var_24 = 11
    var_25 = 14
    var_26 = module_1.ScalarToken(var_23, var_24, var_25, var_15)
    var_27 = [var_18, var_22, var_26]
    var_28 = '{"key": "value", "num": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"a": [1, {"b": true}]}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'a'
    var_33 = var_31.value[var_32]
    var_34 = var_33.value[var_23]
    var_35 = '  {  "space"  :  123  }  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = b'{"byte": true}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"key": "unclosed}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{key: 123}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '[1, 2,]'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '123.45.67'
    var_48 = module_0.tokenize_json(var_47)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = module_0.tokenize_json(var_4)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = module_0.tokenize_json(var_8)
    var_11 = var_10.value
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_0.tokenize_json(var_12)
    var_15 = var_14.value
    assert var_15 is None
    var_16 = '123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 == 123
    var_19 = '-123'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    assert var_21 == -123
    var_22 = '123.456'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = '1e10'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = '[1, "two", false]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = '{"key": "value", "num": 10}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"a": [1, {"b": 2}]}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = 1
    var_37 = 'a'
    var_38 = var_35.value[var_37]
    var_39 = var_38.value[var_36]
    var_40 = var_39.value
    var_41 = 'byte'
    var_42 = b'{"byte": true}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value[var_41]
    var_45 = var_44.value
    assert var_45 is True
    var_46 = '   '
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"unclosed": "string"'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{key: "value"}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"a" "b"}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '[1, 2,]'
    var_55 = module_0.tokenize_json(var_54)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '[1, "two", null]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 3
    var_30 = '{"key": "value", "num": 1}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"a": [1, {"b": 2}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = 'a'
    var_36 = var_33.value[var_35]
    var_37 = var_36.value
    var_38 = 'bytes'
    var_39 = b'{"bytes": true}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value[var_38]
    var_42 = var_41.value
    assert var_42 is True
    var_43 = '  \n  "spaced"  \t '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    assert var_45 == 'spaced'
    var_46 = ''
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"unclosed": "string"'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{invalid}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"key" "value"}'
    var_53 = module_0.tokenize_json(var_52)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -1
    var_18 = '123.456'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = '1e10'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '{"key": "value", "number": 123, "bool": true}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, "two", {"three": 3}]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"outer": {"inner": [1, 2]}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'"bytes"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    assert var_32 == 'bytes'
    var_33 = '   '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key" "value"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key": "unclosed}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"a": 1,}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{123: "value"}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"a": 1.2.3}'
    var_44 = module_0.tokenize_json(var_43)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": "value", "num": 1}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_17.value
    var_21 = 0
    var_22 = [pair[var_21].value for pair in var_20]
    var_23 = '{"a": [1, {"b": 2}]}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = 1
    var_26 = var_24.value[var_21][var_25]
    var_27 = var_26.value[var_25]
    var_28 = b'{"byte": true}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '   \n\t  '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "unclosed quote}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key" "value"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '[1, 2'
    var_39 = module_0.tokenize_json(var_38)



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123.45'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    assert var_6 is True
    var_7 = 'false'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = var_8.value
    assert var_9 is False
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    assert var_12 is None
    var_13 = '{"key": "value", "num": 1}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = 'key'
    var_16 = 'num'
    var_17 = 'value'
    var_18 = 8
    var_19 = 13
    var_20 = module_1.ScalarToken(var_17, var_18, var_19, var_13)
    var_21 = 1
    var_22 = 17
    var_23 = module_1.ScalarToken(var_21, var_22, var_22, var_13)
    var_24 = {var_15: var_20, var_16: var_23}
    var_25 = '[1, "two"]'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = b'{"a": 1}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"key": "unclosed}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key" "value"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"a": 1,}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"a": [1, {"b": 2}]}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = 'a'
    var_42 = var_40.value[var_41]
    var_43 = var_42.value[var_21]



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123.45e2'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = '{"key": "value", "num": 1, "bool": true}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '[1, "two", {"three": 3}]'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"a": [{"b": 2}]}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = b'{"a": 1}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = '   '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = "{'single': 'quotes'}"
    var_30 = '{"missing_bracket": 1'
    var_31 = '[1, 2, ]'
    var_32 = '{"key" "value"}'
    var_33 = '{"key": value}'
    var_34 = [var_29, var_30, var_31, var_32, var_33]
    var_35 = '{"key": 123'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '"string with \\"quotes\\""'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    assert var_39 == 'string with "quotes"'



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = '123'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    assert var_6 == 123
    var_7 = '-123'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = var_8.value
    assert var_9 == -123
    var_10 = '123.456'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = '1e10'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 is True
    var_19 = 'false'
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    assert var_22 is None
    var_23 = '{"key": "value", "num": 123}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '[1, "two", {"three": 3}]'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = '{"a": [1, {"b": 2}]}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   "space"   '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    assert var_33 == 'space'
    var_34 = '{\n  "key" : \t 1 \n}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = b'{"bytes": true}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = '   '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": "unclosed quote}'
    var_43 = '{"key": 123'
    var_44 = '[1, 2, ]'
    var_45 = 'not json'
    var_46 = '{"key" : }'
    var_47 = [var_42, var_43, var_44, var_45, var_46]
    var_48 = '{"key": 123'
    var_49 = module_0.tokenize_json(var_48)



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -120
    var_18 = '123.456'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = '1e10'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '{"key": "value", "num": 1, "bool": true}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, "two", {"three": 3}]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 2
    var_29 = var_27.value[var_28]
    var_30 = '{}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = '[]'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = '  "spaced"  '
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    assert var_38 == 'spaced'
    var_39 = '{\n  "a":\t1\n}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = b'"bytes"'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value
    assert var_44 == 'bytes'
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"key": "unclosed'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '123.45.67'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key" "value"}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{key: "value"}'
    var_54 = module_0.tokenize_json(var_53)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = 'true'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    assert var_5 is True
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is False
    var_9 = 'null'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is None
    var_12 = '123'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 == 123
    var_15 = '-123'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 == -1



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = '[1, 2, 3]'
    var_4 = module_0.tokenize_json(var_3)
    var_5 = var_4.value
    var_6 = '"string"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 == 'string'
    var_9 = 'true'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is True
    var_12 = 'false'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 is False
    var_15 = 'null'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 is None
    var_18 = '123'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    assert var_20 == 123
    var_21 = '123.456'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '1e10'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = '{"a": [1, {"b": 2}], "c": 3}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = b'{"key": "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = '  {  "a"  :  1  }  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "missing_quote}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"key": [1, 2, }'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key" "value"}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '[1, 2, 3,]'
    var_45 = module_0.tokenize_json(var_44)

def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = '{"name": "John", "age": "not_an_int"}'



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '[1, "a", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": "value", "num": 1}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'key'
    var_21 = '{}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '[]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = b'{"a": 1}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '  \n  "space"  \t '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"key": "unclosed}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key" "value"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{key: "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '[1, 2,]'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = module_0.tokenize_json(var_0)
    var_4 = var_3.value
    assert var_4 == 'hello'
    var_5 = 'true'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = var_6.value
    assert var_7 is True
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is False
    var_11 = 'null'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is None
    var_14 = '123'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 == 123
    var_17 = '-123'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    assert var_19 == -123
    var_20 = '123.456'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '1e10'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '{"key": "value", "num": 123}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, "two", {"three": 3}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'"bytes"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    assert var_32 == 'bytes'
    var_33 = '   '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": "unclosed quote}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key" "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"a": 1 "b": 2}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '123.45.67'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{123: "value"}'
    var_44 = module_0.tokenize_json(var_43)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = var_7.value
    assert var_8 is True
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 is False
    var_12 = 'null'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    assert var_14 is None
    var_15 = '[1, "two", {"three": 3}]'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = 2
    var_20 = var_16.value[var_19]
    var_21 = '{"key": "value", "num": 42}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '[]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = b'{"a": 1}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '  \n  "space"  \t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '"unclosed'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key" "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '[1, 2,]'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{key: "value"}'
    var_42 = module_0.tokenize_json(var_41)



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '123.45e2'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"key": "value", "num": 1}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'value'
    var_15 = 7
    var_16 = 13
    var_17 = module_1.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = '[1, "a", true]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = '{"a": [1, {"b": 2}]}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 1
    var_25 = 'a'
    var_26 = var_23.value[var_25]
    var_27 = var_26.value[var_24]
    var_28 = b'{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "missing_quote}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '[1, 2'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{1: "value"}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = module_0.tokenize_json(var_0)
    var_3 = var_2.value
    assert var_3 == 'hello'
    var_4 = '123'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '123.45e2'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    assert var_10 is True
    var_11 = 'false'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = var_12.value
    assert var_13 is False
    var_14 = 'null'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    assert var_16 is None
    var_17 = '[1, "two", true]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = '{"key": "value", "num": 10}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '{}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = '[]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = '  \n  "spaced"  \t '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    assert var_31 == 'spaced'
    var_32 = 'bytes'
    var_33 = b'{"bytes": true}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value[var_32]
    assert var_35 is True
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key" "value"}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '"unclosed'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{123: "value"}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '[1, 2, ]'
    var_45 = module_0.tokenize_json(var_44)



