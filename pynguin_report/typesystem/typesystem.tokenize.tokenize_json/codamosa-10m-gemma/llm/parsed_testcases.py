####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": "value", "num": 1}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'{"a": 1}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '   '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key: "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key" "value"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "value",}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #2
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
    var_20 = 'key'
    var_21 = '{}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"a": 1}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '   '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"key": "unclosed quote}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"key" "value"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"a": 1,}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"list": [1, {"nested": true}], "val": null}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = 'list'



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123.45'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, "two", true]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '{"key": "value", "num": 10}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"a": [1, {"b": 2}]}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 1
    var_17 = 'a'
    var_18 = var_15.value[var_17]
    var_19 = var_18.value[var_16]
    var_20 = var_19.value
    var_21 = b'{"a": 1}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = ''
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"key: "value"}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"key" "value"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"key": "value"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '1.2.3'
    var_32 = module_0.tokenize_json(var_31)



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
    var_26 = '{"key": "value", "num": 123, "bool": true, "list": [1, 2, {"nested": "item"}]}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = b'{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = '  {  "a"  :  1  }  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = ''
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key": "value"'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key" "value"}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{key: "value"}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"num": 1.2.3}'
    var_43 = module_0.tokenize_json(var_42)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



# Parsed testcases at query #6
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
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": "value", "num": 1}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"a": [1, {"b": 2}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"bytes": true}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    var_23 = '[]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "unclosed}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key" "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"a": 1,}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'abc'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #7
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
    var_7 = 'null'
    var_8 = module_0.tokenize_json(var_7)
    var_9 = '[1, "a", true]'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = '{"key": "value", "num": 1}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = '{"a": [1, {"b": 2}]}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = 1
    var_18 = 'a'
    var_19 = var_16.value[var_18]
    var_20 = var_19.value[var_17]
    var_21 = var_20.value
    var_22 = '  {  "space"  :  123  }  '
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"key": "missing_quote}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '[1, 2,'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = b'{"byte": true}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '-42'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.23E-4'
    var_35 = module_0.tokenize_json(var_34)



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)



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
    assert var_17 == -121
    var_18 = '123.456'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = '1e10'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '{"key": "value", "number": 123, "list": [1, 2, 3]}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[1, "a", {"b": true}]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = '[]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = '  "space"  '
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    assert var_36 == 'space'
    var_37 = '{\n  "a" : 1 \n}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = b'"bytes"'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = var_41.value
    assert var_42 == 'bytes'
    var_43 = ''
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"key": "unclosed}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"key" "value"}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"a": 1,}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{a: 1}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"a": 1.2.3}'
    var_54 = module_0.tokenize_json(var_53)



# Parsed testcases at query #10
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
    var_9 = '123'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = var_10.value
    assert var_11 == 123
    var_12 = '12.34'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = 'true'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 is True
    var_18 = 'false'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    assert var_20 is False
    var_21 = 'null'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    assert var_23 is None
    var_24 = '{"a": [1, {"b": 2}], "c": 3}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = '  {"  key  "  :  123  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = b'{"a": 1}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = "{'single_quotes': 1}"
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key" "value"}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"a": 1 "b": 2}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"a": 1'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '0'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    assert var_45 == 0
    var_46 = '-5'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = var_47.value
    assert var_48 == -5
    var_49 = '1e10'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = '1.5e-2'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = var_53.value



# Parsed testcases at query #11
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
    var_15 = '{"key": "value", "num": 1}'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = '[1, "two", {"three": 3}]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 3
    var_21 = '{"a": [1, {"b": 2}]}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = b'{"key": 1}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "a"  :  1  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = ''
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '   '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"key": "missing_quote}'
    var_32 = '{"key": 1,}'
    var_33 = '[1, 2, '
    var_34 = 'not json'
    var_35 = [var_31, var_32, var_33, var_34]
    var_36 = '{key: 1}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #12
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
    var_16 = '{"key": "value", "num": 10}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = '{"a": [1, {"b": 2}]}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"a": 1}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '   \n\t  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key: "value"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key" "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "value",}'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #13
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
    var_28 = '[1, "two", true]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = '{"key": "value", "num": 1}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value
    var_35 = dict(var_34)
    var_36 = '{"a": [1, {"b": 2}]}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = 'a'
    var_40 = var_37.value[var_39]
    var_41 = var_40.value
    var_42 = 1
    var_43 = var_37.value[var_39]
    var_44 = var_43.value[var_42]
    var_45 = var_44.value
    var_46 = b'"bytes"'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = var_47.value
    assert var_48 == 'bytes'
    var_49 = '  \n  "spaced"  \t '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    assert var_51 == 'spaced'
    var_52 = '   '
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{"unclosed": "brace"'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{"key" "value"}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '{"a": 1 "b": 2}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = '{123: "value"}'
    var_61 = module_0.tokenize_json(var_60)



# Parsed testcases at query #14
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
    var_28 = '[1, "a", true]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = '{"key": "value", "num": 1}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '[]'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = '{}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = 'a'
    var_41 = b'{"a": 1}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value[var_40]
    var_44 = var_43.value
    assert var_44 == 1
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   \n\t  '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"key": "unclosed quote}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key" "value"}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"a": 1,}'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{"outer": [1, {"inner": true}]}'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = 1
    var_58 = 'outer'
    var_59 = var_56.value[var_58]
    var_60 = var_59.value[var_57]



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": "value", "num": 10}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"a": [1, {"b": 2}]}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = b'{"a": 1}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '  {  "a"  :  1  }  '
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"key" "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '"unclosed'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"a": 1,}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{ "a": 1 ]'
    var_31 = module_0.tokenize_json(var_30)



# Parsed testcases at query #3
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
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": "value", "num": 1}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"a": [1, {"b": 2}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 0
    var_21 = 'a'
    var_22 = var_19.value[var_21]
    var_23 = var_22.value[var_20]
    var_24 = 1
    var_25 = var_19.value[var_21]
    var_26 = var_25.value[var_24]
    var_27 = var_26.value
    var_28 = b'{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = '[]'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = ''
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '   \n  '
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"unclosed": "string'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key" "value"}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"a": 1 "b": 2}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{key: "value"}'
    var_47 = module_0.tokenize_json(var_46)



# Parsed testcases at query #4
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
    var_12 = '[1, "two", false]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": 123, "nested": {"a": true}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'nested'
    var_19 = var_17.value[var_18]
    var_20 = var_19.value
    var_21 = '{}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '[]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'{"a": 1}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '  {  "space"  :  [ 1 ]  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '   '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{"missing_quote: 1}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '123.45.67'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '[1, 2, '
    var_36 = module_0.tokenize_json(var_35)



# Parsed testcases at query #5
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
    var_13 = '123'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    assert var_15 == 123
    var_16 = '-123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    assert var_18 == -1
    var_19 = '123.45'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = '1e10'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = '{"key": "value", "num": 123}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '[1, "two", {"three": 3}]'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'{"a": 1}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = '   '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": "value"'
    var_35 = '{"key" "value"}'
    var_36 = '[1, 2,]'
    var_37 = 'not json'
    var_38 = '{"key": unquoted}'
    var_39 = [var_34, var_35, var_36, var_37, var_38]
    var_40 = '\n    {\n        "list": [1, 2, {"inner": true}],\n        "nested_obj": {\n            "a": null,\n            "b": 0.5\n        },\n        "string": "line\\nbreak"\n    }\n    '
    var_41 = module_0.tokenize_json(var_40)



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
    var_16 = '{"key": "value", "num": 10}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 0
    var_21 = '{"a": [1, {"b": 2}]}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = 1
    var_24 = b'{"a": 1}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = ''
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": "missing_bracket"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key" "value"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{key: "value"}'
    var_33 = module_0.tokenize_json(var_32)



# Parsed testcases at query #7
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
    var_19 = '{"key": "value", "num": 1}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = '[]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = b'{"a": 1}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '  {  "a"  :  1  }  '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"unclosed": "string"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key" "value"}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"a": 1 "b": 2}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{key: "value"}'
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #8
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
    var_8 = module_0.tokenize_json(var_5)
    var_9 = var_8.value
    assert var_9 == 123
    var_10 = '-123.45e2'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = var_11.value
    var_13 = module_0.tokenize_json(var_10)
    var_14 = var_13.value
    var_15 = 'true'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    assert var_17 is True
    var_18 = 'false'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    assert var_20 is False
    var_21 = 'null'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    assert var_23 is None
    var_24 = '[1, "two", {"three": 3}]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = 2
    var_29 = var_25.value[var_28]
    var_30 = '{"key": "value", "num": 1}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'a'
    var_33 = '  {"a" : 1}  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value[var_32]
    var_36 = var_35.value
    assert var_36 == 1
    var_37 = b'{"a": 1}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value[var_32]
    var_40 = var_39.value
    assert var_40 == 1
    var_41 = ''
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"key: "value"}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"key" "value"}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"a": 1,}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '[1, 2'
    var_50 = module_0.tokenize_json(var_49)



# Parsed testcases at query #9
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
    var_12 = '[1, "two", true]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = '{"key": "value", "num": 10}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"a": [1, {"b": 2}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'a'
    var_21 = var_19.value[var_20]
    var_22 = var_21.value
    var_23 = 1
    var_24 = var_22[var_23]
    var_25 = var_24.value
    var_26 = b'{"key": "val"}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"key": "missing_quote}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '[1, 2'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '12.34.56'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"key" "value"}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #10
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
    var_28 = '[1, "two", {"three": 3}]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = 2
    var_33 = var_29.value[var_32]
    var_34 = '{"key": "value", "num": 1}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = b'{"a": 1}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = ''
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"key": "unclosed}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"key" "value"}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{key: "value"}'
    var_49 = module_0.tokenize_json(var_48)



# Parsed testcases at query #11
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
    var_15 = '[1, "two", true]'
    var_16 = module_0.tokenize_json(var_15)
    var_17 = var_16.value
    var_18 = len(var_17)
    assert var_18 == 3
    var_19 = '{"key": "value", "num": 1}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '{"a": [1, {"b": 2}]}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = 1
    var_24 = 'a'
    var_25 = var_22.value[var_24]
    var_26 = var_25.value[var_23]
    var_27 = var_26.value
    var_28 = b'{"a": 1}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key": "unclosed quote}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key" "value"}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"a": 1,}'
    var_37 = module_0.tokenize_json(var_36)



# Parsed testcases at query #12
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
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"key": "value", "num": 1}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'value'
    var_11 = 8
    var_12 = 13
    var_13 = module_1.ScalarToken(var_10, var_11, var_12, var_8)
    var_14 = '[1, "two", true]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = '{"a": [1, {"b": 2}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'a'
    var_21 = var_19.value[var_20]
    var_22 = 1
    var_23 = var_19.value[var_20]
    var_24 = var_23.value[var_22]
    var_25 = b'{"a": 1}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '   '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"key: "value"}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '[1, 2, ]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{"a": 1'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '1e10'
    var_36 = module_0.tokenize_json(var_35)



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
    var_20 = 'false'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = var_21.value
    assert var_22 is False
    var_23 = 'null'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    assert var_25 is None
    var_26 = '[1, "two", {"three": 3}]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = var_27.value
    var_29 = len(var_28)
    assert var_29 == 3
    var_30 = 2
    var_31 = var_27.value[var_30]
    var_32 = '{"key": "value", "num": 42}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = '[]'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = 'a'
    var_41 = b'{"a": 1}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value[var_40]
    var_44 = var_43.value
    assert var_44 == 1
    var_45 = '  \n  "space"  \t  '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = var_46.value
    assert var_47 == 'space'
    var_48 = ''
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"unclosed": "string'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"key" "value"}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{"a": 1 "b": 2}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{key: "value"}'
    var_57 = module_0.tokenize_json(var_56)



# Parsed testcases at query #14
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
    var_18 = '{"a": [1, {"b": 2}]}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 1
    var_21 = 'a'
    var_22 = var_19.value[var_21]
    var_23 = var_22.value[var_20]
    var_24 = b'{"byte": true}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '  {  "space"  :  1  }  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"key" "value"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '"unclosed'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"a": 1,}'
    var_37 = module_0.tokenize_json(var_36)



