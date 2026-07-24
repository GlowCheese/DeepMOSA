####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 14
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = '{"a": 1, "b": 2}'
    var_15 = 'a'
    var_16 = 2
    var_17 = module_0.ScalarToken(var_15, var_2, var_16, var_14)
    var_18 = 5
    var_19 = 6
    var_20 = module_0.ScalarToken(var_2, var_18, var_19, var_14)
    var_21 = 'b'
    var_22 = 9
    var_23 = 10
    var_24 = module_0.ScalarToken(var_21, var_22, var_23, var_14)
    var_25 = module_0.ScalarToken(var_16, var_7, var_11, var_14)
    var_26 = {var_17: var_20, var_24: var_25}
    var_27 = 15



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_0)
    var_10 = 2
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_0)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = 4
    var_15 = module_0.ScalarToken(var_13, var_1, var_14, var_13)
    var_16 = {var_12: var_15}
    var_17 = module_0.ScalarToken(var_12, var_1, var_10, var_12)
    var_18 = module_0.ScalarToken(var_13, var_14, var_6, var_13)
    var_19 = {var_17: var_18}
    var_20 = 'key: value'
    var_21 = module_0.ScalarToken(var_12, var_1, var_10, var_12)
    var_22 = module_0.ScalarToken(var_13, var_14, var_6, var_13)
    var_23 = {var_21: var_22}
    var_24 = 'key2'
    var_25 = module_0.ScalarToken(var_24, var_1, var_2, var_24)
    var_26 = 5
    var_27 = 9
    var_28 = module_0.ScalarToken(var_13, var_26, var_27, var_13)
    var_29 = {var_25: var_28}
    var_30 = 'key2: value'
    var_31 = 'item'
    var_32 = module_0.ScalarToken(var_31, var_1, var_2, var_31)
    var_33 = [var_32]
    var_34 = module_0.ListToken(var_33, var_1, var_2, var_31)
    var_35 = module_0.ScalarToken(var_31, var_1, var_2, var_31)
    var_36 = [var_35]
    var_37 = module_0.ListToken(var_36, var_1, var_2, var_31)
    var_38 = 'item2'
    var_39 = module_0.ScalarToken(var_38, var_1, var_14, var_38)
    var_40 = [var_39]
    var_41 = module_0.ListToken(var_40, var_1, var_14, var_38)



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 42}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'number'
    var_10 = 15
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 42
    var_14 = 23
    var_15 = 24
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = [var_1]
    var_20 = [var_9]
    var_21 = [var_1]
    var_22 = [var_9]
    var_23 = '{"key": "value"}'
    var_24 = module_0.ScalarToken(var_1, var_2, var_3, var_23)
    var_25 = module_0.ScalarToken(var_5, var_6, var_7, var_23)
    var_26 = {var_24: var_25}
    var_27 = '{}'
    var_28 = {}



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'test1'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 'test2'
    var_9 = module_0.ScalarToken(var_8, var_1, var_6, var_8)
    var_10 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_11 = 1
    var_12 = module_0.ScalarToken(var_0, var_11, var_2, var_0)
    var_13 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_14 = 2
    var_15 = module_0.ScalarToken(var_0, var_1, var_14, var_0)
    var_16 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_17 = 'key'
    var_18 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_19 = {var_17: var_18}
    var_20 = module_0.ScalarToken(var_17, var_1, var_14, var_17)
    var_21 = 'value'
    var_22 = 8
    var_23 = module_0.ScalarToken(var_21, var_6, var_22, var_21)
    var_24 = {var_20: var_23}
    var_25 = 'key: value'
    var_26 = module_0.ScalarToken(var_17, var_1, var_14, var_17)
    var_27 = module_0.ScalarToken(var_21, var_6, var_22, var_21)
    var_28 = {var_26: var_27}
    var_29 = 'item1'
    var_30 = module_0.ScalarToken(var_29, var_1, var_6, var_29)
    var_31 = 'item2'
    var_32 = 6
    var_33 = 10
    var_34 = module_0.ScalarToken(var_31, var_32, var_33, var_31)
    var_35 = [var_30, var_34]
    var_36 = 'item1, item2'
    var_37 = module_0.ListToken(var_35, var_1, var_33, var_36)
    var_38 = module_0.ScalarToken(var_29, var_1, var_6, var_29)
    var_39 = module_0.ScalarToken(var_31, var_32, var_33, var_31)
    var_40 = [var_38, var_39]
    var_41 = module_0.ListToken(var_40, var_1, var_33, var_36)
    var_42 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_43 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_44 = 'test content'
    var_45 = module_0.ScalarToken(var_0, var_1, var_2, var_44)
    var_46 = 5
    var_47 = module_0.ScalarToken(var_0, var_46, var_22, var_44)



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'other'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 3
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = 'key'
    var_12 = 2
    var_13 = '{"key": "value"}'
    var_14 = module_0.ScalarToken(var_11, var_1, var_12, var_13)
    var_15 = 7
    var_16 = 13
    var_17 = module_0.ScalarToken(var_0, var_15, var_16, var_13)
    var_18 = {var_14: var_17}
    var_19 = {var_14: var_17}
    var_20 = '{"key": "other"}'
    var_21 = module_0.ScalarToken(var_5, var_15, var_16, var_20)
    var_22 = {var_14: var_21}
    var_23 = 'item'
    var_24 = 5
    var_25 = '["item"]'
    var_26 = module_0.ScalarToken(var_23, var_7, var_24, var_25)
    var_27 = [var_26]
    var_28 = 6
    var_29 = module_0.ListToken(var_27, var_1, var_28, var_25)
    var_30 = [var_26]
    var_31 = module_0.ListToken(var_30, var_1, var_28, var_25)
    var_32 = '["other"]'
    var_33 = module_0.ScalarToken(var_5, var_7, var_28, var_32)
    var_34 = [var_33]
    var_35 = module_0.ListToken(var_34, var_1, var_15, var_32)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = 'test content'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('test')"
    var_6 = 'key'
    var_7 = 2
    var_8 = 'key: value'
    var_9 = module_0.ScalarToken(var_6, var_1, var_7, var_8)
    var_10 = 'value'
    var_11 = 5
    var_12 = 9
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_8)
    var_14 = {var_9: var_13}
    var_15 = 'item'
    var_16 = '[item]'
    var_17 = module_0.ScalarToken(var_15, var_1, var_2, var_16)
    var_18 = [var_17]
    var_19 = module_0.ListToken(var_18, var_1, var_11, var_16)
    var_20 = repr(var_19)
    assert var_20 == "ListToken('[item]')"
    var_21 = ''
    var_22 = -1
    var_23 = module_0.ScalarToken(var_21, var_1, var_22, var_21)
    var_24 = repr(var_23)
    assert var_24 == "ScalarToken('')"
    var_25 = '\n\t'
    var_26 = 1
    var_27 = '\n\tcontent'
    var_28 = module_0.ScalarToken(var_25, var_1, var_26, var_27)
    var_29 = repr(var_28)
    assert var_29 == "ScalarToken('\\n\\t')"
    var_30 = 'café'
    var_31 = 4
    var_32 = 'café content'
    var_33 = module_0.ScalarToken(var_30, var_1, var_31, var_32)
    var_34 = repr(var_33)
    assert var_34 == "ScalarToken('café')"



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content for tokens'
    var_1 = 42
    var_2 = 0
    var_3 = 2
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = []
    var_6 = 'key1'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_2, var_7, var_0)
    var_9 = 'value1'
    var_10 = 5
    var_11 = 10
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = module_0.DictToken()
    var_15 = [var_6]
    var_16 = 'nested_key'
    var_17 = 9
    var_18 = module_0.ScalarToken(var_16, var_2, var_17, var_0)
    var_19 = 'nested_value'
    var_20 = 11
    var_21 = 22
    var_22 = module_0.ScalarToken(var_19, var_20, var_21, var_0)
    var_23 = {var_18: var_22}
    var_24 = module_0.DictToken()
    var_25 = 'outer_key'
    var_26 = 8
    var_27 = module_0.ScalarToken(var_25, var_2, var_26, var_0)
    var_28 = {var_27: var_24}
    var_29 = module_0.DictToken()
    var_30 = [var_25, var_16]
    var_31 = 'item1'
    var_32 = 4
    var_33 = module_0.ScalarToken(var_31, var_2, var_32, var_0)
    var_34 = 'item2'
    var_35 = 6
    var_36 = module_0.ScalarToken(var_34, var_35, var_11, var_0)
    var_37 = [var_33, var_36]
    var_38 = module_0.ListToken(var_37, var_2, var_11, var_0)
    var_39 = [var_2]
    var_40 = 1
    var_41 = [var_40]
    var_42 = 'dict_key'
    var_43 = 7
    var_44 = module_0.ScalarToken(var_42, var_2, var_43, var_0)
    var_45 = 'dict_value'
    var_46 = 18
    var_47 = module_0.ScalarToken(var_45, var_17, var_46, var_0)
    var_48 = {var_44: var_47}
    var_49 = module_0.DictToken()
    var_50 = [var_49]
    var_51 = module_0.ListToken(var_50, var_2, var_46, var_0)
    var_52 = [var_2, var_42]
    var_53 = []
    var_54 = []
    var_55 = []



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 12
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = 'other'
    var_15 = 5
    var_16 = module_0.ScalarToken(var_14, var_2, var_15, var_0)
    var_17 = {var_16: var_8}
    var_18 = '{"a": 1, "b": 2}'
    var_19 = 'a'
    var_20 = module_0.ScalarToken(var_19, var_2, var_2, var_18)
    var_21 = 6
    var_22 = module_0.ScalarToken(var_2, var_21, var_21, var_18)
    var_23 = 'b'
    var_24 = 10
    var_25 = module_0.ScalarToken(var_23, var_24, var_24, var_18)
    var_26 = 2
    var_27 = 15
    var_28 = module_0.ScalarToken(var_26, var_27, var_27, var_18)
    var_29 = {var_20: var_22, var_25: var_28}
    var_30 = 16



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1: value1\nkey2: value2\nkey3: nested_key: nested_value'
    var_1 = 'nested_key'
    var_2 = 40
    var_3 = 48
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'nested_value'
    var_6 = 51
    var_7 = 62
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 'key1'
    var_11 = 0
    var_12 = 3
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_14 = 'value1'
    var_15 = 6
    var_16 = 11
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_0)
    var_18 = 'key2'
    var_19 = 13
    var_20 = 16
    var_21 = module_0.ScalarToken(var_18, var_19, var_20, var_0)
    var_22 = 'value2'
    var_23 = 19
    var_24 = 24
    var_25 = module_0.ScalarToken(var_22, var_23, var_24, var_0)
    var_26 = 'key3'
    var_27 = 26
    var_28 = 29
    var_29 = module_0.ScalarToken(var_26, var_27, var_28, var_0)
    var_30 = [var_10]
    var_31 = [var_18]
    var_32 = [var_26, var_1]
    var_33 = [var_26, var_1]
    var_34 = [var_10]
    var_35 = 'string'
    var_36 = 'value'
    var_37 = 'start'
    var_38 = 'end'



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.Token(var_0, var_1, var_2, var_0)
    var_4 = var_3.value
    var_5 = None
    var_6 = 5
    var_7 = 10
    var_8 = 'line1\nline2\nline3'
    var_9 = module_0.Token(var_5, var_6, var_7, var_8)
    var_10 = 7
    var_11 = module_0.Token(var_0, var_1, var_2, var_0)
    var_12 = module_0.Token(var_0, var_1, var_2, var_0)
    var_13 = 'different'
    var_14 = module_0.Token(var_13, var_1, var_2, var_0)
    var_15 = repr(var_11)
    assert var_15 == "Token('test')"
    var_16 = 0
    var_17 = [var_16]
    var_18 = var_3.lookup(var_17)
    var_19 = 0
    var_20 = [var_19]
    var_21 = var_3.lookup_key(var_20)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.start
    var_6 = var_4.end
    var_7 = hash(var_4)
    var_8 = hash(var_0)
    var_9 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_10 = 43
    var_11 = '43'
    var_12 = module_0.ScalarToken(var_10, var_1, var_2, var_11)
    var_13 = 2
    var_14 = ' 42'
    var_15 = module_0.ScalarToken(var_0, var_2, var_13, var_14)
    var_16 = repr(var_4)
    assert var_16 == "ScalarToken('42')"
    var_17 = 'hello'
    var_18 = 4
    var_19 = module_0.ScalarToken(var_17, var_1, var_18, var_17)
    var_20 = hash(var_19)
    var_21 = hash(var_17)
    var_22 = None
    var_23 = 3
    var_24 = 'null'
    var_25 = module_0.ScalarToken(var_22, var_1, var_23, var_24)
    var_26 = True
    var_27 = 'true'
    var_28 = module_0.ScalarToken(var_26, var_1, var_23, var_27)
    var_29 = 3.14
    var_30 = '3.14'
    var_31 = module_0.ScalarToken(var_29, var_1, var_23, var_30)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('42')"
    var_6 = 'key'
    var_7 = 2
    var_8 = '{"key": "value"}'
    var_9 = module_0.ScalarToken(var_6, var_1, var_7, var_8)
    var_10 = 'value'
    var_11 = 7
    var_12 = 13
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_8)
    var_14 = {var_9: var_13}
    var_15 = 14
    var_16 = module_0.DictToken()
    var_17 = repr(var_16)
    assert var_17 == 'DictToken(\'{"key": "value"}\')'
    var_18 = 'item'
    var_19 = 4
    var_20 = '["item"]'
    var_21 = module_0.ScalarToken(var_18, var_2, var_19, var_20)
    var_22 = [var_21]
    var_23 = 6
    var_24 = module_0.ListToken(var_22, var_1, var_23, var_20)
    var_25 = repr(var_24)
    assert var_25 == 'ListToken(\'["item"]\')'
    var_26 = ''
    var_27 = -1
    var_28 = module_0.ScalarToken(var_26, var_1, var_27, var_26)
    var_29 = repr(var_28)
    assert var_29 == "ScalarToken('')"
    var_30 = 'test\nvalue'
    var_31 = 9
    var_32 = module_0.ScalarToken(var_30, var_1, var_31, var_30)
    var_33 = repr(var_32)
    assert var_33 == "ScalarToken('test\\nvalue')"
    var_34 = 'partial'
    var_35 = 5
    var_36 = 11
    var_37 = 'full content with partial text'
    var_38 = module_0.ScalarToken(var_34, var_35, var_36, var_37)
    var_39 = repr(var_38)
    assert var_39 == "ScalarToken('partial')"



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 5
    var_8 = 8
    var_9 = 'other test'
    var_10 = module_0.ScalarToken(var_0, var_7, var_8, var_9)
    var_11 = hash(var_3)
    var_12 = hash(var_10)
    var_13 = 'different'
    var_14 = module_0.ScalarToken(var_13, var_1, var_8, var_13)
    var_15 = hash(var_3)
    var_16 = hash(var_14)
    var_17 = 42
    var_18 = 1
    var_19 = '42'
    var_20 = module_0.ScalarToken(var_17, var_1, var_18, var_19)
    var_21 = 10
    var_22 = 11
    var_23 = ' 42 '
    var_24 = module_0.ScalarToken(var_17, var_21, var_22, var_23)
    var_25 = hash(var_20)
    var_26 = hash(var_24)
    var_27 = True
    var_28 = 'True'
    var_29 = module_0.ScalarToken(var_27, var_1, var_2, var_28)
    var_30 = True
    var_31 = ' True'
    var_32 = module_0.ScalarToken(var_30, var_7, var_8, var_31)
    var_33 = hash(var_29)
    var_34 = hash(var_32)
    var_35 = 'consistent'
    var_36 = 9
    var_37 = module_0.ScalarToken(var_35, var_1, var_36, var_35)
    var_38 = hash(var_37)
    var_39 = hash(var_37)
    var_40 = None
    var_41 = 'None'
    var_42 = module_0.ScalarToken(var_40, var_1, var_2, var_41)
    var_43 = module_0.ScalarToken(var_40, var_7, var_8, var_41)
    var_44 = hash(var_42)
    var_45 = hash(var_43)
    var_46 = 'a'
    var_47 = module_0.ScalarToken(var_46, var_1, var_1, var_46)
    var_48 = ' a'
    var_49 = module_0.ScalarToken(var_46, var_30, var_30, var_48)
    var_50 = 'b'
    var_51 = module_0.ScalarToken(var_50, var_1, var_1, var_50)
    var_52 = 2
    var_53 = ' b'
    var_54 = module_0.ScalarToken(var_50, var_52, var_52, var_53)
    var_55 = {var_47, var_49, var_51, var_54}
    var_56 = len(var_55)
    assert var_56 == 2



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 123
    var_2 = 0
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('test')"
    var_6 = 'key'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_2, var_7, var_0)
    var_9 = 'value'
    var_10 = 4
    var_11 = 8
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = module_0.DictToken()
    var_15 = repr(var_14)
    assert var_15 == "DictToken('test cont')"
    var_16 = 'item'
    var_17 = module_0.ScalarToken(var_16, var_2, var_3, var_0)
    var_18 = [var_17]
    var_19 = module_0.ListToken(var_18, var_2, var_3, var_0)
    var_20 = repr(var_19)
    assert var_20 == "ListToken('test')"
    var_21 = None
    var_22 = -1
    var_23 = ''
    var_24 = module_0.ScalarToken(var_21, var_2, var_22, var_23)
    var_25 = repr(var_24)
    assert var_25 == "ScalarToken('')"
    var_26 = 'a'
    var_27 = module_0.ScalarToken(var_26, var_2, var_2, var_26)
    var_28 = repr(var_27)
    assert var_28 == "ScalarToken('a')"
    var_29 = 'line1\nline2'
    var_30 = 'text'
    var_31 = module_0.ScalarToken(var_30, var_2, var_10, var_29)
    var_32 = repr(var_31)
    assert var_32 == "ScalarToken('line1')"



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 5
    var_8 = 8
    var_9 = 'other test'
    var_10 = module_0.ScalarToken(var_0, var_7, var_8, var_9)
    var_11 = hash(var_3)
    var_12 = hash(var_10)
    var_13 = 'different'
    var_14 = module_0.ScalarToken(var_13, var_1, var_8, var_13)
    var_15 = hash(var_3)
    var_16 = hash(var_14)
    var_17 = 42
    var_18 = 1
    var_19 = '42'
    var_20 = module_0.ScalarToken(var_17, var_1, var_18, var_19)
    var_21 = 10
    var_22 = 11
    var_23 = 'xx42xx'
    var_24 = module_0.ScalarToken(var_17, var_21, var_22, var_23)
    var_25 = hash(var_20)
    var_26 = hash(var_24)
    var_27 = None
    var_28 = 'null'
    var_29 = module_0.ScalarToken(var_27, var_1, var_2, var_28)
    var_30 = module_0.ScalarToken(var_27, var_7, var_8, var_28)
    var_31 = hash(var_29)
    var_32 = hash(var_30)
    var_33 = 'consistent'
    var_34 = module_0.ScalarToken(var_33, var_1, var_8, var_33)
    var_35 = hash(var_34)
    var_36 = hash(var_34)
    var_37 = True
    var_38 = 'true'
    var_39 = module_0.ScalarToken(var_37, var_1, var_2, var_38)
    var_40 = True
    var_41 = 13
    var_42 = module_0.ScalarToken(var_40, var_21, var_41, var_38)
    var_43 = False
    var_44 = 4
    var_45 = 'false'
    var_46 = module_0.ScalarToken(var_43, var_43, var_44, var_45)
    var_47 = hash(var_39)
    var_48 = hash(var_42)
    var_49 = hash(var_39)
    var_50 = hash(var_46)
    var_51 = 3.14
    var_52 = '3.14'
    var_53 = module_0.ScalarToken(var_51, var_43, var_2, var_52)
    var_54 = module_0.ScalarToken(var_51, var_7, var_8, var_52)
    var_55 = hash(var_53)
    var_56 = hash(var_54)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_0)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 2
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = 'key'
    var_12 = '"key": "value"'
    var_13 = module_0.ScalarToken(var_11, var_1, var_9, var_12)
    var_14 = 'value'
    var_15 = 6
    var_16 = 12
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_13: var_17}
    var_19 = {var_13: var_17}
    var_20 = 'item'
    var_21 = '["item"]'
    var_22 = module_0.ScalarToken(var_20, var_1, var_2, var_21)
    var_23 = [var_22]
    var_24 = module_0.ListToken(var_23, var_1, var_15, var_21)
    var_25 = [var_22]
    var_26 = module_0.ListToken(var_25, var_1, var_15, var_21)
    var_27 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_28 = {}



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = ' test'
    var_10 = module_0.ScalarToken(var_0, var_8, var_2, var_9)
    var_11 = 2
    var_12 = 'tes'
    var_13 = module_0.ScalarToken(var_0, var_1, var_11, var_12)
    var_14 = 'key'
    var_15 = '{"key": "value"}'
    var_16 = module_0.ScalarToken(var_14, var_1, var_11, var_15)
    var_17 = 'value'
    var_18 = 7
    var_19 = 13
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_15)
    var_21 = {var_16: var_20}
    var_22 = 15
    var_23 = module_0.ScalarToken(var_14, var_1, var_11, var_15)
    var_24 = module_0.ScalarToken(var_17, var_18, var_19, var_15)
    var_25 = {var_23: var_24}
    var_26 = 'key2'
    var_27 = '{"key2": "value"}'
    var_28 = module_0.ScalarToken(var_26, var_1, var_2, var_27)
    var_29 = 9
    var_30 = 14
    var_31 = module_0.ScalarToken(var_17, var_29, var_30, var_27)
    var_32 = {var_28: var_31}
    var_33 = 16
    var_34 = 'item1'
    var_35 = 4
    var_36 = '["item1"]'
    var_37 = module_0.ScalarToken(var_34, var_1, var_35, var_36)
    var_38 = 'item2'
    var_39 = 11
    var_40 = '["item1", "item2"]'
    var_41 = module_0.ScalarToken(var_38, var_18, var_39, var_40)
    var_42 = [var_37, var_41]
    var_43 = module_0.ListToken(var_42, var_1, var_19, var_40)
    var_44 = module_0.ScalarToken(var_34, var_1, var_35, var_36)
    var_45 = module_0.ScalarToken(var_38, var_18, var_39, var_40)
    var_46 = [var_44, var_45]
    var_47 = module_0.ListToken(var_46, var_1, var_19, var_40)
    var_48 = 'item3'
    var_49 = '["item3"]'
    var_50 = module_0.ScalarToken(var_48, var_1, var_35, var_49)
    var_51 = [var_50]
    var_52 = 6
    var_53 = module_0.ListToken(var_51, var_1, var_52, var_49)
    var_54 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_55 = 5
    var_56 = 'abc test'
    var_57 = module_0.ScalarToken(var_0, var_55, var_6, var_56)
    var_58 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_59 = 'best'
    var_60 = module_0.ScalarToken(var_59, var_1, var_2, var_59)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = 4
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'other'
    var_6 = module_0.ScalarToken(var_5, var_1, var_2, var_5)
    var_7 = 1
    var_8 = module_0.ScalarToken(var_0, var_7, var_2, var_0)
    var_9 = 3
    var_10 = module_0.ScalarToken(var_0, var_1, var_9, var_0)
    var_11 = 'key'
    var_12 = 2
    var_13 = module_0.ScalarToken(var_11, var_1, var_12, var_11)
    var_14 = 8
    var_15 = module_0.ScalarToken(var_0, var_2, var_14, var_0)
    var_16 = {var_13: var_15}
    var_17 = 'key: value'
    var_18 = {var_13: var_15}
    var_19 = [var_15]
    var_20 = '[value]'
    var_21 = module_0.ListToken(var_19, var_1, var_14, var_20)
    var_22 = [var_15]
    var_23 = module_0.ListToken(var_22, var_1, var_14, var_20)
    var_24 = 'test'
    var_25 = module_0.ScalarToken(var_24, var_1, var_9, var_24)
    var_26 = [var_25]
    var_27 = '[test]'
    var_28 = module_0.ListToken(var_26, var_1, var_9, var_27)
    var_29 = 'different content'
    var_30 = module_0.ScalarToken(var_0, var_1, var_2, var_29)
    var_31 = 'other content'
    var_32 = module_0.ScalarToken(var_0, var_1, var_2, var_31)
    var_33 = 'nested'
    var_34 = 5
    var_35 = module_0.ScalarToken(var_33, var_1, var_34, var_33)
    var_36 = 'data'
    var_37 = 7
    var_38 = 10
    var_39 = module_0.ScalarToken(var_36, var_37, var_38, var_36)
    var_40 = {var_35: var_39}
    var_41 = 'nested: data'
    var_42 = 'outer'
    var_43 = module_0.ScalarToken(var_42, var_1, var_2, var_42)
    var_44 = 'outer: nested: data'



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 0
    var_2 = 3
    var_3 = 'name: John'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'John'
    var_6 = 6
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'age'
    var_10 = 11
    var_11 = 13
    var_12 = 'name: John, age: 30'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 30
    var_15 = 17
    var_16 = 18
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = [var_0]
    var_20 = [var_9]
    var_21 = [var_0]
    var_22 = [var_9]
    var_23 = 'different content'
    var_24 = 1
    var_25 = 'data'
    var_26 = 'data: {x: 1}'
    var_27 = module_0.ScalarToken(var_25, var_1, var_2, var_26)
    var_28 = 'x'
    var_29 = 7
    var_30 = module_0.ScalarToken(var_28, var_29, var_29, var_26)
    var_31 = 10
    var_32 = module_0.ScalarToken(var_24, var_31, var_31, var_26)
    var_33 = {var_30: var_32}
    var_34 = [var_25]
    var_35 = [var_25, var_28]
    var_36 = [var_25]



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 14
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = '{"a": 1, "b": 2}'
    var_15 = 'a'
    var_16 = 2
    var_17 = module_0.ScalarToken(var_15, var_2, var_16, var_14)
    var_18 = 5
    var_19 = 6
    var_20 = module_0.ScalarToken(var_2, var_18, var_19, var_14)
    var_21 = 'b'
    var_22 = 8
    var_23 = 9
    var_24 = module_0.ScalarToken(var_21, var_22, var_23, var_14)
    var_25 = 12
    var_26 = module_0.ScalarToken(var_16, var_25, var_7, var_14)
    var_27 = {var_17: var_20, var_24: var_26}



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "num": 42}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'num'
    var_10 = 15
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 42
    var_14 = 20
    var_15 = 21
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = [var_1]
    var_20 = [var_9]
    var_21 = [var_1]
    var_22 = [var_9]
    var_23 = 5
    var_24 = 25
    var_25 = '{}'
    var_26 = {}



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 14
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = 'other'
    var_15 = 6
    var_16 = module_0.ScalarToken(var_14, var_2, var_15, var_0)
    var_17 = {var_16: var_8}
    var_18 = 'different content'
    var_19 = '{"a": 1, "b": 2}'
    var_20 = 'a'
    var_21 = 2
    var_22 = module_0.ScalarToken(var_20, var_2, var_21, var_19)
    var_23 = 'b'
    var_24 = 8
    var_25 = 9
    var_26 = module_0.ScalarToken(var_23, var_24, var_25, var_19)
    var_27 = 5
    var_28 = module_0.ScalarToken(var_2, var_27, var_15, var_19)
    var_29 = 12
    var_30 = module_0.ScalarToken(var_21, var_29, var_7, var_19)
    var_31 = {var_22: var_28, var_26: var_30}
    var_32 = [var_20]
    var_33 = [var_23]
    var_34 = [var_20]
    var_35 = [var_23]



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 42}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 8
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'number'
    var_10 = 17
    var_11 = 23
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 42
    var_14 = 26
    var_15 = 28
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = 29
    var_20 = [var_1]
    var_21 = [var_9]
    var_22 = [var_1]
    var_23 = [var_9]
    var_24 = {}
    var_25 = 2
    var_26 = '{}'



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = {var_4: var_8}
    var_10 = 0
    var_11 = 12
    var_12 = [var_1]
    var_13 = [var_1]
    var_14 = '{"a": 1, "b": 2}'
    var_15 = 'a'
    var_16 = module_0.ScalarToken(var_15, var_2, var_2, var_14)
    var_17 = 6
    var_18 = module_0.ScalarToken(var_2, var_17, var_17, var_14)
    var_19 = 'b'
    var_20 = 10
    var_21 = module_0.ScalarToken(var_19, var_20, var_20, var_14)
    var_22 = 2
    var_23 = 15
    var_24 = module_0.ScalarToken(var_22, var_23, var_23, var_14)
    var_25 = {var_16: var_18, var_21: var_24}
    var_26 = 16
    var_27 = '{}'
    var_28 = {}



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 42}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 8
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'number'
    var_10 = 17
    var_11 = 23
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 42
    var_14 = 26
    var_15 = 28
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = [var_1]
    var_20 = [var_9]
    var_21 = [var_1]
    var_22 = [var_9]
    var_23 = {}
    var_24 = 2
    var_25 = '{}'
    var_26 = '{\n  "key": "value"\n}'
    var_27 = {}
    var_28 = 16



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = 'key1: value1, key2: value2'
    var_11 = module_0.ScalarToken(var_0, var_1, var_2, var_10)
    var_12 = module_0.ScalarToken(var_5, var_6, var_7, var_10)
    var_13 = 'key2'
    var_14 = 14
    var_15 = 17
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_10)
    var_17 = 'value2'
    var_18 = 20
    var_19 = 25
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_10)
    var_21 = {var_11: var_12, var_16: var_20}
    var_22 = [var_0]
    var_23 = [var_13]
    var_24 = [var_0]
    var_25 = [var_13]
    var_26 = 'different content'
    var_27 = 'test'
    var_28 = module_0.ScalarToken(var_27, var_1, var_2, var_27)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value", "num": 42}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 3
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'num'
    var_10 = 16
    var_11 = 18
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 42
    var_14 = 21
    var_15 = 22
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_4: var_8, var_12: var_16}
    var_18 = 0
    var_19 = [var_1]
    var_20 = [var_9]
    var_21 = [var_1]
    var_22 = [var_9]
    var_23 = 'different content'
    var_24 = {}
    var_25 = '{}'



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = 'key1'
    var_4 = 3
    var_5 = module_0.ScalarToken(var_3, var_1, var_4, var_3)
    var_6 = 'value1'
    var_7 = 5
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_6)
    var_10 = 'key2'
    var_11 = 12
    var_12 = 15
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_10)
    var_14 = 123
    var_15 = 17
    var_16 = 19
    var_17 = '123'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_5: var_9, var_13: var_18}
    var_20 = 'key1:value1 key2:123'
    var_21 = [var_3]
    var_22 = [var_10]
    var_23 = [var_3]
    var_24 = [var_10]
    var_25 = 'different content'
    var_26 = 'test'
    var_27 = module_0.ScalarToken(var_26, var_1, var_4, var_26)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 1
    var_3 = '42'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = repr(var_4)
    assert var_5 == "ScalarToken('42')"
    var_6 = 'key'
    var_7 = 2
    var_8 = '{"key": "value"}'
    var_9 = module_0.ScalarToken(var_6, var_1, var_7, var_8)
    var_10 = 'value'
    var_11 = 7
    var_12 = 13
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_8)
    var_14 = {var_9: var_13}
    var_15 = 14
    var_16 = module_0.DictToken()
    var_17 = repr(var_16)
    assert var_17 == 'DictToken(\'{"key": "value"}\')'
    var_18 = 'item'
    var_19 = 4
    var_20 = '["item"]'
    var_21 = module_0.ScalarToken(var_18, var_2, var_19, var_20)
    var_22 = [var_21]
    var_23 = 6
    var_24 = module_0.ListToken(var_22, var_1, var_23, var_20)
    var_25 = repr(var_24)
    assert var_25 == 'ListToken(\'["item"]\')'
    var_26 = ''
    var_27 = -1
    var_28 = module_0.ScalarToken(var_26, var_1, var_27, var_26)
    var_29 = repr(var_28)
    assert var_29 == "ScalarToken('')"
    var_30 = 'hello\nworld'
    var_31 = 10
    var_32 = module_0.ScalarToken(var_30, var_1, var_31, var_30)
    var_33 = repr(var_32)
    assert var_33 == "ScalarToken('hello\\nworld')"
    var_34 = 'test'
    var_35 = 3
    var_36 = 'test\t\n\r"'
    var_37 = module_0.ScalarToken(var_34, var_1, var_35, var_36)
    var_38 = repr(var_37)
    assert var_38 == 'ScalarToken(\'test\\t\\n\\r"\')'



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = repr(var_3)
    assert var_4 == "ScalarToken('test')"
    var_5 = 'key'
    var_6 = 2
    var_7 = '{"key": "value"}'
    var_8 = module_0.ScalarToken(var_5, var_1, var_6, var_7)
    var_9 = 'value'
    var_10 = 7
    var_11 = 13
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_7)
    var_13 = {var_8: var_12}
    var_14 = 14
    var_15 = module_0.DictToken()
    var_16 = repr(var_15)
    assert var_16 == 'DictToken(\'{"key": "value"}\')'
    var_17 = 'item'
    var_18 = 1
    var_19 = 4
    var_20 = '["item"]'
    var_21 = module_0.ScalarToken(var_17, var_18, var_19, var_20)
    var_22 = [var_21]
    var_23 = 6
    var_24 = module_0.ListToken(var_22, var_1, var_23, var_20)
    var_25 = repr(var_24)
    assert var_25 == 'ListToken(\'["item"]\')'
    var_26 = ''
    var_27 = -1
    var_28 = module_0.ScalarToken(var_26, var_1, var_27, var_26)
    var_29 = repr(var_28)
    assert var_29 == "ScalarToken('')"
    var_30 = 'test\nvalue'
    var_31 = 9
    var_32 = module_0.ScalarToken(var_30, var_1, var_31, var_30)
    var_33 = repr(var_32)
    assert var_33 == "ScalarToken('test\\nvalue')"



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = 42
    var_13 = 1
    var_14 = '42'
    var_15 = module_0.ScalarToken(var_12, var_1, var_13, var_14)
    var_16 = module_0.ScalarToken(var_12, var_1, var_13, var_14)
    var_17 = hash(var_15)
    var_18 = hash(var_16)
    var_19 = True
    var_20 = 'True'
    var_21 = module_0.ScalarToken(var_19, var_1, var_2, var_20)
    var_22 = True
    var_23 = module_0.ScalarToken(var_22, var_1, var_2, var_20)
    var_24 = hash(var_21)
    var_25 = hash(var_23)
    var_26 = None
    var_27 = 'None'
    var_28 = module_0.ScalarToken(var_26, var_1, var_2, var_27)
    var_29 = module_0.ScalarToken(var_26, var_1, var_2, var_27)
    var_30 = hash(var_28)
    var_31 = hash(var_29)
    var_32 = 'test_value'
    var_33 = len(var_32)
    var_34 = var_33 - var_22
    var_35 = module_0.ScalarToken(var_32, var_1, var_34, var_32)
    var_36 = hash(var_35)
    var_37 = hash(var_32)
    var_38 = 3.14
    var_39 = '3.14'
    var_40 = module_0.ScalarToken(var_38, var_1, var_2, var_39)
    var_41 = module_0.ScalarToken(var_38, var_1, var_2, var_39)
    var_42 = hash(var_40)
    var_43 = hash(var_41)
    var_44 = 'same'
    var_45 = module_0.ScalarToken(var_44, var_1, var_2, var_44)
    var_46 = 5
    var_47 = 'xxxxxsame'
    var_48 = module_0.ScalarToken(var_44, var_46, var_8, var_47)
    var_49 = hash(var_45)
    var_50 = hash(var_48)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = repr(var_3)
    assert var_4 == "ScalarToken('test')"
    var_5 = ''
    var_6 = -1
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = repr(var_7)
    assert var_8 == "ScalarToken('')"
    var_9 = 'a\nb'
    var_10 = 2
    var_11 = module_0.ScalarToken(var_9, var_1, var_10, var_9)
    var_12 = repr(var_11)
    assert var_12 == "ScalarToken('a\\nb')"
    var_13 = 'key'
    var_14 = module_0.ScalarToken(var_13, var_1, var_10, var_13)
    var_15 = 'value'
    var_16 = 4
    var_17 = 8
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_15)
    var_19 = {var_14: var_18}
    var_20 = 'key: value'
    var_21 = module_0.DictToken()
    var_22 = repr(var_21)
    assert var_22 == "DictToken('key: value')"
    var_23 = 'item'
    var_24 = 1
    var_25 = '[item]'
    var_26 = module_0.ScalarToken(var_23, var_24, var_16, var_25)
    var_27 = [var_26]
    var_28 = 5
    var_29 = module_0.ListToken(var_27, var_1, var_28, var_25)
    var_30 = repr(var_29)
    assert var_30 == "ListToken('[item]')"
    var_31 = 'hello'
    var_32 = 6
    var_33 = 'xhelloy'
    var_34 = module_0.ScalarToken(var_31, var_10, var_32, var_33)
    var_35 = repr(var_34)
    assert var_35 == "ScalarToken('hello')"



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = repr(var_3)
    assert var_4 == "ScalarToken('test')"
    var_5 = ''
    var_6 = -1
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = repr(var_7)
    assert var_8 == "ScalarToken('')"
    var_9 = 'hello\nworld'
    var_10 = 10
    var_11 = module_0.ScalarToken(var_9, var_1, var_10, var_9)
    var_12 = repr(var_11)
    assert var_12 == "ScalarToken('hello\\nworld')"
    var_13 = '\t\n\r'
    var_14 = 2
    var_15 = module_0.ScalarToken(var_13, var_1, var_14, var_13)
    var_16 = repr(var_15)
    assert var_16 == "ScalarToken('\\t\\n\\r')"
    var_17 = '{"key": "value"}'
    var_18 = 'key'
    var_19 = 1
    var_20 = 4
    var_21 = module_0.ScalarToken(var_18, var_19, var_20, var_17)
    var_22 = 'value'
    var_23 = 7
    var_24 = 13
    var_25 = module_0.ScalarToken(var_22, var_23, var_24, var_17)
    var_26 = {var_21: var_25}
    var_27 = 14
    var_28 = '[1, 2, 3]'
    var_29 = module_0.ScalarToken(var_19, var_19, var_19, var_28)
    var_30 = module_0.ScalarToken(var_14, var_20, var_20, var_28)
    var_31 = module_0.ScalarToken(var_2, var_23, var_23, var_28)
    var_32 = [var_29, var_30, var_31]
    var_33 = 8
    var_34 = module_0.ListToken(var_32, var_1, var_33, var_28)
    var_35 = repr(var_34)
    assert var_35 == "ListToken('[1, 2, 3]')"
    var_36 = 'partial'
    var_37 = 5
    var_38 = 11
    var_39 = 'full content partial text'
    var_40 = module_0.ScalarToken(var_36, var_37, var_38, var_39)
    var_41 = repr(var_40)
    assert var_41 == "ScalarToken('partial')"



# Parsed testcases at query #21
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": {"nested_key": "nested_value"}}'
    var_1 = 'key1'
    var_2 = 2
    var_3 = 6
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value1'
    var_6 = 10
    var_7 = 16
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 20
    var_11 = 24
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'nested_key'
    var_14 = 28
    var_15 = 38
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = 'nested_value'
    var_18 = 42
    var_19 = 54
    var_20 = module_0.ScalarToken(var_17, var_18, var_19, var_0)
    var_21 = {var_16: var_20}
    var_22 = 27
    var_23 = 55
    var_24 = 1
    var_25 = 56
    var_26 = 0
    var_27 = [var_26]
    var_28 = [var_24]
    var_29 = [var_24, var_26]
    var_30 = None
    var_31 = ''
    var_32 = [var_26]
    var_33 = {}
    var_34 = 0
    var_35 = [var_34]



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 0
    var_2 = 3
    var_3 = 'key1: value1'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value1'
    var_6 = 6
    var_7 = 11
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = 'key2'
    var_10 = 13
    var_11 = 16
    var_12 = 'key1: value1, key2: value2'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = 'value2'
    var_15 = 19
    var_16 = 24
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_12)
    var_18 = {var_4: var_8, var_13: var_17}
    var_19 = [var_0]
    var_20 = [var_9]
    var_21 = [var_0]
    var_22 = [var_9]
    var_23 = 'different content'
    var_24 = 'test'
    var_25 = module_0.ScalarToken(var_24, var_1, var_2, var_24)
    var_26 = {}
    var_27 = ''
    var_28 = 'nested'
    var_29 = 5
    var_30 = 'nested: [1, 2]'
    var_31 = module_0.ScalarToken(var_28, var_1, var_29, var_30)
    var_32 = 1
    var_33 = 8
    var_34 = module_0.ScalarToken(var_32, var_33, var_33, var_30)
    var_35 = 2
    var_36 = module_0.ScalarToken(var_35, var_7, var_7, var_30)
    var_37 = [var_34, var_36]
    var_38 = 7
    var_39 = 12
    var_40 = module_0.ListToken(var_37, var_38, var_39, var_30)
    var_41 = {var_31: var_40}



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = '{"key": ["item1", "item2"]}'
    var_1 = 'item1'
    var_2 = 11
    var_3 = 16
    var_4 = 'item2'
    var_5 = 18
    var_6 = 23
    var_7 = 10
    var_8 = 24
    var_9 = 'key'
    var_10 = 2
    var_11 = 4
    var_12 = 0
    var_13 = 25
    var_14 = [var_9, var_12]
    var_15 = []
    var_16 = 1
    var_17 = [var_16]
    var_18 = [var_9]
    var_19 = 0
    var_20 = [var_19]
    var_21 = 'inner'
    var_22 = 6
    var_23 = 'value'
    var_24 = 9
    var_25 = 13
    var_26 = 14
    var_27 = 'outer'
    var_28 = 17
    var_29 = 21
    var_30 = 15
    var_31 = [var_27, var_21]
    var_32 = [var_9]
    var_33 = 5
    var_34 = [var_33]
    var_35 = 'nonexistent'
    var_36 = [var_35]



# Parsed testcases at query #24
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = []
    var_5 = '[1, 2, 3]'
    var_6 = 1
    var_7 = module_0.ScalarToken(var_6, var_6, var_6, var_5)
    var_8 = 2
    var_9 = 4
    var_10 = module_0.ScalarToken(var_8, var_9, var_9, var_5)
    var_11 = 7
    var_12 = module_0.ScalarToken(var_2, var_11, var_11, var_5)
    var_13 = [var_7, var_10, var_12]
    var_14 = 8
    var_15 = module_0.ListToken(var_13, var_1, var_14, var_5)
    var_16 = [var_1]
    var_17 = [var_6]
    var_18 = [var_8]
    var_19 = '[[1, 2], [3, 4]]'
    var_20 = module_0.ScalarToken(var_6, var_8, var_8, var_19)
    var_21 = 5
    var_22 = module_0.ScalarToken(var_8, var_21, var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = 6
    var_25 = module_0.ListToken(var_23, var_6, var_24, var_19)
    var_26 = 10
    var_27 = module_0.ScalarToken(var_2, var_26, var_26, var_19)
    var_28 = 13
    var_29 = module_0.ScalarToken(var_9, var_28, var_28, var_19)
    var_30 = [var_27, var_29]
    var_31 = 9
    var_32 = 14
    var_33 = module_0.ListToken(var_30, var_31, var_32, var_19)
    var_34 = [var_25, var_33]
    var_35 = 15
    var_36 = module_0.ListToken(var_34, var_1, var_35, var_19)
    var_37 = [var_1, var_1]
    var_38 = [var_1, var_6]
    var_39 = [var_6, var_1]
    var_40 = [var_6, var_6]
    var_41 = '{"a": 1, "b": 2}'
    var_42 = 'a'
    var_43 = module_0.ScalarToken(var_42, var_6, var_6, var_41)
    var_44 = 'b'
    var_45 = module_0.ScalarToken(var_44, var_31, var_31, var_41)
    var_46 = module_0.ScalarToken(var_6, var_21, var_21, var_41)
    var_47 = module_0.ScalarToken(var_8, var_28, var_28, var_41)
    var_48 = {var_43: var_46, var_45: var_47}
    var_49 = [var_42]
    var_50 = [var_44]
    var_51 = '{"x": {"y": 5}}'
    var_52 = 'y'
    var_53 = module_0.ScalarToken(var_52, var_11, var_11, var_51)
    var_54 = 12
    var_55 = module_0.ScalarToken(var_21, var_54, var_54, var_51)
    var_56 = {var_53: var_55}
    var_57 = 'x'
    var_58 = module_0.ScalarToken(var_57, var_6, var_6, var_51)
    var_59 = [var_57, var_52]
    var_60 = '{"list": [{"nested": "value"}]}'
    var_61 = 'nested'
    var_62 = 11
    var_63 = 16
    var_64 = module_0.ScalarToken(var_61, var_62, var_63, var_60)
    var_65 = 'value'
    var_66 = 20
    var_67 = 24
    var_68 = module_0.ScalarToken(var_65, var_66, var_67, var_60)
    var_69 = {var_64: var_68}
    var_70 = 25
    var_71 = 26
    var_72 = 'list'
    var_73 = module_0.ScalarToken(var_72, var_6, var_9, var_60)
    var_74 = 27
    var_75 = [var_72, var_1, var_61]
    var_76 = []
    var_77 = []
    var_78 = []



# Parsed testcases at query #25
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_0)
    var_10 = 2
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_0)
    var_12 = 'key'
    var_13 = '"key": "value"'
    var_14 = module_0.ScalarToken(var_12, var_1, var_10, var_13)
    var_15 = 'value'
    var_16 = 7
    var_17 = 13
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_13)
    var_19 = {var_14: var_18}
    var_20 = {var_14: var_18}
    var_21 = 'item'
    var_22 = module_0.ScalarToken(var_21, var_1, var_2, var_21)
    var_23 = [var_22]
    var_24 = module_0.ListToken(var_23, var_1, var_2, var_21)
    var_25 = module_0.ScalarToken(var_21, var_1, var_2, var_21)
    var_26 = [var_25]
    var_27 = module_0.ListToken(var_26, var_1, var_2, var_21)
    var_28 = 'test content'
    var_29 = module_0.ScalarToken(var_0, var_1, var_2, var_28)
    var_30 = 'different content'
    var_31 = module_0.ScalarToken(var_0, var_1, var_2, var_30)
    var_32 = 'test test'
    var_33 = module_0.ScalarToken(var_0, var_1, var_2, var_32)
    var_34 = 5
    var_35 = module_0.ScalarToken(var_0, var_34, var_6, var_32)



# Parsed testcases at query #26
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = hash(var_3)
    var_6 = hash(var_4)
    var_7 = 'different'
    var_8 = 8
    var_9 = module_0.ScalarToken(var_7, var_1, var_8, var_7)
    var_10 = hash(var_3)
    var_11 = hash(var_9)
    var_12 = 5
    var_13 = 'othertest'
    var_14 = module_0.ScalarToken(var_0, var_12, var_8, var_13)
    var_15 = hash(var_3)
    var_16 = hash(var_14)
    var_17 = 42
    var_18 = 1
    var_19 = '42'
    var_20 = module_0.ScalarToken(var_17, var_1, var_18, var_19)
    var_21 = module_0.ScalarToken(var_17, var_1, var_18, var_19)
    var_22 = hash(var_20)
    var_23 = hash(var_21)
    var_24 = 3.14
    var_25 = '3.14'
    var_26 = module_0.ScalarToken(var_24, var_1, var_2, var_25)
    var_27 = module_0.ScalarToken(var_24, var_1, var_2, var_25)
    var_28 = hash(var_26)
    var_29 = hash(var_27)
    var_30 = None
    var_31 = 'null'
    var_32 = module_0.ScalarToken(var_30, var_1, var_2, var_31)
    var_33 = module_0.ScalarToken(var_30, var_1, var_2, var_31)
    var_34 = hash(var_32)
    var_35 = hash(var_33)
    var_36 = True
    var_37 = 'true'
    var_38 = module_0.ScalarToken(var_36, var_1, var_2, var_37)
    var_39 = True
    var_40 = module_0.ScalarToken(var_39, var_1, var_2, var_37)
    var_41 = hash(var_38)
    var_42 = hash(var_40)
    var_43 = 'consistent'
    var_44 = module_0.ScalarToken(var_43, var_1, var_8, var_43)
    var_45 = hash(var_44)
    var_46 = hash(var_44)



# Parsed testcases at query #27
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = 4
    var_10 = ' test'
    var_11 = module_0.ScalarToken(var_0, var_8, var_9, var_10)
    var_12 = 'tes'
    var_13 = 2
    var_14 = module_0.ScalarToken(var_12, var_1, var_13, var_0)
    var_15 = 'key1'
    var_16 = '{"key1": "value1"}'
    var_17 = module_0.ScalarToken(var_15, var_1, var_2, var_16)
    var_18 = 'value1'
    var_19 = 7
    var_20 = 13
    var_21 = module_0.ScalarToken(var_18, var_19, var_20, var_16)
    var_22 = {var_17: var_21}
    var_23 = 15
    var_24 = module_0.ScalarToken(var_15, var_1, var_2, var_16)
    var_25 = module_0.ScalarToken(var_18, var_19, var_20, var_16)
    var_26 = {var_24: var_25}
    var_27 = '{"key1": "value2"}'
    var_28 = module_0.ScalarToken(var_15, var_1, var_2, var_27)
    var_29 = 'value2'
    var_30 = module_0.ScalarToken(var_29, var_19, var_20, var_27)
    var_31 = {var_28: var_30}
    var_32 = 'item1'
    var_33 = 5
    var_34 = '["item1"]'
    var_35 = module_0.ScalarToken(var_32, var_8, var_33, var_34)
    var_36 = [var_35]
    var_37 = module_0.ListToken(var_36, var_1, var_19, var_34)
    var_38 = module_0.ScalarToken(var_32, var_8, var_33, var_34)
    var_39 = [var_38]
    var_40 = module_0.ListToken(var_39, var_1, var_19, var_34)
    var_41 = 'item2'
    var_42 = '["item2"]'
    var_43 = module_0.ScalarToken(var_41, var_8, var_33, var_42)
    var_44 = [var_43]
    var_45 = module_0.ListToken(var_44, var_1, var_19, var_42)



# Parsed testcases at query #28
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = 3
    var_3 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_0)
    var_5 = 'different'
    var_6 = 8
    var_7 = module_0.ScalarToken(var_5, var_1, var_6, var_5)
    var_8 = 1
    var_9 = module_0.ScalarToken(var_0, var_8, var_2, var_0)
    var_10 = 2
    var_11 = module_0.ScalarToken(var_0, var_1, var_10, var_0)
    var_12 = 'key'
    var_13 = '"key": "value"'
    var_14 = module_0.ScalarToken(var_12, var_1, var_10, var_13)
    var_15 = 'value'
    var_16 = 6
    var_17 = 12
    var_18 = module_0.ScalarToken(var_15, var_16, var_17, var_13)
    var_19 = {var_14: var_18}
    var_20 = {var_14: var_18}
    var_21 = 'key2'
    var_22 = '"key2": "value"'
    var_23 = module_0.ScalarToken(var_21, var_1, var_2, var_22)
    var_24 = {var_23: var_18}
    var_25 = 13
    var_26 = 'item'
    var_27 = module_0.ScalarToken(var_26, var_1, var_2, var_26)
    var_28 = [var_27]
    var_29 = module_0.ListToken(var_28, var_1, var_2, var_26)
    var_30 = module_0.ScalarToken(var_26, var_1, var_2, var_26)
    var_31 = [var_30]
    var_32 = module_0.ListToken(var_31, var_1, var_2, var_26)
    var_33 = 'other'
    var_34 = 4
    var_35 = module_0.ScalarToken(var_33, var_1, var_34, var_33)
    var_36 = [var_35]
    var_37 = module_0.ListToken(var_36, var_1, var_34, var_33)



# Parsed testcases at query #29
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 2
    var_3 = 'key: value'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'value'
    var_6 = 5
    var_7 = 9
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_3)
    var_9 = {var_4: var_8}
    var_10 = [var_0]
    var_11 = [var_0]
    var_12 = 'key1'
    var_13 = 3
    var_14 = 'key1: val1, key2: val2'
    var_15 = module_0.ScalarToken(var_12, var_1, var_13, var_14)
    var_16 = 'val1'
    var_17 = 7
    var_18 = 10
    var_19 = module_0.ScalarToken(var_16, var_17, var_18, var_14)
    var_20 = 'key2'
    var_21 = 13
    var_22 = 16
    var_23 = module_0.ScalarToken(var_20, var_21, var_22, var_14)
    var_24 = 'val2'
    var_25 = 20
    var_26 = 23
    var_27 = module_0.ScalarToken(var_24, var_25, var_26, var_14)
    var_28 = {var_15: var_19, var_23: var_27}
    var_29 = [var_12]
    var_30 = [var_20]
    var_31 = [var_12]
    var_32 = [var_20]



# Parsed testcases at query #30
#--------------------------


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 'test'
    var_2 = 0
    var_3 = 3
    var_4 = module_0.Token(var_1, var_2, var_3, var_0)
    var_5 = var_4.start
    var_6 = var_4.end
    var_7 = module_0.Token(var_1, var_2, var_3, var_0)
    var_8 = 'different'
    var_9 = module_0.Token(var_8, var_2, var_3, var_0)
    var_10 = repr(var_4)
    assert var_10 == "Token('test')"
    var_11 = var_4.value
    var_12 = 0
    var_13 = [var_12]
    var_14 = var_4.lookup(var_13)
    var_15 = 0
    var_16 = [var_15]
    var_17 = var_4.lookup_key(var_16)



