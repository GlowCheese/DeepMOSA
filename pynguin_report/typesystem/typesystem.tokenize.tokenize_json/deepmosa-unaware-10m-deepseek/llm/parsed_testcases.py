####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '12.34'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '-42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '1.23e4'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 0
    var_21 = '{"outer": {"inner": 42}}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '[]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '[1, 2, 3]'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = var_26.value
    var_30 = '{"array": [1, 2], "nested": {"bool": true}}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'array'
    var_33 = 1
    var_34 = 6
    var_35 = module_1.ScalarToken(var_32, var_33, var_34, var_30)
    var_36 = var_31.value[var_35]
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = '  {  "key"  :  "value"  }  '
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = b'"test"'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   \n  \t  '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{invalid}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '"unclosed'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": }'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{key: "value"}'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '{"key": "value",}'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '[1, 2,]'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '{\n  "key": "value"\n}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = len(var_61)
    var_64 = var_63 - var_33



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": 3}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'b'
    var_28 = 12
    var_29 = module_1.ScalarToken(var_27, var_28, var_28, var_23)
    var_30 = var_24.value[var_29]
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '  {  "key"  :  "value"  }  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = '{}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 0
    var_41 = '[]'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value
    var_44 = len(var_43)
    assert var_44 == 0
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   \n\t  '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{invalid}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '"unclosed string'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": }'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '3.14'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '-42'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '1e3'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = 'false'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = var_64.value
    var_66 = len(var_65)
    assert var_66 == 2
    var_67 = 'users'
    var_68 = 1
    var_69 = 5
    var_70 = module_1.ScalarToken(var_67, var_68, var_69, var_63)
    var_71 = var_64.value[var_70]
    var_72 = var_71.value
    var_73 = len(var_72)
    assert var_73 == 2



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'null'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '3.14'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '{"key": "value"}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 0
    var_22 = '{"array": [1, 2], "nested": {"key": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 'array'
    var_25 = 1
    var_26 = 6
    var_27 = module_1.ScalarToken(var_24, var_25, var_26, var_22)
    var_28 = var_23.value[var_27]
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '   \n\t  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{invalid}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": "value"'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"key": "value",}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = '1.23e-4'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '[]'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 0
    var_53 = '{}'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = var_54.value
    var_56 = len(var_55)
    assert var_56 == 0
    var_57 = '"line1\\nline2"'
    var_58 = module_0.tokenize_json(var_57)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '{"key": "value"}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 0
    var_18 = '{"nested": {"inner": [1, 2]}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'nested'
    var_21 = 1
    var_22 = 7
    var_23 = module_1.ScalarToken(var_20, var_21, var_22, var_18)
    var_24 = var_19.value[var_23]
    var_25 = 'inner'
    var_26 = 11
    var_27 = 16
    var_28 = module_1.ScalarToken(var_25, var_26, var_27, var_18)
    var_29 = var_24.value[var_28]
    var_30 = b'"test"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = ''
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '   \n\t  '
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{invalid}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '"unclosed string'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '[1, 2,'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": "value"'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{key: "value"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_47 = module_0.tokenize_json(var_46)
    var_48 = var_47.value
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = '123.456'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '-42'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '1e3'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{"a": 1, "b": 2}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = len(var_56)
    var_59 = var_58 - var_21



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"outer": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = b'"hello"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 0
    var_35 = '[]'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = '  {  "key"  :  "value"  }  '
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = ''
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '   '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{invalid}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"key": }'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '[1, 2,'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '3.14'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '1.23e-4'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = 'false'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '{"a": 1, "b": 2, "c": 3}'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = var_60.value
    var_62 = len(var_61)
    assert var_62 == 3



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '{"key": "value"}'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 0
    var_18 = '{"list": [1, 2], "nested": {"key": "value"}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'list'
    var_21 = 1
    var_22 = 5
    var_23 = module_1.ScalarToken(var_20, var_21, var_22, var_18)
    var_24 = var_19.value[var_23]
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '  {  "key"  :  "value"  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '   \n  \t  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{invalid}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '"unclosed string'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '3.14'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '-42'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '1e3'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = 'false'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 2



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'null'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '42'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '3.14'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '{"key": "value"}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 0
    var_22 = '{"array": [1, 2], "nested": {"key": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = b'"hello"'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   \n\t  '
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{invalid}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '"unclosed string'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[1, 2'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "value"'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{key: "value"}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '\n    {\n        "name": "John",\n        "age": 30,\n        "active": true,\n        "scores": [95.5, 87.0, 92.3],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value
    var_45 = len(var_44)
    assert var_45 == 5



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'"hello"'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = ''
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '   \n\t  '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '{invalid}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '"unclosed string'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": }'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '3.14'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '-42'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '1e10'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "coding"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 4
    var_47 = 'hobbies'
    var_48 = ''
    var_49 = module_1.ScalarToken(var_47, var_22, var_22, var_48)
    var_50 = var_44.value[var_49]
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 2



# Parsed testcases at query #9
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"a": [1, 2], "b": {"c": 3}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '  {  "key"  :  "value"  }  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = '{}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 0
    var_41 = '[]'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value
    var_44 = len(var_43)
    assert var_44 == 0
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{invalid}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": }'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '[1, 2,'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{"key": "value"'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '"unclosed string'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '{key: "value"}'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '[1, 2,]'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = '1.23e4'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = '-42'
    var_66 = module_0.tokenize_json(var_65)
    var_67 = '0'
    var_68 = module_0.tokenize_json(var_67)
    var_69 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_70 = module_0.tokenize_json(var_69)
    var_71 = var_70.value
    var_72 = len(var_71)
    assert var_72 == 4



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 0
    var_18 = var_14.value[var_17]
    var_19 = 1
    var_20 = var_14.value[var_19]
    var_21 = '{"key": "value"}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = '{"outer": {"inner": 42}}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = '[]'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 0
    var_41 = '  {  "key"  :  "value"  }  '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = var_42.value
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{invalid}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": }'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '[1, 2,'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '3.14'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '1.23e4'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '-42'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '"line1\\nline2"'
    var_62 = module_0.tokenize_json(var_61)



# Parsed testcases at query #11
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '{"key": "value"}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 0
    var_22 = '{"list": [1, 2], "nested": {"key": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'list'
    var_27 = var_23.value[var_26]
    var_28 = 'nested'
    var_29 = var_23.value[var_28]
    var_30 = b'"hello"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = ''
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '   \n\t  '
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{invalid}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '"unclosed string'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '[1, 2'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"key": "value"'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{key: "value"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"key": "value",}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_49 = module_0.tokenize_json(var_48)
    var_50 = var_49.value
    var_51 = len(var_50)
    assert var_51 == 4
    var_52 = 'hobbies'
    var_53 = var_49.value[var_52]
    var_54 = var_53.value
    var_55 = len(var_54)
    assert var_55 == 2



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'false'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'null'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '123.456'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '1.23e4'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '[1, 2, 3]'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = var_15.value
    var_19 = '[[1, 2], [3, 4]]'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = var_20.value
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = var_20.value
    var_24 = '{"key": "value"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = var_25.value
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = 0
    var_29 = '{"nested": {"inner": 42}}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '  {  "key"  :  "value"  }  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = '{}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = '[]'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 0
    var_43 = b'"hello"'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = ''
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '   '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{invalid}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": }'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": "value"'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{"name": "John", "age": 30}'
    var_56 = module_0.tokenize_json(var_55)



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'"hello"'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = '[]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 0
    var_35 = '  {  "key"  :  "value"  }  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '   '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{invalid}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"key": }'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"key": "value"'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '3.14'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '1.23e4'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_54 = module_0.tokenize_json(var_53)



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '{"key": "value"}'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 0
    var_22 = '{"list": [1, 2], "nested": {"key": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'list'
    var_27 = 1
    var_28 = 5
    var_29 = module_1.ScalarToken(var_26, var_27, var_28, var_22)
    var_30 = var_23.value[var_29]
    var_31 = 'nested'
    var_32 = 16
    var_33 = 22
    var_34 = module_1.ScalarToken(var_31, var_32, var_33, var_22)
    var_35 = var_23.value[var_34]
    var_36 = b'"hello"'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = ''
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '   \n\t  '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{invalid}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"key": }'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '"unclosed'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}], "count": 2}'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = var_49.value
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = '"line1\\nline2"'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '"café"'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '1.23e-4'
    var_57 = module_0.tokenize_json(var_56)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"key": "value"}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 0
    var_13 = '[1, 2, 3]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = var_14.value
    var_18 = '{"nested": {"inner": [1, 2]}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = b'"hello"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = ''
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '   \n\t  '
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{invalid}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": }'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '[1, 2'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '3.14'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '1.23e-4'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '\n    {\n        "name": "test",\n        "values": [1, 2, 3],\n        "nested": {\n            "flag": true,\n            "count": null\n        }\n    }\n    '
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = len(var_38)
    assert var_39 == 3



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'key'
    var_23 = var_19.value[var_22]
    var_24 = '{"a": [1, 2], "b": {"c": 3}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'a'
    var_27 = var_25.value[var_26]
    var_28 = 'b'
    var_29 = var_25.value[var_28]
    var_30 = b'"hello"'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '  {  "key"  :  "value"  }  '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = '[]'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = var_39.value
    var_41 = len(var_40)
    assert var_41 == 0
    var_42 = ''
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '   \n  \t  '
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{invalid}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"key": }'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '[1, 2'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '3.14'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '-42'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '1e3'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'false'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_61 = module_0.tokenize_json(var_60)
    var_62 = 'hobbies'
    var_63 = var_61.value[var_62]
    var_64 = var_63.value
    var_65 = len(var_64)
    assert var_65 == 2



# Parsed testcases at query #17
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
    var_8 = '{"outer": {"inner": 42}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'outer'
    var_11 = var_9.value[var_10]
    var_12 = 'inner'
    var_13 = 42
    var_14 = 18
    var_15 = 19
    var_16 = module_1.ScalarToken(var_13, var_14, var_15, var_8)
    var_17 = {var_12: var_16}
    var_18 = '[1, 2, 3]'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 3
    var_22 = 1
    var_23 = module_1.ScalarToken(var_22, var_22, var_22, var_18)
    var_24 = '"hello"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '42'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '3.14'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'true'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = 'false'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'null'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = b'{"test": "bytes"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'test'
    var_39 = 'bytes'
    var_40 = 9
    var_41 = 15
    var_42 = '{"test": "bytes"}'
    var_43 = module_1.ScalarToken(var_39, var_40, var_41, var_42)
    var_44 = {var_38: var_43}
    var_45 = '{}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[]'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '  {  "key"  :  "value"  }  '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = 16
    var_52 = 22
    var_53 = module_1.ScalarToken(var_3, var_51, var_52, var_49)
    var_54 = {var_2: var_53}
    var_55 = ''
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '   '
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '{invalid}'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '{"key": }'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = '{"key": "value"'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = '\n    {\n        "name": "test",\n        "numbers": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null\n        }\n    }\n    '
    var_66 = module_0.tokenize_json(var_65)
    var_67 = 'numbers'
    var_68 = var_66.value[var_67]
    var_69 = 'nested'
    var_70 = var_66.value[var_69]



# Parsed testcases at query #18
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"outer": {"inner": 42}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = b'"hello"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = ''
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '   \n\t  '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{invalid}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '"unclosed string'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"key": }'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{key: "value"}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '"line1\\nline2"'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '"café"'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '1.23e-4'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '-42'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '0'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '[]'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 0
    var_61 = '{}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = var_62.value
    var_64 = len(var_63)
    assert var_64 == 0



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'key'
    var_23 = var_19.value[var_22]
    var_24 = '{"nested": {"inner": 42}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'nested'
    var_27 = var_25.value[var_26]
    var_28 = '  {  "key"  :  "value"  }  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = '[]'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = var_35.value
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = b'"test"'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '3.14'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '-123.456e-10'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value
    var_45 = '1.23e+5'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = ''
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '   '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{invalid}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": }'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '[1, 2,'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '"unclosed'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '{key: "value"}'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '"line1\\nline2"'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = '"café"'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        },\n        "number": 42\n    }\n    '
    var_66 = module_0.tokenize_json(var_65)
    var_67 = 'array'
    var_68 = var_66.value[var_67]
    var_69 = var_66.value[var_26]



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '3.14'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 0
    var_19 = '{"outer": {"inner": 42}}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '[1, 2, 3]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 3
    var_27 = '{"array": [1, {"nested": true}], "string": "test"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = b'"hello"'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = ''
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '   \n\t  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{invalid}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '"unclosed string'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"key": }'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{key: "value"}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"a": 1 "b": 2}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[1 2]'
    var_48 = module_0.tokenize_json(var_47)



# Parsed testcases at query #21
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"outer": {"inner": 42}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = b'"hello"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = b'[1, 2, 3]'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 3
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '   \n\t  '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{invalid}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"key": }'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[1, 2,'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '"unclosed string'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": "value"'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '"line1\\nline2"'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '"quoted \\"string\\""'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '1.23e4'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '-1.23e-4'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '[]'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = var_62.value
    var_64 = len(var_63)
    assert var_64 == 0
    var_65 = '{}'
    var_66 = module_0.tokenize_json(var_65)
    var_67 = var_66.value
    var_68 = len(var_67)
    assert var_68 == 0
    var_69 = '[1, 2, 3]   '
    var_70 = module_0.tokenize_json(var_69)
    var_71 = var_70.value
    var_72 = len(var_71)
    assert var_72 == 3



# Parsed testcases at query #22
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": 3}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = b'"hello"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   \n\t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{invalid}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '"unclosed string'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key": }'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '3.14'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '1.23e-4'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 4
    var_47 = '{}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = var_48.value
    var_50 = len(var_49)
    assert var_50 == 0
    var_51 = '[]'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = var_52.value
    var_54 = len(var_53)
    assert var_54 == 0
    var_55 = '{ "key" : "value" }'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = var_56.value
    var_58 = len(var_57)
    assert var_58 == 1



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  { "key" : 123 }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'{"bytes": "input"}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n\t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"unclosed": "string}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": "value"'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '3.14'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '1.23e4'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '-42'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null\n        },\n        "string": "test"\n    }\n    '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = var_48.value
    var_50 = len(var_49)
    assert var_50 == 3



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": 3}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = b'"hello"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   \n\t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{invalid}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '"unclosed string'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '  "test"  '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '123.45'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '1.23e4'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = 'false'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = var_46.value
    var_48 = len(var_47)
    assert var_48 == 4
    var_49 = '[1, "two", true, null]'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 4



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "key"  :  "value"  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = b'"hello"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '"unclosed string'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": }'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '[1, 2, ]'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '\n    {\n        "string": "value",\n        "number": 123.45,\n        "boolean": true,\n        "null": null,\n        "array": [1, 2, 3],\n        "object": {"nested": "value"}\n    }\n    '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 6



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  { "key" : "value" }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = b'"hello"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n\t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '{"key": "value"'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{key: "value"}'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '3.14'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '1.23e-4'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = 'false'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_48 = module_0.tokenize_json(var_47)
    var_49 = var_48.value
    var_50 = len(var_49)
    assert var_50 == 2



# Parsed testcases at query #5
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'key'
    var_27 = var_23.value[var_26]
    var_28 = '{"outer": {"inner": 42}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'outer'
    var_31 = var_29.value[var_30]
    var_32 = var_29.value[var_30]
    var_33 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = 'array'
    var_36 = var_34.value[var_35]
    var_37 = 'nested'
    var_38 = var_34.value[var_37]
    var_39 = b'"hello"'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = ''
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '   \n  \t  '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{invalid}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '"unclosed string'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '[1, 2,'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"key": "value"'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{key: "value"}'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '"line1\\nline2"'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '"café"'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '[]'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = var_60.value
    var_62 = len(var_61)
    assert var_62 == 0
    var_63 = '{}'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = var_64.value
    var_66 = len(var_65)
    assert var_66 == 0
    var_67 = '1.23e-4'
    var_68 = module_0.tokenize_json(var_67)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'"hello"'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 0
    var_31 = '[]'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 0
    var_35 = '  {  "key"  :  "value"  }  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '   '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{invalid}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{"key": }'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[1, 2,'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '3.14'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '1.23e4'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = 'false'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_56 = module_0.tokenize_json(var_55)
    var_57 = var_56.value
    var_58 = len(var_57)
    assert var_58 == 2



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "key"  :  "value"  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = b'"hello"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n  \t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '"unclosed string'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": }'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{"a": 1, "b": [2, 3]}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = len(var_41)
    var_44 = 1
    var_45 = var_43 - var_44
    var_46 = 'b'
    var_47 = 7
    var_48 = 9
    var_49 = module_1.ScalarToken(var_46, var_47, var_48, var_41)
    var_50 = var_42.value[var_49]
    var_51 = '3.14'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '1.23e4'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '-42'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '\n    {\n        "name": "test",\n        "values": [1, 2, 3],\n        "nested": {\n            "flag": true,\n            "text": null\n        }\n    }\n    '
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 3



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  {  "key"  :  "value"  }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = var_26.value
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = b'"hello"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n\t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '"unclosed string'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": }'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '3.14'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '1.23e4'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = var_46.value
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = 'array'
    var_50 = 13
    var_51 = 17
    var_52 = module_1.ScalarToken(var_49, var_50, var_51, var_45)
    var_53 = var_46.value[var_52]
    var_54 = var_53.value
    var_55 = len(var_54)
    assert var_55 == 3



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"outer": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = b'"test"'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '  {  "key"  :  "value"  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = '{}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 0
    var_35 = '[]'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = ''
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '   \n  \t  '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{invalid}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '"unclosed string'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '[1, 2,'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"key": "value"'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{key: "value"}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '3.14'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '1.23e4'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '-42'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_60 = module_0.tokenize_json(var_59)
    var_61 = 'array'
    var_62 = ''
    var_63 = module_1.ScalarToken(var_61, var_22, var_22, var_62)
    var_64 = var_60.value[var_63]
    var_65 = 'nested'
    var_66 = module_1.ScalarToken(var_65, var_22, var_22, var_62)
    var_67 = var_60.value[var_66]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": 3}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '  {  "key"  :  "value"  }  '
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = b'{"key": "value"}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = '{}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = '[]'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 0
    var_43 = ''
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '   '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{invalid}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"key": }'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '[1, 2,'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '3.14'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '1.23e4'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '-42'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = 'false'
    var_60 = module_0.tokenize_json(var_59)
    var_61 = '\n    {\n        "name": "test",\n        "values": [1, 2, 3],\n        "nested": {\n            "inner": true,\n            "items": []\n        }\n    }\n    '
    var_62 = module_0.tokenize_json(var_61)
    var_63 = var_62.value
    var_64 = len(var_63)
    assert var_64 == 3



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'key'
    var_23 = var_19.value[var_22]
    var_24 = '{"outer": {"inner": 42}}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'outer'
    var_27 = var_25.value[var_26]
    var_28 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'array'
    var_31 = var_29.value[var_30]
    var_32 = 'nested'
    var_33 = var_29.value[var_32]
    var_34 = b'"hello"'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = b'[1, 2, 3]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = var_37.value
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = ''
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '   \n\t  '
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{invalid}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '"unclosed string'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '[1, 2,'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"key": "value"'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{key: "value"}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '3.14'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '1.23e4'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '-42'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = 'false'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = '[]'
    var_63 = module_0.tokenize_json(var_62)
    var_64 = var_63.value
    var_65 = len(var_64)
    assert var_65 == 0
    var_66 = '{}'
    var_67 = module_0.tokenize_json(var_66)
    var_68 = var_67.value
    var_69 = len(var_68)
    assert var_69 == 0
    var_70 = '[ 1 , 2 , 3 ]'
    var_71 = module_0.tokenize_json(var_70)
    var_72 = var_71.value
    var_73 = len(var_72)
    assert var_73 == 3
    var_74 = '{ "key" : "value" }'
    var_75 = module_0.tokenize_json(var_74)
    var_76 = '"line1\\nline2"'
    var_77 = module_0.tokenize_json(var_76)
    var_78 = '"café"'
    var_79 = module_0.tokenize_json(var_78)
    var_80 = b'"caf\xc3\xa9"'
    var_81 = module_0.tokenize_json(var_80)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '3.14'
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
    var_16 = 0
    var_17 = var_13.value[var_16]
    var_18 = 1
    var_19 = var_13.value[var_18]
    var_20 = 2
    var_21 = var_13.value[var_20]
    var_22 = '{"key": "value", "num": 42}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = '{"arr": [1, 2], "obj": {"nested": "value"}}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = b'"hello"'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = ''
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '   \n\t  '
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"key": value}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[1, 2, 3'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"key": "value"'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '"line1\\nline2"'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '"café"'
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
    var_52 = '{ "key" : "value" }'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = var_53.value
    var_55 = len(var_54)
    assert var_55 == 1



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"nested": {"inner": 42}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = '  { "key" : 123 }  '
    var_26 = module_0.tokenize_json(var_25)
    var_27 = b'"test"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   \n\t  '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{invalid}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"unclosed": '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '[1, 2, 3'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "hello"\n        }\n    }\n    '
    var_40 = module_0.tokenize_json(var_39)



# Parsed testcases at query #14
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"outer": {"inner": 42}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = b'"hello"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = ''
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '   \n  \t  '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"unclosed":'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '"unclosed string'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '12.34.56'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '"line1\\nline2"'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '"café"'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 0
    var_53 = '[]'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = var_54.value
    var_56 = len(var_55)
    assert var_56 == 0
    var_57 = '{ "key" : "value" }'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 1



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": 3}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = b'"hello"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '  {  "key"  :  "value"  }  '
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = '{}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value
    var_36 = len(var_35)
    assert var_36 == 0
    var_37 = '[]'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 0
    var_41 = ''
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '   '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '{invalid}'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"key": }'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '[1, 2,'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '3.14'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '1.23e4'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '-42'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = 'false'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_60 = module_0.tokenize_json(var_59)
    var_61 = var_60.value
    var_62 = len(var_61)
    assert var_62 == 4



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = var_9.value
    var_13 = '[[1, 2], [3, 4]]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = var_14.value
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14.value
    var_18 = '{"key": "value"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = var_19.value
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 0
    var_23 = '{"a": [1, 2], "b": {"c": true}}'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = b'"hello"'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = ''
    var_30 = module_0.tokenize_json(var_29)
    var_31 = '   '
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '{invalid}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{"key": }'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '  {  "key"  :  "value"  }  '
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '3.14'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '-42'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '1e10'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = 'false'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '\n    {\n        "array": [1, 2, 3],\n        "nested": {\n            "bool": true,\n            "null": null,\n            "string": "test"\n        }\n    }\n    '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = var_50.value
    var_52 = len(var_51)
    assert var_52 == 2



# Parsed testcases at query #17
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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"outer": {"inner": 42}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = '{"array": [1, 2, 3], "nested": {"bool": true}}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = var_30.value
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = b'"hello"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 0
    var_39 = '[]'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = var_40.value
    var_42 = len(var_41)
    assert var_42 == 0
    var_43 = '  {  "key"  :  "value"  }  '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = var_44.value
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = ''
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '   '
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{invalid}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": }'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '[1, 2,'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '"unclosed string'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = '{"key": "value"'
    var_60 = module_0.tokenize_json(var_59)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '42'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '3.14'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'true'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": "value"}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 0
    var_19 = '{"outer": {"inner": 42}}'
    var_20 = module_0.tokenize_json(var_19)
    var_21 = '[]'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = '[1, 2, 3]'
    var_24 = module_0.tokenize_json(var_23)
    var_25 = var_24.value
    var_26 = len(var_25)
    assert var_26 == 3
    var_27 = '{"array": [1, 2], "nested": {"bool": true}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'"test"'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = ''
    var_32 = module_0.tokenize_json(var_31)
    var_33 = '   \n\t  '
    var_34 = module_0.tokenize_json(var_33)
    var_35 = '{invalid}'
    var_36 = module_0.tokenize_json(var_35)
    var_37 = '"unclosed string'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = '{"key": }'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = '{key: "value"}'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '{"key": "value",}'
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '[1, 2, 3,]'
    var_46 = module_0.tokenize_json(var_45)



# Parsed testcases at query #19
#--------------------------


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '123'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = 'true'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'null'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '12.34'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '-42'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '1.23e4'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"key": "value"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 0
    var_21 = '{"outer": {"inner": 42}}'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = var_22.value
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = '[]'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '[1, "two", true]'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 3
    var_31 = '{"array": [1, 2, 3], "nested": {"bool": false}}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '  {  "key"  :  "value"  }  '
    var_36 = module_0.tokenize_json(var_35)
    var_37 = var_36.value
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = b'"hello"'
    var_40 = module_0.tokenize_json(var_39)
    var_41 = ''
    var_42 = module_0.tokenize_json(var_41)
    var_43 = '   \n  \t  '
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '"hello'
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{"key": "value"'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{key: "value"}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '[1 2]'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '"line1\\nline2"'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '"café"'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '{"a": 1, "b": 2, "c": 3}'
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 3



# Parsed testcases at query #20
#--------------------------


import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

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
    var_12 = '[1, 2, 3]'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = var_13.value
    var_15 = len(var_14)
    assert var_15 == 3
    var_16 = var_13.value
    var_17 = '[[1, 2], [3, 4]]'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = var_18.value
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_18.value
    var_22 = '{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 0
    var_27 = '{"a": [1, 2], "b": {"c": 3}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = var_28.value
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 'a'
    var_32 = 1
    var_33 = 3
    var_34 = module_1.ScalarToken(var_31, var_32, var_33, var_27)
    var_35 = var_28.value[var_34]
    var_36 = 'b'
    var_37 = 13
    var_38 = 15
    var_39 = module_1.ScalarToken(var_36, var_37, var_38, var_27)
    var_40 = var_28.value[var_39]
    var_41 = b'"hello"'
    var_42 = module_0.tokenize_json(var_41)
    var_43 = ''
    var_44 = module_0.tokenize_json(var_43)
    var_45 = '   \n  \t  '
    var_46 = module_0.tokenize_json(var_45)
    var_47 = '{invalid}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '"unclosed string'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '[1, 2,'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = '{"key": "value"'
    var_54 = module_0.tokenize_json(var_53)
    var_55 = '{"key": "value",}'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = '\n    {\n        "name": "John",\n        "age": 30,\n        "hobbies": ["reading", "swimming"],\n        "address": {\n            "street": "123 Main St",\n            "city": "Boston"\n        }\n    }\n    '
    var_58 = module_0.tokenize_json(var_57)
    var_59 = var_58.value
    var_60 = len(var_59)
    assert var_60 == 4
    var_61 = '{"a": 1}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = module_1.ScalarToken(var_31, var_32, var_33, var_61)
    var_64 = var_62.value[var_63]



