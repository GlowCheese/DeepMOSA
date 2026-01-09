####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = '"Hello, World!"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'Test case 3 passed'
    var_11 = print(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 4 passed'
    var_15 = print(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 5 passed'
    var_19 = print(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 6 passed'
    var_23 = print(var_22)
    var_24 = '{}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 7 passed'
    var_27 = print(var_26)
    var_28 = '[]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 8 passed'
    var_31 = print(var_30)
    var_32 = '{"person": {"name": "Alice", "age": 25}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 9 passed'
    var_35 = print(var_34)
    var_36 = '  {  "key"  :  "value"  }  '
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'Test case 10 passed'
    var_39 = print(var_38)
    var_40 = '{"name": "John"'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'Test case 11 failed: Expected ParseError'
    var_43 = print(var_42)
    var_44 = '{"name": "John",}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'Test case 12 failed: Expected ParseError'
    var_47 = print(var_46)
    var_48 = ''
    var_49 = module_0.tokenize_json(var_48)
    var_50 = 'Test case 13 failed: Expected ParseError'
    var_51 = print(var_50)
    var_52 = b'{"key": "value"}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'Test case 14 passed'
    var_55 = print(var_54)
    var_56 = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'users'
    var_59 = 'total'
    var_60 = 'id'
    var_61 = 'name'
    var_62 = 1
    var_63 = 'Alice'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = 2
    var_66 = 'Bob'
    var_67 = {var_60: var_65, var_61: var_66}
    var_68 = [var_64, var_67]
    var_69 = {var_58: var_68, var_59: var_65}
    var_70 = 'Test case 15 passed'
    var_71 = print(var_70)
    var_72 = 'All test cases passed!'
    var_73 = print(var_72)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = b'{"name": "John", "age": 30}'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_3
    var_9 = ''
    var_10 = module_0.tokenize_json(var_9)
    var_11 = '{"name": "John", "age": 30'
    var_12 = module_0.tokenize_json(var_11)
    var_13 = '[1, 2, 3]'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_3
    var_17 = 'null'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = len(var_17)
    var_20 = var_19 - var_3
    var_21 = 'true'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = len(var_21)
    var_24 = var_23 - var_3
    var_25 = 'false'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = len(var_25)
    var_28 = var_27 - var_3
    var_29 = '42'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = len(var_29)
    var_32 = var_31 - var_3
    var_33 = '"Hello, World!"'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = len(var_33)
    var_36 = var_35 - var_3
    var_37 = 'All tests passed!'
    var_38 = print(var_37)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John\\"Doe"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'true'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'null'
    var_17 = module_0.tokenize_json(var_16)



# Parsed testcases at query #5
#--------------------------



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
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_15 = {var_6: var_14}
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed.'
    var_9 = print(var_8)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"person": {"name": "John", "age": 30}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"active": true, "value": null}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = '[1, 2, 3]'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '"Hello, World!"'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '42'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'true'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = ''
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"name": "John", "age": 30'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = b'{"name": "John", "age": 30}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = '[1, 2, 3]'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_3
    var_9 = '"Hello, World!"'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_3
    var_13 = '42'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_3
    var_17 = 'true'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = len(var_17)
    var_20 = var_19 - var_3
    var_21 = 'null'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = len(var_21)
    var_24 = var_23 - var_3
    var_25 = ''
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'{"name": "John", "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = len(var_29)
    var_32 = var_31 - var_3
    var_33 = 'utf-8'
    var_34 = 'All tests passed!'
    var_35 = print(var_34)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = '[1, 2, 3]'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_3
    var_9 = '"Hello, World!"'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_3
    var_13 = '42'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_3
    var_17 = 'true'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = len(var_17)
    var_20 = var_19 - var_3
    var_21 = 'null'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = len(var_21)
    var_24 = var_23 - var_3
    var_25 = ''
    var_26 = module_0.tokenize_json(var_25)
    var_27 = '{"name": "John", "age": 30'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = b'{"name": "John", "age": 30}'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = len(var_29)
    var_32 = var_31 - var_3
    var_33 = '{"person": {"name": "John", "age": 30}}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = 'person'
    var_36 = var_34.value[var_35]
    var_37 = '[{"name": "John"}, {"name": "Jane"}]'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = var_38.value
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.value
    var_42 = '  { "name" : "John" }  '
    var_43 = module_0.tokenize_json(var_42)
    var_44 = len(var_42)
    var_45 = 3
    var_46 = var_44 - var_45
    var_47 = '{"message": "Hello\\nWorld"}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = '{"name": "Jöhn"}'
    var_50 = module_0.tokenize_json(var_49)
    var_51 = '{"value": 1.23e-4}'
    var_52 = module_0.tokenize_json(var_51)
    var_53 = 'All tests passed!'
    var_54 = print(var_53)



# Parsed testcases at query #11
#--------------------------



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
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_15 = {var_6: var_14}
    var_16 = '{"key": 123}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 123
    var_19 = 9
    var_20 = module_1.ScalarToken(var_18, var_8, var_19, var_16)
    var_21 = {var_6: var_20}
    var_22 = '{"key": true}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = True
    var_25 = 10
    var_26 = module_1.ScalarToken(var_24, var_8, var_25, var_22)
    var_27 = {var_6: var_26}
    var_28 = '{"key": null}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = None
    var_31 = module_1.ScalarToken(var_30, var_8, var_25, var_28)
    var_32 = {var_6: var_31}
    var_33 = '{"key": [1, 2, 3]}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = var_34.value[var_6]
    var_36 = module_1.ScalarToken(var_24, var_19, var_19, var_33)
    var_37 = 2
    var_38 = 12
    var_39 = module_1.ScalarToken(var_37, var_38, var_38, var_33)
    var_40 = 3
    var_41 = 15
    var_42 = module_1.ScalarToken(var_40, var_41, var_41, var_33)
    var_43 = [var_36, var_39, var_42]
    var_44 = '{"key": {"nested": "value"}}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = var_45.value[var_6]
    var_47 = 'nested'
    var_48 = 18
    var_49 = 24
    var_50 = module_1.ScalarToken(var_7, var_48, var_49, var_44)
    var_51 = {var_47: var_50}
    var_52 = 'All tests passed!'
    var_53 = print(var_52)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = '"Hello, World!"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'Test case 3 passed'
    var_11 = print(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 4 passed'
    var_15 = print(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 5 passed'
    var_19 = print(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 6 passed'
    var_23 = print(var_22)
    var_24 = '{}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 7 passed'
    var_27 = print(var_26)
    var_28 = '[]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 8 passed'
    var_31 = print(var_30)
    var_32 = '{"person": {"name": "Alice", "age": 25}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 9 passed'
    var_35 = print(var_34)
    var_36 = '[[1, 2], [3, 4]]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'Test case 10 passed'
    var_39 = print(var_38)
    var_40 = '  { "key" : "value" }  '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'Test case 11 passed'
    var_43 = print(var_42)
    var_44 = '{"message": "Hello\\nWorld"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'Test case 12 passed'
    var_47 = print(var_46)
    var_48 = '{"emoji": "😀"}'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = 'Test case 13 passed'
    var_51 = print(var_50)
    var_52 = '{"number": 1.23e-4}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'Test case 14 passed'
    var_55 = print(var_54)
    var_56 = '{"temperature": -10}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'Test case 15 passed'
    var_59 = print(var_58)
    var_60 = '{"zero": 0}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = 'Test case 16 passed'
    var_63 = print(var_62)
    var_64 = '{"empty": ""}'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = 'Test case 17 passed'
    var_67 = print(var_66)
    var_68 = '{"key-with-dash": "value"}'
    var_69 = module_0.tokenize_json(var_68)
    var_70 = 'Test case 18 passed'
    var_71 = print(var_70)
    var_72 = '{"a": {"b": {"c": {"d": "value"}}}}'
    var_73 = module_0.tokenize_json(var_72)
    var_74 = 'Test case 19 passed'
    var_75 = print(var_74)
    var_76 = '[1, "two", true, null]'
    var_77 = module_0.tokenize_json(var_76)
    var_78 = 'Test case 20 passed'
    var_79 = print(var_78)
    var_80 = '{"a": 1,}'
    var_81 = module_0.tokenize_json(var_80)
    var_82 = 'Test case 21 failed: Expected ParseError'
    var_83 = print(var_82)
    var_84 = '{"a": 1'
    var_85 = module_0.tokenize_json(var_84)
    var_86 = 'Test case 22 failed: Expected ParseError'
    var_87 = print(var_86)
    var_88 = '{"invalid": "\\x"}'
    var_89 = module_0.tokenize_json(var_88)
    var_90 = 'Test case 23 failed: Expected ParseError'
    var_91 = print(var_90)
    var_92 = '{"number": 123.}'
    var_93 = module_0.tokenize_json(var_92)
    var_94 = 'Test case 24 failed: Expected ParseError'
    var_95 = print(var_94)
    var_96 = '{"key": "first", "key": "second"}'
    var_97 = module_0.tokenize_json(var_96)
    var_98 = 'Test case 25 passed'
    var_99 = print(var_98)
    var_100 = '{"big": 12345678901234567890}'
    var_101 = module_0.tokenize_json(var_100)
    var_102 = 'Test case 26 passed'
    var_103 = print(var_102)
    var_104 = '{"pi": 3.14159}'
    var_105 = module_0.tokenize_json(var_104)
    var_106 = 'Test case 27 passed'
    var_107 = print(var_106)
    var_108 = '{"large": 1e6}'
    var_109 = module_0.tokenize_json(var_108)
    var_110 = 'Test case 28 passed'
    var_111 = print(var_110)
    var_112 = '{"small": 1e-6}'
    var_113 = module_0.tokenize_json(var_112)
    var_114 = 'Test case 29 passed'
    var_115 = print(var_114)
    var_116 = '{"positive": 1e+6}'
    var_117 = module_0.tokenize_json(var_116)
    var_118 = 'Test case 30 passed'
    var_119 = print(var_118)
    var_120 = '\n\t\r {"key":\n\t\r "value"\n\t\r }\n\t\r '
    var_121 = module_0.tokenize_json(var_120)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #14
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = module_1.ScalarToken(var_5, var_6, var_7, var_20)
    var_25 = {var_4: var_24}
    var_26 = '   '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"key": [1, 2, {"nested": "value"}]}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = var_29.value[var_4]
    var_31 = 'nested'
    var_32 = 30
    var_33 = 36
    var_34 = module_1.ScalarToken(var_5, var_32, var_33, var_28)
    var_35 = {var_31: var_34}
    var_36 = '{"key": "value with \\"quotes\\""}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"int": 42, "float": 3.14, "negative": -10}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"true": true, "false": false, "null": null}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'All tests passed!'
    var_43 = print(var_42)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"person": {"name": "John", "age": 30}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"numbers": [1, 2, 3]}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"flag": true, "value": null}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"int": 42, "float": 3.14, "exp": 1e5}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"text": "Hello, \\"world\\"!"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'All test cases passed!'
    var_21 = print(var_20)



# Parsed testcases at query #16
#--------------------------



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
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = '"Hello, World!"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'Test case 3 passed'
    var_11 = print(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 4 passed'
    var_15 = print(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 5 passed'
    var_19 = print(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 6 passed'
    var_23 = print(var_22)
    var_24 = '{"name": "John"'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 7 failed: Expected ParseError'
    var_27 = print(var_26)
    var_28 = ''
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 8 failed: Expected ParseError'
    var_31 = print(var_30)
    var_32 = b'{"key": "value"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 9 passed'
    var_35 = print(var_34)
    var_36 = '{"person": {"name": "Alice", "age": 25}}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'Test case 10 passed'
    var_39 = print(var_38)
    var_40 = '  {  "key"  :  "value"  }  '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'Test case 11 passed'
    var_43 = print(var_42)
    var_44 = '{"message": "Hello\\nWorld"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'Test case 12 passed'
    var_47 = print(var_46)
    var_48 = '3.14'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = 'Test case 13 passed'
    var_51 = print(var_50)
    var_52 = '1.23e-4'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'Test case 14 passed'
    var_55 = print(var_54)
    var_56 = '[1, "two", true, null]'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'Test case 15 passed'
    var_59 = print(var_58)
    var_60 = 'All test cases passed!'
    var_61 = print(var_60)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 13
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{invalid}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #19
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = module_1.ScalarToken(var_5, var_6, var_7, var_20)
    var_25 = {var_4: var_24}
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = '"Hello, World!"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'Test case 3 passed'
    var_11 = print(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 4 passed'
    var_15 = print(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 5 passed'
    var_19 = print(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 6 passed'
    var_23 = print(var_22)
    var_24 = ''
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 7 failed: Expected ParseError'
    var_27 = print(var_26)
    var_28 = '{"name": "John", "age": 30'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 8 failed: Expected ParseError'
    var_31 = print(var_30)
    var_32 = b'{"name": "John", "age": 30}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 9 passed'
    var_35 = print(var_34)
    var_36 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'Test case 10 passed'
    var_39 = print(var_38)
    var_40 = '  { "name" : "John" , "age" : 30 }  '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'Test case 11 passed'
    var_43 = print(var_42)
    var_44 = '{"message": "Hello\\nWorld"}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'Test case 12 passed'
    var_47 = print(var_46)
    var_48 = '{"name": "Jöhn"}'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = 'Test case 13 passed'
    var_51 = print(var_50)
    var_52 = '{"pi": 3.14159}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'Test case 14 passed'
    var_55 = print(var_54)
    var_56 = '{"number": 1.23e4}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'Test case 15 passed'
    var_59 = print(var_58)
    var_60 = '{"temperature": -10}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = 'Test case 16 passed'
    var_63 = print(var_62)
    var_64 = '{}'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = 'Test case 17 passed'
    var_67 = print(var_66)
    var_68 = '[]'
    var_69 = module_0.tokenize_json(var_68)
    var_70 = 'Test case 18 passed'
    var_71 = print(var_70)
    var_72 = '[1, "two", true, null]'
    var_73 = module_0.tokenize_json(var_72)
    var_74 = 'Test case 19 passed'
    var_75 = print(var_74)
    var_76 = '[[1, 2], [3, 4]]'
    var_77 = module_0.tokenize_json(var_76)
    var_78 = 'Test case 20 passed'
    var_79 = print(var_78)
    var_80 = 'All test cases passed!'
    var_81 = print(var_80)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"person": {"name": "John", "age": 30}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"names": ["John", "Jane"]}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"active": true, "value": null}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"integer": 42, "float": 3.14}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #23
#--------------------------



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
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_15 = {var_6: var_14}
    var_16 = '123'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'true'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '[1, 2, 3]'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = var_23.value
    var_27 = '{"nested": {"key": "value"}}'
    var_28 = module_0.tokenize_json(var_27)
    var_29 = 'nested'
    var_30 = var_28.value[var_29]
    var_31 = 20
    var_32 = 26
    var_33 = module_1.ScalarToken(var_7, var_31, var_32, var_27)
    var_34 = {var_6: var_33}
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = '[1, 2, 3]'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = '"Hello, World!"'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'Test case 3 passed'
    var_11 = print(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 4 passed'
    var_15 = print(var_14)
    var_16 = 'true'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 5 passed'
    var_19 = print(var_18)
    var_20 = 'null'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 6 passed'
    var_23 = print(var_22)
    var_24 = '{}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 7 passed'
    var_27 = print(var_26)
    var_28 = '[]'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 8 passed'
    var_31 = print(var_30)
    var_32 = '{"person": {"name": "Alice", "age": 25}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 9 passed'
    var_35 = print(var_34)
    var_36 = '[[1, 2], [3, 4]]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = 'Test case 10 passed'
    var_39 = print(var_38)
    var_40 = '  {  "key"  :  "value"  }  '
    var_41 = module_0.tokenize_json(var_40)
    var_42 = 'Test case 11 passed'
    var_43 = print(var_42)
    var_44 = '"Line 1\\nLine 2"'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'Test case 12 passed'
    var_47 = print(var_46)
    var_48 = '"Hello, 世界"'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = 'Test case 13 passed'
    var_51 = print(var_50)
    var_52 = '1.23e-4'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'Test case 14 passed'
    var_55 = print(var_54)
    var_56 = '-42'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = 'Test case 15 passed'
    var_59 = print(var_58)
    var_60 = '3.14159'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = 'Test case 16 passed'
    var_63 = print(var_62)
    var_64 = '""'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = 'Test case 17 passed'
    var_67 = print(var_66)
    var_68 = '{"key-with-dash": "value"}'
    var_69 = module_0.tokenize_json(var_68)
    var_70 = 'Test case 18 passed'
    var_71 = print(var_70)
    var_72 = '{"a": 1, "b": 2, "c": 3}'
    var_73 = module_0.tokenize_json(var_72)
    var_74 = 'Test case 19 passed'
    var_75 = print(var_74)
    var_76 = '[1, "two", true, null]'
    var_77 = module_0.tokenize_json(var_76)
    var_78 = 'Test case 20 passed'
    var_79 = print(var_78)
    var_80 = '{"a": {"b": {"c": {"d": "value"}}}}'
    var_81 = module_0.tokenize_json(var_80)
    var_82 = 'Test case 21 passed'
    var_83 = print(var_82)
    var_84 = '[{"id": 1}, {"id": 2}, {"id": 3}]'
    var_85 = module_0.tokenize_json(var_84)
    var_86 = 'Test case 22 passed'
    var_87 = print(var_86)
    var_88 = '{"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}'
    var_89 = module_0.tokenize_json(var_88)
    var_90 = 'Test case 23 passed'
    var_91 = print(var_90)
    var_92 = '12345678901234567890'
    var_93 = module_0.tokenize_json(var_92)
    var_94 = 'Test case 24 passed'
    var_95 = print(var_94)
    var_96 = '0'
    var_97 = module_0.tokenize_json(var_96)
    var_98 = 'Test case 25 passed'
    var_99 = print(var_98)
    var_100 = '-0'
    var_101 = module_0.tokenize_json(var_100)
    var_102 = 'Test case 26 passed'
    var_103 = print(var_102)
    var_104 = '1e3'
    var_105 = module_0.tokenize_json(var_104)
    var_106 = 'Test case 27 passed'
    var_107 = print(var_106)
    var_108 = '1e-3'
    var_109 = module_0.tokenize_json(var_108)
    var_110 = 'Test case 28 passed'
    var_111 = print(var_110)
    var_112 = '1e+3'
    var_113 = module_0.tokenize_json(var_112)
    var_114 = 'Test case 29 passed'
    var_115 = print(var_114)
    var_116 = '1.23e2'
    var_117 = module_0.tokenize_json(var_116)
    var_118 = 'Test case 30 passed'
    var_119 = print(var_118)
    var_120 = '1.'
    var_121 = module_0.tokenize_json(var_120)
    var_122 = 'Test case 31 failed: Should have raised ParseError'
    var_123 = print(var_122)
    var_124 = '.1'



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = len(var_0)
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = '[1, 2, 3]'
    var_6 = module_0.tokenize_json(var_5)
    var_7 = len(var_5)
    var_8 = var_7 - var_3
    var_9 = '"Hello, World!"'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = len(var_9)
    var_12 = var_11 - var_3
    var_13 = '42'
    var_14 = module_0.tokenize_json(var_13)
    var_15 = len(var_13)
    var_16 = var_15 - var_3
    var_17 = 'true'
    var_18 = module_0.tokenize_json(var_17)
    var_19 = len(var_17)
    var_20 = var_19 - var_3
    var_21 = 'null'
    var_22 = module_0.tokenize_json(var_21)
    var_23 = len(var_21)
    var_24 = var_23 - var_3
    var_25 = '{}'
    var_26 = module_0.tokenize_json(var_25)
    var_27 = len(var_25)
    var_28 = var_27 - var_3
    var_29 = '[]'
    var_30 = module_0.tokenize_json(var_29)
    var_31 = len(var_29)
    var_32 = var_31 - var_3
    var_33 = '{"person": {"name": "Alice", "age": 25}}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = len(var_33)
    var_36 = var_35 - var_3
    var_37 = '[[1, 2], [3, 4]]'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = len(var_37)
    var_40 = var_39 - var_3
    var_41 = '  { "key" : "value" }  '
    var_42 = module_0.tokenize_json(var_41)
    var_43 = len(var_41)
    var_44 = 3
    var_45 = var_43 - var_44
    var_46 = '"Hello\\nWorld"'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = len(var_46)
    var_49 = var_48 - var_3
    var_50 = '"Hello A"'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = len(var_50)
    var_53 = var_52 - var_3
    var_54 = '1.23e-4'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = len(var_54)
    var_57 = var_56 - var_3
    var_58 = '-42'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = len(var_58)
    var_61 = var_60 - var_3
    var_62 = '3.14'
    var_63 = module_0.tokenize_json(var_62)
    var_64 = len(var_62)
    var_65 = var_64 - var_3
    var_66 = '""'
    var_67 = module_0.tokenize_json(var_66)
    var_68 = len(var_66)
    var_69 = var_68 - var_3
    var_70 = '{"key-with-dash": "value"}'
    var_71 = module_0.tokenize_json(var_70)
    var_72 = len(var_70)
    var_73 = var_72 - var_3
    var_74 = '{"array": [1, 2, {"nested": "object"}]}'
    var_75 = module_0.tokenize_json(var_74)
    var_76 = len(var_74)
    var_77 = var_76 - var_3
    var_78 = '12345678901234567890'
    var_79 = module_0.tokenize_json(var_78)
    var_80 = len(var_78)
    var_81 = var_80 - var_3
    var_82 = '-12345678901234567890'
    var_83 = module_0.tokenize_json(var_82)
    var_84 = len(var_82)
    var_85 = var_84 - var_3
    var_86 = '1.2345678901234567e+308'
    var_87 = module_0.tokenize_json(var_86)
    var_88 = len(var_86)
    var_89 = var_88 - var_3
    var_90 = '-1.2345678901234567e+308'
    var_91 = module_0.tokenize_json(var_90)
    var_92 = len(var_90)
    var_93 = var_92 - var_3
    var_94 = '"He said, \\"Hello\\""'
    var_95 = module_0.tokenize_json(var_94)
    var_96 = len(var_94)
    var_97 = var_96 - var_3
    var_98 = '"C:\\\\path\\\\to\\\\file"'
    var_99 = module_0.tokenize_json(var_98)
    var_100 = len(var_98)
    var_101 = var_100 - var_3
    var_102 = '[1, "two", true, null, {"key": "value"}]'
    var_103 = module_0.tokenize_json(var_102)
    var_104 = len(var_102)
    var_105 = var_104 - var_3
    var_106 = '[{}]'
    var_107 = module_0.tokenize_json(var_106)
    var_108 = len(var_106)
    var_109 = var_108 - var_3
    var_110 = '{"empty": []}'
    var_111 = module_0.tokenize_json(var_110)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 13
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"key": "value"'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": {"nested": "value"}}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = var_15.value[var_2]
    var_17 = 'nested'
    var_18 = 17
    var_19 = 23
    var_20 = module_1.ScalarToken(var_3, var_18, var_19, var_14)
    var_21 = {var_17: var_20}
    var_22 = '[1, 2, 3]'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 1
    var_25 = module_1.ScalarToken(var_24, var_24, var_24, var_22)
    var_26 = 2
    var_27 = 4
    var_28 = module_1.ScalarToken(var_26, var_27, var_27, var_22)
    var_29 = 3
    var_30 = module_1.ScalarToken(var_29, var_4, var_4, var_22)
    var_31 = [var_25, var_28, var_30]
    var_32 = '{"true": true, "false": false}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'true'
    var_35 = 'false'
    var_36 = True
    var_37 = 9
    var_38 = 12
    var_39 = module_1.ScalarToken(var_36, var_37, var_38, var_32)
    var_40 = False
    var_41 = 27
    var_42 = module_1.ScalarToken(var_40, var_19, var_41, var_32)
    var_43 = {var_34: var_39, var_35: var_42}
    var_44 = '{"null": null}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = 'null'
    var_47 = None
    var_48 = module_1.ScalarToken(var_47, var_37, var_38, var_44)
    var_49 = {var_46: var_48}
    var_50 = '{"int": 42, "float": 3.14}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = 'int'
    var_53 = 'float'
    var_54 = 42
    var_55 = 8
    var_56 = module_1.ScalarToken(var_54, var_55, var_37, var_50)
    var_57 = 3.14
    var_58 = 20
    var_59 = module_1.ScalarToken(var_57, var_58, var_19, var_50)
    var_60 = {var_52: var_56, var_53: var_59}
    var_61 = '  {  "key"  :  "value"  }  '
    var_62 = module_0.tokenize_json(var_61)
    var_63 = 15
    var_64 = 21
    var_65 = module_1.ScalarToken(var_3, var_63, var_64, var_61)
    var_66 = {var_2: var_65}
    var_67 = 'All tests passed!'
    var_68 = print(var_67)



# Parsed testcases at query #3
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = module_1.ScalarToken(var_5, var_6, var_7, var_20)
    var_25 = {var_4: var_24}
    var_26 = '  { "key" : "value" }  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 11
    var_29 = 17
    var_30 = module_1.ScalarToken(var_5, var_28, var_29, var_26)
    var_31 = {var_4: var_30}
    var_32 = '{"key": [1, 2, {"nested": "value"}]}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = var_33.value[var_4]
    var_35 = var_34.value[var_14]
    var_36 = 'nested'
    var_37 = 28
    var_38 = 34
    var_39 = module_1.ScalarToken(var_5, var_37, var_38, var_32)
    var_40 = {var_36: var_39}
    var_41 = 'All tests passed!'
    var_42 = print(var_41)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30, "city": "New York"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30, "city": "New York"}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30, "city": "New York"'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3, 4, 5]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 'null'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 'true'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '42'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '3.14'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '"Hello, World!"'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"person": {"name": "John", "age": 30}, "city": "New York"}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"numbers": [1, 2, 3], "letters": ["a", "b", "c"]}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '[]'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '  {  "name"  :  "John"  ,  "age"  :  30  }  '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '{"message": "Hello\\nWorld"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{"name": "Jöhn", "city": "München"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{"value": 1.23e4}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '{"temperature": -10}'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"int": 42, "float": 3.14, "bool": true, "null": null, "string": "hello", "array": [1, 2, 3], "object": {"key": "value"}}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"large": 12345678901234567890}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"pi": 3.141592653589793}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"quote": "He said, \\"Hello\\""}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"path": "C:\\\\Windows\\\\System32"}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"text": "Hello\\tWorld"}'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"text": "Hello\\rWorld"}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"text": "Hello\\fWorld"}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{"text": "Hello\\bWorld"}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{"star": "\\u2605"}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '{"emoji": "\\uD83D\\uDE00"}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = '{"name": ""}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = '{"zero": 0}'
    var_63 = module_0.tokenize_json(var_62)



# Parsed testcases at query #6
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"nested": {"inner": [1, 2]}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = 'nested'
    var_25 = var_23.value[var_24]
    var_26 = 'inner'
    var_27 = var_25.value[var_26]
    var_28 = 23
    var_29 = module_1.ScalarToken(var_12, var_28, var_28, var_22)
    var_30 = 26
    var_31 = module_1.ScalarToken(var_14, var_30, var_30, var_22)
    var_32 = [var_29, var_31]
    var_33 = '{"bool": true, "null": null}'
    var_34 = module_0.tokenize_json(var_33)
    var_35 = 'bool'
    var_36 = 'null'
    var_37 = True
    var_38 = 9
    var_39 = 12
    var_40 = module_1.ScalarToken(var_37, var_38, var_39, var_33)
    var_41 = None
    var_42 = 22
    var_43 = 25
    var_44 = module_1.ScalarToken(var_41, var_42, var_43, var_33)
    var_45 = {var_35: var_40, var_36: var_44}
    var_46 = '{"int": 42, "float": 3.14}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = 'int'
    var_49 = 'float'
    var_50 = 42
    var_51 = 8
    var_52 = module_1.ScalarToken(var_50, var_51, var_38, var_46)
    var_53 = 3.14
    var_54 = 20
    var_55 = 24
    var_56 = module_1.ScalarToken(var_53, var_54, var_55, var_46)
    var_57 = {var_48: var_52, var_49: var_56}
    var_58 = b'{"key": "value"}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = module_1.ScalarToken(var_5, var_6, var_7, var_20)
    var_61 = {var_4: var_60}
    var_62 = 'All tests passed!'
    var_63 = print(var_62)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '[1, 2, 3]'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"message": "Hello, \\"world\\"!"}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '42'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'true'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 'null'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"person": {"name": "John", "age": 30}}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = '{"numbers": [1, 2, [3, 4]]}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"text": "Line 1\\nLine 2"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = '{"text": "Hello, 世界"}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = '{"value": 1.23e-4}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = '{"name": "John"}   '
    var_29 = module_0.tokenize_json(var_28)
    var_30 = '   {"name": "John"}'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = '{\n\t"name": "John",\n\t"age": 30\n}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = '{}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = '[]'
    var_37 = module_0.tokenize_json(var_36)
    var_38 = '{"text": "Hello   World"}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = '{"path": "C:\\\\Users\\\\John"}'
    var_41 = module_0.tokenize_json(var_40)
    var_42 = '{"int": 42, "float": 3.14, "bool": true, "null": null, "string": "hello"}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = '{"large": 12345678901234567890}'
    var_45 = module_0.tokenize_json(var_44)
    var_46 = '{"negative": -42}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = '{"zero": 0}'
    var_49 = module_0.tokenize_json(var_48)
    var_50 = '{"decimal": 0.5}'
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"exponent": 1e3}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = '{"negative_exponent": 1e-3}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = '{"positive_exponent": 1e+3}'
    var_57 = module_0.tokenize_json(var_56)
    var_58 = '{"capital_exponent": 1E3}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = '{"complex": 1+2j}'
    var_61 = module_0.tokenize_json(var_60)
    var_62 = '{"name": "John"'
    var_63 = module_0.tokenize_json(var_62)
    var_64 = '{"name": "John" "age": 30}'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = '{"name": "John", "age": 30,}'
    var_67 = module_0.tokenize_json(var_66)
    var_68 = '{name: "John"}'
    var_69 = module_0.tokenize_json(var_68)
    var_70 = "{'name': 'John'}"



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #9
#--------------------------



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
    var_12 = b'{"key": "value"}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_15 = {var_6: var_14}
    var_16 = '{"key": {"nested": "value"}}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = var_17.value[var_6]
    var_19 = 'nested'
    var_20 = 20
    var_21 = 26
    var_22 = module_1.ScalarToken(var_7, var_20, var_21, var_16)
    var_23 = {var_19: var_22}
    var_24 = '[1, 2, 3]'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 1
    var_27 = module_1.ScalarToken(var_26, var_26, var_26, var_24)
    var_28 = 2
    var_29 = 4
    var_30 = module_1.ScalarToken(var_28, var_29, var_29, var_24)
    var_31 = 3
    var_32 = module_1.ScalarToken(var_31, var_8, var_8, var_24)
    var_33 = [var_27, var_30, var_32]
    var_34 = '{"null": null, "true": true, "false": false}'
    var_35 = module_0.tokenize_json(var_34)
    var_36 = 'null'
    var_37 = 'true'
    var_38 = 'false'
    var_39 = None
    var_40 = 9
    var_41 = 12
    var_42 = module_1.ScalarToken(var_39, var_40, var_41, var_34)
    var_43 = True
    var_44 = 22
    var_45 = 25
    var_46 = module_1.ScalarToken(var_43, var_44, var_45, var_34)
    var_47 = False
    var_48 = 36
    var_49 = 40
    var_50 = module_1.ScalarToken(var_47, var_48, var_49, var_34)
    var_51 = {var_36: var_42, var_37: var_46, var_38: var_50}
    var_52 = '{"number": 123.45}'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = 'number'
    var_55 = 123.45
    var_56 = 11
    var_57 = 16
    var_58 = module_1.ScalarToken(var_55, var_56, var_57, var_52)
    var_59 = {var_54: var_58}
    var_60 = 'All tests passed!'
    var_61 = print(var_60)



# Parsed testcases at query #10
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = b'{"key": "value"}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = module_1.ScalarToken(var_5, var_6, var_7, var_20)
    var_25 = {var_4: var_24}
    var_26 = '  { "key" : "value" }  '
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 12
    var_29 = 18
    var_30 = module_1.ScalarToken(var_5, var_28, var_29, var_26)
    var_31 = {var_4: var_30}
    var_32 = '{"nested": {"inner": [1, 2]}}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'nested'
    var_35 = var_33.value[var_34]
    var_36 = 'inner'
    var_37 = var_35.value[var_36]
    var_38 = 23
    var_39 = module_1.ScalarToken(var_12, var_38, var_38, var_32)
    var_40 = 26
    var_41 = module_1.ScalarToken(var_14, var_40, var_40, var_32)
    var_42 = [var_39, var_41]
    var_43 = 'All tests passed!'
    var_44 = print(var_43)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 7
    var_5 = 13
    var_6 = module_1.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = {var_2: var_6}
    var_8 = '{"key": {"nested": "value"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = var_9.value[var_2]
    var_11 = 'nested'
    var_12 = 18
    var_13 = 24
    var_14 = module_1.ScalarToken(var_3, var_12, var_13, var_8)
    var_15 = {var_11: var_14}
    var_16 = '[1, 2, 3]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 1
    var_19 = module_1.ScalarToken(var_18, var_18, var_18, var_16)
    var_20 = 2
    var_21 = 4
    var_22 = module_1.ScalarToken(var_20, var_21, var_21, var_16)
    var_23 = 3
    var_24 = module_1.ScalarToken(var_23, var_4, var_4, var_16)
    var_25 = [var_19, var_22, var_24]
    var_26 = '{"bool": true, "null": null}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = 'bool'
    var_29 = 'null'
    var_30 = True
    var_31 = 9
    var_32 = 12
    var_33 = module_1.ScalarToken(var_30, var_31, var_32, var_26)
    var_34 = None
    var_35 = 21
    var_36 = module_1.ScalarToken(var_34, var_35, var_13, var_26)
    var_37 = {var_28: var_33, var_29: var_36}
    var_38 = '{"int": 42, "float": 3.14}'
    var_39 = module_0.tokenize_json(var_38)
    var_40 = 'int'
    var_41 = 'float'
    var_42 = 42
    var_43 = 8
    var_44 = module_1.ScalarToken(var_42, var_43, var_31, var_38)
    var_45 = 3.14
    var_46 = 19
    var_47 = 22
    var_48 = module_1.ScalarToken(var_45, var_46, var_47, var_38)
    var_49 = {var_40: var_44, var_41: var_48}
    var_50 = ''
    var_51 = module_0.tokenize_json(var_50)
    var_52 = '{"key": "value"'
    var_53 = module_0.tokenize_json(var_52)
    var_54 = b'{"key": "value"}'
    var_55 = module_0.tokenize_json(var_54)
    var_56 = 'All tests passed!'
    var_57 = print(var_56)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'Test case 1 passed'
    var_3 = print(var_2)
    var_4 = b'{"name": "John", "age": 30}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'Test case 2 passed'
    var_7 = print(var_6)
    var_8 = ''
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"name": "John", "age": 30'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"person": {"name": "John", "age": 30}}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = 'Test case 5 passed'
    var_15 = print(var_14)
    var_16 = '[1, 2, 3]'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = 'Test case 6 passed'
    var_19 = print(var_18)
    var_20 = '{"active": true, "value": null}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = 'Test case 7 passed'
    var_23 = print(var_22)
    var_24 = '{"integer": 42, "float": 3.14}'
    var_25 = module_0.tokenize_json(var_24)
    var_26 = 'Test case 8 passed'
    var_27 = print(var_26)
    var_28 = '{"message": "Hello\\nWorld"}'
    var_29 = module_0.tokenize_json(var_28)
    var_30 = 'Test case 9 passed'
    var_31 = print(var_30)
    var_32 = '{"name": "José"}'
    var_33 = module_0.tokenize_json(var_32)
    var_34 = 'Test case 10 passed'
    var_35 = print(var_34)
    var_36 = 'All test cases passed!'
    var_37 = print(var_36)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = '{"name": "John", "age": 30}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b'{"name": "John", "age": 30}'
    var_3 = module_0.tokenize_json(var_2)
    var_4 = ''
    var_5 = module_0.tokenize_json(var_4)
    var_6 = '{"name": "John", "age": 30'
    var_7 = module_0.tokenize_json(var_6)
    var_8 = '{"name": "John", "address": {"street": "123 Main St", "city": "New York"}}'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = '{"fruits": ["apple", "banana", "orange"]}'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = '{"integer": 42, "float": 3.14, "negative": -10}'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"flag": true, "empty": null}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = '{"message": "Hello\\nWorld"}'
    var_17 = module_0.tokenize_json(var_16)
    var_18 = '{"name": "Jöhn", "city": "München"}'
    var_19 = module_0.tokenize_json(var_18)
    var_20 = 'All test cases passed!'
    var_21 = print(var_20)



# Parsed testcases at query #14
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": {"nested": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value[var_4]
    var_25 = 'nested'
    var_26 = 20
    var_27 = 26
    var_28 = module_1.ScalarToken(var_5, var_26, var_27, var_22)
    var_29 = {var_25: var_28}
    var_30 = '[{"key": "value1"}, {"key": "value2"}]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 0
    var_35 = var_31.value[var_34]
    var_36 = var_31.value[var_12]
    var_37 = 'value1'
    var_38 = 8
    var_39 = 15
    var_40 = module_1.ScalarToken(var_37, var_38, var_39, var_30)
    var_41 = {var_4: var_40}
    var_42 = 'value2'
    var_43 = 27
    var_44 = 34
    var_45 = module_1.ScalarToken(var_42, var_43, var_44, var_30)
    var_46 = {var_4: var_45}
    var_47 = '{"int": 42, "float": 3.14, "negative": -10}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = 'int'
    var_50 = 'float'
    var_51 = 'negative'
    var_52 = 42
    var_53 = 9
    var_54 = module_1.ScalarToken(var_52, var_38, var_53, var_47)
    var_55 = 3.14
    var_56 = 19
    var_57 = 22
    var_58 = module_1.ScalarToken(var_55, var_56, var_57, var_47)
    var_59 = -10
    var_60 = 36
    var_61 = 38
    var_62 = module_1.ScalarToken(var_59, var_60, var_61, var_47)
    var_63 = {var_49: var_54, var_50: var_58, var_51: var_62}
    var_64 = '{"true": true, "false": false, "null": null}'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = 'true'
    var_67 = 'false'
    var_68 = 'null'
    var_69 = True
    var_70 = 10
    var_71 = module_1.ScalarToken(var_69, var_70, var_7, var_64)
    var_72 = False
    var_73 = 23
    var_74 = module_1.ScalarToken(var_72, var_73, var_43, var_64)
    var_75 = None
    var_76 = 37
    var_77 = 40
    var_78 = module_1.ScalarToken(var_75, var_76, var_77, var_64)
    var_79 = {var_66: var_71, var_67: var_74, var_68: var_78}
    var_80 = '{"quote": "\\"", "backslash": "\\\\"}'
    var_81 = module_0.tokenize_json(var_80)
    var_82 = 'quote'
    var_83 = 'backslash'
    var_84 = '"'
    var_85 = 11
    var_86 = module_1.ScalarToken(var_84, var_85, var_7, var_80)
    var_87 = '\\'
    var_88 = 29
    var_89 = 31
    var_90 = module_1.ScalarToken(var_87, var_88, var_89, var_80)
    var_91 = {var_82: var_86, var_83: var_90}
    var_92 = '{"emoji": "😀", "text": "café"}'
    var_93 = module_0.tokenize_json(var_92)
    var_94 = 'emoji'
    var_95 = 'text'
    var_96 = '😀'
    var_97 = 14
    var_98 = module_1.ScalarToken(var_96, var_85, var_97, var_92)
    var_99 = 'café'
    var_100 = 24
    var_101 = module_1.ScalarToken(var_99, var_100, var_88, var_92)
    var_102 = {var_94: var_98, var_95: var_101}
    var_103 = 'All tests passed!'
    var_104 = print(var_103)



# Parsed testcases at query #15
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": {"nested": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value[var_4]
    var_25 = 'nested'
    var_26 = 20
    var_27 = 26
    var_28 = module_1.ScalarToken(var_5, var_26, var_27, var_22)
    var_29 = {var_25: var_28}
    var_30 = '[{"key": "value1"}, {"key": "value2"}]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 0
    var_35 = var_31.value[var_34]
    var_36 = var_31.value[var_12]
    var_37 = '{"int": 42, "float": 3.14}'
    var_38 = module_0.tokenize_json(var_37)
    var_39 = 42
    var_40 = 8
    var_41 = 9
    var_42 = module_1.ScalarToken(var_39, var_40, var_41, var_37)
    var_43 = 3.14
    var_44 = 23
    var_45 = module_1.ScalarToken(var_43, var_26, var_44, var_37)
    var_46 = '{"true": true, "false": false, "null": null}'
    var_47 = module_0.tokenize_json(var_46)
    var_48 = True
    var_49 = 12
    var_50 = module_1.ScalarToken(var_48, var_41, var_49, var_46)
    var_51 = False
    var_52 = 27
    var_53 = module_1.ScalarToken(var_51, var_44, var_52, var_46)
    var_54 = None
    var_55 = 36
    var_56 = 39
    var_57 = module_1.ScalarToken(var_54, var_55, var_56, var_46)
    var_58 = '{"key": "value with \\"quotes\\""}'
    var_59 = module_0.tokenize_json(var_58)
    var_60 = 'value with "quotes"'
    var_61 = 30
    var_62 = module_1.ScalarToken(var_60, var_40, var_61, var_58)
    var_63 = '{"key": "café"}'
    var_64 = module_0.tokenize_json(var_63)
    var_65 = 'café'
    var_66 = module_1.ScalarToken(var_65, var_40, var_7, var_63)
    var_67 = 'All tests passed!'
    var_68 = print(var_67)



# Parsed testcases at query #16
#--------------------------



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
    var_10 = '[1, 2, 3]'
    var_11 = module_0.tokenize_json(var_10)
    var_12 = 1
    var_13 = module_1.ScalarToken(var_12, var_12, var_12, var_10)
    var_14 = 2
    var_15 = 4
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_10)
    var_17 = 3
    var_18 = module_1.ScalarToken(var_17, var_6, var_6, var_10)
    var_19 = [var_13, var_16, var_18]
    var_20 = '{"key": "value"'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = '{"key": {"nested": "value"}}'
    var_23 = module_0.tokenize_json(var_22)
    var_24 = var_23.value[var_4]
    var_25 = 'nested'
    var_26 = 20
    var_27 = 26
    var_28 = module_1.ScalarToken(var_5, var_26, var_27, var_22)
    var_29 = {var_25: var_28}
    var_30 = '[{"key": "value1"}, {"key": "value2"}]'
    var_31 = module_0.tokenize_json(var_30)
    var_32 = var_31.value
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 0
    var_35 = var_31.value[var_34]
    var_36 = var_31.value[var_12]
    var_37 = 'value1'
    var_38 = 8
    var_39 = 15
    var_40 = module_1.ScalarToken(var_37, var_38, var_39, var_30)
    var_41 = {var_4: var_40}
    var_42 = 'value2'
    var_43 = 27
    var_44 = 34
    var_45 = module_1.ScalarToken(var_42, var_43, var_44, var_30)
    var_46 = {var_4: var_45}
    var_47 = '{"int": 42, "float": 3.14, "negative": -10}'
    var_48 = module_0.tokenize_json(var_47)
    var_49 = 'int'
    var_50 = 'float'
    var_51 = 'negative'
    var_52 = 42
    var_53 = 9
    var_54 = module_1.ScalarToken(var_52, var_38, var_53, var_47)
    var_55 = 3.14
    var_56 = 19
    var_57 = 22
    var_58 = module_1.ScalarToken(var_55, var_56, var_57, var_47)
    var_59 = -10
    var_60 = 36
    var_61 = 38
    var_62 = module_1.ScalarToken(var_59, var_60, var_61, var_47)
    var_63 = {var_49: var_54, var_50: var_58, var_51: var_62}
    var_64 = '{"true": true, "false": false, "null": null}'
    var_65 = module_0.tokenize_json(var_64)
    var_66 = 'true'
    var_67 = 'false'
    var_68 = 'null'
    var_69 = True
    var_70 = 10
    var_71 = module_1.ScalarToken(var_69, var_70, var_7, var_64)
    var_72 = False
    var_73 = 24
    var_74 = 28
    var_75 = module_1.ScalarToken(var_72, var_73, var_74, var_64)
    var_76 = None
    var_77 = 41
    var_78 = module_1.ScalarToken(var_76, var_61, var_77, var_64)
    var_79 = {var_66: var_71, var_67: var_75, var_68: var_78}
    var_80 = '{"key": "value with \\"quotes\\""}'
    var_81 = module_0.tokenize_json(var_80)
    var_82 = 'value with "quotes"'
    var_83 = 30
    var_84 = module_1.ScalarToken(var_82, var_38, var_83, var_80)
    var_85 = {var_4: var_84}
    var_86 = '{"key": "café"}'
    var_87 = module_0.tokenize_json(var_86)
    var_88 = 'café'
    var_89 = module_1.ScalarToken(var_88, var_38, var_7, var_86)
    var_90 = {var_4: var_89}
    var_91 = '  {  "key"  :  "value"  }  '
    var_92 = module_0.tokenize_json(var_91)
    var_93 = 16
    var_94 = module_1.ScalarToken(var_5, var_93, var_57, var_91)
    var_95 = {var_4: var_94}
    var_96 = '{"array": [[1, 2], [3, 4]]}'
    var_97 = module_0.tokenize_json(var_96)
    var_98 = 'array'
    var_99 = var_97.value[var_98]
    var_100 = var_99.value
    var_101 = len(var_100)
    assert var_101 == 2
    var_102 = var_99.value[var_72]
    var_103 = var_99.value[var_69]
    var_104 = 12
    var_105 = module_1.ScalarToken(var_69, var_104, var_104, var_96)
    var_106 = module_1.ScalarToken(var_14, var_39, var_39, var_96)
    var_107 = [var_105, var_106]
    var_108 = module_1.ScalarToken(var_17, var_26, var_26, var_96)
    var_109 = 23
    var_110 = module_1.ScalarToken(var_15, var_109, var_109, var_96)
    var_111 = [var_108, var_110]
    var_112 = '{"number": 1.23e4}'
    var_113 = module_0.tokenize_json(var_112)
    var_114 = 'number'
    var_115 = 12300.0
    var_116 = 17
    var_117 = module_1.ScalarToken(var_115, var_104, var_116, var_112)
    var_118 = {var_114: var_117}
    var_119 = '{"key": "value",}'
    var_120 = module_0.tokenize_json(var_119)
    var_121 = '{"key1": "value1" "key2": "value2"}'
    var_122 = module_0.tokenize_json(var_121)
    var_123 = '{"key": "\\x"}'
    var_124 = module_0.tokenize_json(var_123)
    var_125 = '{"key": "value'
    var_126 = module_0.tokenize_json(var_125)
    var_127 = '{"number": 123.}'
    var_128 = module_0.tokenize_json(var_127)
    var_129 = '{} {}'
    var_130 = module_0.tokenize_json(var_129)
    var_131 = b'{"key": "value"}'
    var_132 = module_0.tokenize_json(var_131)
    var_133 = module_1.ScalarToken(var_5, var_6, var_7, var_129)
    var_134 = {var_4: var_133}
    var_135 = 'All tests passed!'
    var_136 = print(var_135)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = b''
    var_3 = module_0.tokenize_json(var_2)
    var_4 = '{"key": "value"}'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = 7
    var_9 = 13
    var_10 = module_1.ScalarToken(var_7, var_8, var_9, var_4)
    var_11 = {var_6: var_10}
    var_12 = '{"key": "value"'
    var_13 = module_0.tokenize_json(var_12)
    var_14 = '{"key": 123}'
    var_15 = module_0.tokenize_json(var_14)
    var_16 = 123
    var_17 = 9
    var_18 = module_1.ScalarToken(var_16, var_8, var_17, var_14)
    var_19 = {var_6: var_18}
    var_20 = '{"key": true}'
    var_21 = module_0.tokenize_json(var_20)
    var_22 = True
    var_23 = 10
    var_24 = module_1.ScalarToken(var_22, var_8, var_23, var_20)
    var_25 = {var_6: var_24}
    var_26 = '{"key": null}'
    var_27 = module_0.tokenize_json(var_26)
    var_28 = None
    var_29 = module_1.ScalarToken(var_28, var_8, var_23, var_26)
    var_30 = {var_6: var_29}
    var_31 = '{"key": [1, 2, 3]}'
    var_32 = module_0.tokenize_json(var_31)
    var_33 = var_32.value[var_6]
    var_34 = module_1.ScalarToken(var_22, var_17, var_17, var_31)
    var_35 = 2
    var_36 = 12
    var_37 = module_1.ScalarToken(var_35, var_36, var_36, var_31)
    var_38 = 3
    var_39 = 15
    var_40 = module_1.ScalarToken(var_38, var_39, var_39, var_31)
    var_41 = [var_34, var_37, var_40]
    var_42 = '{"key": {"nested": "value"}}'
    var_43 = module_0.tokenize_json(var_42)
    var_44 = var_43.value[var_6]
    var_45 = 'nested'
    var_46 = 18
    var_47 = 24
    var_48 = module_1.ScalarToken(var_7, var_46, var_47, var_42)
    var_49 = {var_45: var_48}
    var_50 = '  {  "key"  :  "value"  }  '
    var_51 = module_0.tokenize_json(var_50)
    var_52 = 19
    var_53 = module_1.ScalarToken(var_7, var_9, var_52, var_50)
    var_54 = {var_6: var_53}
    var_55 = '{"key": "value\\nwith newline"}'
    var_56 = module_0.tokenize_json(var_55)
    var_57 = 'value\nwith newline'
    var_58 = 27
    var_59 = module_1.ScalarToken(var_57, var_8, var_58, var_55)
    var_60 = {var_6: var_59}
    var_61 = '{"key": "value with unicode: é"}'
    var_62 = module_0.tokenize_json(var_61)
    var_63 = 'value with unicode: é'
    var_64 = 34
    var_65 = module_1.ScalarToken(var_63, var_8, var_64, var_61)
    var_66 = {var_6: var_65}
    var_67 = '{"key": 123.456}'
    var_68 = module_0.tokenize_json(var_67)
    var_69 = 123.456
    var_70 = module_1.ScalarToken(var_69, var_8, var_9, var_67)
    var_71 = {var_6: var_70}
    var_72 = '{"key": 1.23e4}'
    var_73 = module_0.tokenize_json(var_72)
    var_74 = 12300.0
    var_75 = module_1.ScalarToken(var_74, var_8, var_36, var_72)
    var_76 = {var_6: var_75}
    var_77 = '{"key": -123}'
    var_78 = module_0.tokenize_json(var_77)
    var_79 = -123
    var_80 = module_1.ScalarToken(var_79, var_8, var_23, var_77)
    var_81 = {var_6: var_80}
    var_82 = '{"key": 0}'
    var_83 = module_0.tokenize_json(var_82)
    var_84 = 0
    var_85 = module_1.ScalarToken(var_84, var_8, var_8, var_82)
    var_86 = {var_6: var_85}
    var_87 = '{}'
    var_88 = module_0.tokenize_json(var_87)
    var_89 = '[]'
    var_90 = module_0.tokenize_json(var_89)
    var_91 = '{"key1": "value1", "key2": "value2"}'
    var_92 = module_0.tokenize_json(var_91)
    var_93 = 'key1'
    var_94 = 'key2'
    var_95 = 'value1'
    var_96 = 8
    var_97 = module_1.ScalarToken(var_95, var_96, var_39, var_91)
    var_98 = 'value2'
    var_99 = 26
    var_100 = 33
    var_101 = module_1.ScalarToken(var_98, var_99, var_100, var_91)
    var_102 = {var_93: var_97, var_94: var_101}
    var_103 = '{"key": "value",}'
    var_104 = module_0.tokenize_json(var_103)
    var_105 = '{"key1": "value1" "key2": "value2"}'
    var_106 = module_0.tokenize_json(var_105)
    var_107 = '{"key" "value"}'
    var_108 = module_0.tokenize_json(var_107)
    var_109 = '{"key": "value"'
    var_110 = module_0.tokenize_json(var_109)
    var_111 = '["value"'
    var_112 = module_0.tokenize_json(var_111)
    var_113 = '{"key": "\\x"}'
    var_114 = module_0.tokenize_json(var_113)
    var_115 = '{"key": 123.}'
    var_116 = module_0.tokenize_json(var_115)
    var_117 = '{"key": tru}'
    var_118 = module_0.tokenize_json(var_117)
    var_119 = '{"key": fals}'
    var_120 = module_0.tokenize_json(var_119)



