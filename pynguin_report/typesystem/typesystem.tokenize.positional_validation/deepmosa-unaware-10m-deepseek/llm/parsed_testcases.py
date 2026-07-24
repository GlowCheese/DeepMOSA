####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'Jonathan'
    var_11 = {var_0: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'inner'
    var_15 = {}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = error.messages()[0]
    var_19 = ''
    var_20 = 15
    var_21 = {var_0: var_19, var_1: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = True
    var_24 = 5
    var_25 = module_1.String(max_length=var_24)
    var_26 = 'toolong'
    var_27 = module_0.Token(var_26)
    var_28 = module_2.validate_with_positions(token=var_27, validator=var_25)
    var_29 = error.messages()[0]
    var_30 = {}
    var_31 = module_0.Token(var_30)



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = 0
    var_3 = 4
    var_4 = '{"name": "test"}'
    var_5 = module_0.ScalarToken(var_1, var_2, var_3, var_4)
    var_6 = {var_0: var_5}
    var_7 = 20
    var_8 = module_1.Integer()
    var_9 = 'age'
    var_10 = 25
    var_11 = 10
    var_12 = 12
    var_13 = '{"age": 25}'
    var_14 = module_0.ScalarToken(var_10, var_11, var_12, var_13)
    var_15 = {var_9: var_14}
    var_16 = 15
    var_17 = error.messages()[0]
    var_18 = 'name'
    var_19 = 'user'
    var_20 = {}
    var_21 = 8
    var_22 = '{"user": {}}'
    var_23 = error.messages()[0]
    var_24 = 'toolong'
    var_25 = 19
    var_26 = '{"name": "toolong"}'
    var_27 = module_0.ScalarToken(var_24, var_11, var_25, var_26)
    var_28 = -5
    var_29 = 27
    var_30 = '{"age": -5}'
    var_31 = module_0.ScalarToken(var_28, var_10, var_29, var_30)
    var_32 = {var_0: var_27, var_9: var_31}
    var_33 = 30
    var_34 = '{"name": "toolong", "age": -5}'
    var_35 = module_0.ScalarToken(var_24, var_11, var_25, var_26)
    var_36 = {var_0: var_35}
    var_37 = error.messages()[0]
    var_38 = 5
    var_39 = module_1.String(max_length=var_38)
    var_40 = 6
    var_41 = '"test"'
    var_42 = module_0.ScalarToken(var_1, var_2, var_40, var_41)
    var_43 = module_2.validate_with_positions(token=var_42, validator=var_39)
    assert var_43 == 'test'
    var_44 = 3
    var_45 = module_1.String(max_length=var_44)
    var_46 = 9
    var_47 = '"toolong"'
    var_48 = module_0.ScalarToken(var_24, var_2, var_46, var_47)
    var_49 = module_2.validate_with_positions(token=var_48, validator=var_45)
    var_50 = error.messages()[0]
    var_51 = 'items'
    var_52 = 1
    var_53 = 13
    var_54 = '[1, -1]'
    var_55 = module_0.ScalarToken(var_52, var_12, var_53, var_54)
    var_56 = -1
    var_57 = 17
    var_58 = module_0.ScalarToken(var_56, var_16, var_57, var_54)
    var_59 = [var_55, var_58]
    var_60 = 18
    var_61 = '{"items": [1, -1]}'
    var_62 = module_0.ListToken(var_59, var_11, var_60, var_61)
    var_63 = {var_51: var_62}
    var_64 = module_2.validate_with_positions(token=var_48, validator=var_49)
    var_65 = error.messages()[0]



# Parsed testcases at query #3
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'user'
    var_11 = 'A'
    var_12 = 20
    var_13 = var_11 * var_12
    var_14 = -5
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = {var_10: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = 5
    var_19 = module_1.String(max_length=var_18)
    var_20 = 'too_long'
    var_21 = module_0.Token(var_20)
    var_22 = module_2.validate_with_positions(token=var_21, validator=var_19)
    var_23 = 'c'
    var_24 = {var_23: var_20}
    var_25 = module_0.Token(var_24)
    var_26 = 'inner'
    var_27 = {}
    var_28 = {var_26: var_27}
    var_29 = module_0.Token(var_28)



# Parsed testcases at query #4
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'inner'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 'LongName'
    var_16 = 15
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = True
    var_20 = 5
    var_21 = module_1.String(max_length=var_20)
    var_22 = 'TooLongString'
    var_23 = module_0.Token(var_22)
    var_24 = module_2.validate_with_positions(token=var_23, validator=var_21)
    var_25 = error.messages()[0]
    var_26 = {}
    var_27 = module_0.Token(var_26)
    var_28 = {var_24: var_15}
    var_29 = module_0.Token(var_28)
    var_30 = error.messages()[0]



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 5
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = 25
    var_7 = 7
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_1)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = 11
    var_12 = module_0.DictToken()
    var_13 = 'VeryLongName'
    var_14 = 13
    var_15 = module_0.ScalarToken(var_13, var_3, var_14, var_0)
    var_16 = 15
    var_17 = 18
    var_18 = module_0.ScalarToken(var_6, var_16, var_17, var_1)
    var_19 = {var_0: var_15, var_1: var_18}
    var_20 = 19
    var_21 = module_0.DictToken()
    var_22 = module_1.validate_with_positions(token=var_21, validator=var_0)
    var_23 = error.messages()[0]
    var_24 = 3
    var_25 = module_0.ScalarToken(var_6, var_3, var_24, var_22)
    var_26 = {var_22: var_25}
    var_27 = 4
    var_28 = module_0.DictToken()
    var_29 = module_1.validate_with_positions(token=var_28, validator=var_0)
    var_30 = error.messages()[0]
    var_31 = module_0.ScalarToken(var_13, var_3, var_14, var_0)
    var_32 = -5
    var_33 = module_0.ScalarToken(var_32, var_16, var_17, var_29)
    var_34 = {var_0: var_31, var_29: var_33}
    var_35 = module_0.DictToken()
    var_36 = module_1.validate_with_positions(token=var_35, validator=var_0)
    var_37 = module_2.String(max_length=var_4)
    var_38 = 'TooLong'
    var_39 = 'test'
    var_40 = module_0.ScalarToken(var_38, var_3, var_7, var_39)
    var_41 = module_1.validate_with_positions(token=var_40, validator=var_37)
    var_42 = error.messages()[0]
    var_43 = 'data'
    var_44 = 22
    var_45 = {}
    var_46 = module_0.DictToken()
    var_47 = module_1.validate_with_positions(token=var_46, validator=var_41)



# Parsed testcases at query #6
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'Jonathan'
    var_11 = {var_0: var_10}
    var_12 = module_0.Token(var_11)
    var_13 = error.messages()[0]
    var_14 = 'inner'
    var_15 = {}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = error.messages()[0]
    var_19 = 15
    var_20 = {var_0: var_10, var_1: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = True
    var_23 = 5
    var_24 = module_1.String(max_length=var_23)
    var_25 = 'TooLongString'
    var_26 = module_0.Token(var_25)
    var_27 = module_2.validate_with_positions(token=var_26, validator=var_24)
    var_28 = error.messages()[0]
    var_29 = {}
    var_30 = module_0.Token(var_29)
    var_31 = error.messages()[0]



# Parsed testcases at query #7
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 4
    var_5 = '{"name": "John", "age": 25}'
    var_6 = module_0.ScalarToken(var_2, var_3, var_4, var_5)
    var_7 = 25
    var_8 = 16
    var_9 = 18
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_5)
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = 30
    var_13 = module_0.DictToken()
    var_14 = module_1.Integer()
    var_15 = '{"age": 25}'
    var_16 = module_0.ScalarToken(var_7, var_8, var_9, var_15)
    var_17 = {var_1: var_16}
    var_18 = 20
    var_19 = module_0.DictToken()
    var_20 = error.messages()[var_3]
    var_21 = 3
    var_22 = module_1.String(max_length=var_21)
    var_23 = module_2.validate_with_positions(token=var_19, validator=var_22)
    var_24 = 'first'
    var_25 = 'second'
    var_26 = 'toolong'
    var_27 = 10
    var_28 = 19
    var_29 = '{"first": "toolong", "second": 5}'
    var_30 = module_0.ScalarToken(var_26, var_27, var_28, var_29)
    var_31 = 5
    var_32 = 31
    var_33 = 32
    var_34 = module_0.ScalarToken(var_31, var_32, var_33, var_29)
    var_35 = {var_24: var_30, var_25: var_34}
    var_36 = 45
    var_37 = module_0.DictToken()
    var_38 = 'toolongvalue'
    var_39 = 13
    var_40 = '"toolongvalue"'
    var_41 = module_0.ScalarToken(var_38, var_3, var_39, var_40)
    var_42 = 5
    var_43 = module_1.String(max_length=var_42)
    var_44 = module_2.validate_with_positions(token=var_41, validator=var_43)
    var_45 = 'level2'
    var_46 = 'level3'
    var_47 = 'level1'
    var_48 = 'level2'
    var_49 = {}
    var_50 = 12
    var_51 = 14
    var_52 = '{}'
    var_53 = module_0.DictToken()
    var_54 = {var_48: var_53}
    var_55 = '{"level2": {}}'
    var_56 = module_0.DictToken()
    var_57 = {var_47: var_56}
    var_58 = '{"level1": {"level2": {}}}'
    var_59 = module_0.DictToken()



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 4
    var_5 = '{"name":"John","age":25}'
    var_6 = module_0.ScalarToken(var_2, var_3, var_4, var_5)
    var_7 = 25
    var_8 = 14
    var_9 = 16
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_5)
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = 30
    var_13 = module_0.DictToken()
    var_14 = '{"age":25}'
    var_15 = module_0.ScalarToken(var_7, var_8, var_9, var_14)
    var_16 = {var_1: var_15}
    var_17 = 18
    var_18 = module_0.DictToken()
    var_19 = 'username'
    var_20 = 'score'
    var_21 = 'ab'
    var_22 = 15
    var_23 = 17
    var_24 = '{"username":"ab","score":-5}'
    var_25 = module_0.ScalarToken(var_21, var_22, var_23, var_24)
    var_26 = -5
    var_27 = 32
    var_28 = module_0.ScalarToken(var_26, var_12, var_27, var_24)
    var_29 = {var_19: var_25, var_20: var_28}
    var_30 = 34
    var_31 = module_0.DictToken()
    var_32 = module_1.String()
    var_33 = module_1.String()
    var_34 = module_1.String()
    var_35 = 'address'
    var_36 = 'Alice'
    var_37 = 7
    var_38 = '{"name":"Alice","address":{"street":"Main","city":"NYC"}}'
    var_39 = module_0.ScalarToken(var_36, var_3, var_37, var_38)
    var_40 = 'street'
    var_41 = 'city'
    var_42 = 'Main'
    var_43 = 20
    var_44 = 26
    var_45 = module_0.ScalarToken(var_42, var_43, var_44, var_38)
    var_46 = 'NYC'
    var_47 = 39
    var_48 = module_0.ScalarToken(var_46, var_30, var_47, var_38)
    var_49 = {var_40: var_45, var_41: var_48}
    var_50 = 10
    var_51 = 41
    var_52 = '{"street":"Main","city":"NYC"}'
    var_53 = module_0.DictToken()
    var_54 = {var_0: var_39, var_35: var_53}
    var_55 = 43
    var_56 = module_0.DictToken()
    var_57 = 'Bob'
    var_58 = 5
    var_59 = '{"name":"Bob","address":{"city":"LA"}}'
    var_60 = module_0.ScalarToken(var_57, var_3, var_58, var_59)
    var_61 = 'LA'
    var_62 = 22
    var_63 = module_0.ScalarToken(var_61, var_17, var_62, var_59)
    var_64 = {var_41: var_63}
    var_65 = 8
    var_66 = 24
    var_67 = '{"city":"LA"}'
    var_68 = module_0.DictToken()
    var_69 = {var_0: var_60, var_35: var_68}
    var_70 = module_0.DictToken()
    var_71 = 'toolongusername'
    var_72 = '"toolongusername"'
    var_73 = module_0.ScalarToken(var_71, var_3, var_9, var_72)
    var_74 = 10
    var_75 = module_1.String(max_length=var_74)
    var_76 = module_2.validate_with_positions(token=var_73, validator=var_75)
    var_77 = 'items'
    var_78 = '{"items":[1,2,3]}'
    var_79 = {}
    var_80 = 2
    var_81 = '{}'
    var_82 = module_0.DictToken()



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = 'Jonathan'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = error.messages()[0]
    var_11 = module_1.Integer()
    var_12 = {var_1: var_3}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 15
    var_16 = {var_0: var_7, var_1: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = 'inner'
    var_19 = 'toolong'
    var_20 = {var_18: var_19}
    var_21 = module_0.Token(var_20)
    var_22 = error.messages()[0]
    var_23 = 3
    var_24 = module_1.String(max_length=var_23)
    var_25 = module_0.Token(var_19)
    var_26 = module_2.validate_with_positions(token=var_25, validator=var_24)
    var_27 = error.messages()[0]
    var_28 = 0
    var_29 = module_1.Integer(minimum=var_28)
    var_30 = 42
    var_31 = module_0.Token(var_30)
    var_32 = module_2.validate_with_positions(token=var_31, validator=var_29)
    assert var_32 == 42
    var_33 = {}
    var_34 = module_0.Token(var_33)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = 5
    var_9 = var_2 * var_8
    var_10 = {var_0: var_9, var_1: var_3}
    var_11 = 30
    var_12 = "{'name': 'JohnJohnJohnJohnJohn', 'age': 25}"
    var_13 = exc_info.value.messages()[var_5]
    var_14 = {var_0: var_2}
    var_15 = 15
    var_16 = "{'name': 'John'}"
    var_17 = var_2 * var_8
    var_18 = -5
    var_19 = {var_0: var_17, var_1: var_18}
    var_20 = "{'name': 'JohnJohnJohnJohnJohn', 'age': -5}"
    var_21 = 'data'
    var_22 = 'items'
    var_23 = 'id'
    var_24 = 'value'
    var_25 = 1
    var_26 = 'test'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = 2
    var_29 = {var_23: var_28}
    var_30 = [var_27, var_29]
    var_31 = {var_22: var_30}
    var_32 = {var_21: var_31}
    var_33 = 50
    var_34 = ''
    var_35 = module_0.String(max_length=var_8)
    var_36 = 'toolong'
    var_37 = 7
    var_38 = module_1.Token(var_36, var_5, var_37, var_36)
    var_39 = module_2.validate_with_positions(token=var_38, validator=var_35)
    var_40 = {}
    var_41 = '{}'
    var_42 = module_2.validate_with_positions(token=var_38, validator=var_39)
    var_43 = 'first'
    var_44 = 'second'
    var_45 = 'aa'
    var_46 = 'bb'
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = "{'first': 'aa', 'second': 'bb'}"
    var_49 = module_2.validate_with_positions(token=var_38, validator=var_39)



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 'line_index'
    var_7 = 'column_index'
    var_8 = 0
    var_9 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_10 = 20
    var_11 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_12 = module_0.Token(var_4)
    var_13 = 5
    var_14 = module_1.String(max_length=var_13)
    var_15 = 'toolong'
    var_16 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_17 = 7
    var_18 = {var_5: var_17, var_6: var_8, var_7: var_17}
    var_19 = module_0.Token(var_15)
    var_20 = module_2.validate_with_positions(token=var_19, validator=var_14)
    var_21 = error.messages()[0]
    var_22 = 'inner'
    var_23 = {}
    var_24 = {var_22: var_23}
    var_25 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_26 = 15
    var_27 = {var_5: var_26, var_6: var_8, var_7: var_26}
    var_28 = module_0.Token(var_24)
    var_29 = 'first'
    var_30 = 'second'
    var_31 = {var_29: var_15, var_30: var_13}
    var_32 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_33 = {var_5: var_3, var_6: var_8, var_7: var_3}
    var_34 = module_0.Token(var_31)
    var_35 = 10
    var_36 = {var_5: var_35, var_6: var_8, var_7: var_35}
    var_37 = 17
    var_38 = {var_5: var_37, var_6: var_8, var_7: var_37}
    var_39 = module_0.Token(var_15)
    var_40 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_41 = 21
    var_42 = {var_5: var_41, var_6: var_8, var_7: var_41}
    var_43 = module_0.Token(var_13)
    var_44 = var_34.lookup
    var_45 = 'level1'
    var_46 = 'level2'
    var_47 = {}
    var_48 = {var_46: var_47}
    var_49 = {var_45: var_48}
    var_50 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_51 = 30
    var_52 = {var_5: var_51, var_6: var_8, var_7: var_51}
    var_53 = module_0.Token(var_49)
    var_54 = {}
    var_55 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_56 = 22
    var_57 = {var_5: var_56, var_6: var_8, var_7: var_56}
    var_58 = module_0.Token(var_54)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 4
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_2)
    var_6 = module_2.validate_with_positions(token=var_5, validator=var_1)
    assert var_6 == 'hello'
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'Alice'
    var_10 = 30
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = '{"name": "Alice", "age": 30}'
    var_13 = 1
    var_14 = 5
    var_15 = '"name"'
    var_16 = module_1.ScalarToken(var_7, var_13, var_14, var_15)
    var_17 = 8
    var_18 = 13
    var_19 = '"Alice"'
    var_20 = module_1.ScalarToken(var_9, var_17, var_18, var_19)
    var_21 = 16
    var_22 = 18
    var_23 = '"age"'
    var_24 = module_1.ScalarToken(var_8, var_21, var_22, var_23)
    var_25 = 21
    var_26 = 22
    var_27 = '30'
    var_28 = module_1.ScalarToken(var_10, var_25, var_26, var_27)
    var_29 = [var_16, var_20, var_24, var_28]
    var_30 = {var_8: var_10}
    var_31 = '{"age": 30}'
    var_32 = 3
    var_33 = module_1.ScalarToken(var_8, var_13, var_32, var_23)
    var_34 = 6
    var_35 = 7
    var_36 = module_1.ScalarToken(var_10, var_34, var_35, var_27)
    var_37 = [var_33, var_36]
    var_38 = 'person'
    var_39 = {}
    var_40 = {var_38: var_39}
    var_41 = 12
    var_42 = '{"person": {}}'
    var_43 = '"person"'
    var_44 = module_1.ScalarToken(var_38, var_13, var_34, var_43)
    var_45 = {}
    var_46 = 9
    var_47 = '{}'
    var_48 = []
    var_49 = 'required'
    var_50 = 'The field'
    var_51 = module_0.String(max_length=var_32)
    var_52 = module_1.ScalarToken(var_2, var_3, var_4, var_2)
    var_53 = module_2.validate_with_positions(token=var_52, validator=var_51)
    var_54 = 'a'
    var_55 = 'bb'
    var_56 = 'ccc'
    var_57 = [var_54, var_55, var_56]
    var_58 = 20
    var_59 = '["a", "bb", "ccc"]'
    var_60 = '"a"'
    var_61 = module_1.ScalarToken(var_54, var_13, var_13, var_60)
    var_62 = '"bb"'
    var_63 = module_1.ScalarToken(var_55, var_14, var_34, var_62)
    var_64 = '"ccc"'
    var_65 = module_1.ScalarToken(var_56, var_53, var_41, var_64)
    var_66 = [var_61, var_63, var_65]
    var_67 = module_1.ListToken(var_57, var_3, var_58, var_59)
    var_68 = 2
    var_69 = module_0.String(max_length=var_68)
    var_70 = module_2.validate_with_positions(token=var_67, validator=var_69)
    var_71 = 'b'
    var_72 = 'c'
    var_73 = 'aa'
    var_74 = 'cc'
    var_75 = {var_54: var_73, var_71: var_55, var_72: var_74}
    var_76 = '{"a": "aa", "b": "bb", "c": "cc"}'
    var_77 = module_1.ScalarToken(var_54, var_13, var_13, var_60)
    var_78 = '"aa"'
    var_79 = module_1.ScalarToken(var_73, var_14, var_34, var_78)
    var_80 = '"b"'
    var_81 = module_1.ScalarToken(var_71, var_70, var_70, var_80)
    var_82 = 14
    var_83 = 15
    var_84 = module_1.ScalarToken(var_55, var_82, var_83, var_62)
    var_85 = 19
    var_86 = '"c"'
    var_87 = module_1.ScalarToken(var_72, var_85, var_85, var_86)
    var_88 = 23
    var_89 = 24
    var_90 = '"cc"'
    var_91 = module_1.ScalarToken(var_74, var_88, var_89, var_90)
    var_92 = [var_77, var_79, var_81, var_84, var_87, var_91]
    var_93 = {}
    var_94 = []



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'char_index'
    var_6 = 'line_index'
    var_7 = 'column_index'
    var_8 = 0
    var_9 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_10 = 20
    var_11 = {var_5: var_10, var_6: var_8, var_7: var_10}
    var_12 = module_0.Token(var_4)
    var_13 = 5
    var_14 = var_2 * var_13
    var_15 = {var_0: var_14, var_1: var_3}
    var_16 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_17 = 30
    var_18 = {var_5: var_17, var_6: var_8, var_7: var_17}
    var_19 = module_0.Token(var_15)
    var_20 = e.messages()[0]
    var_21 = {var_1: var_3}
    var_22 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_23 = 15
    var_24 = {var_5: var_23, var_6: var_8, var_7: var_23}
    var_25 = module_0.Token(var_21)
    var_26 = e.messages()[0]
    var_27 = module_1.Integer()
    var_28 = 'user'
    var_29 = 'id'
    var_30 = 'A'
    var_31 = var_30 * var_10
    var_32 = -5
    var_33 = {var_0: var_31, var_1: var_32}
    var_34 = 'not_an_int'
    var_35 = {var_28: var_33, var_29: var_34}
    var_36 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_37 = 50
    var_38 = {var_5: var_37, var_6: var_8, var_7: var_37}
    var_39 = module_0.Token(var_35)
    var_40 = var_39
    var_41 = var_30 * var_10
    var_42 = -5
    var_43 = {var_0: var_41, var_1: var_42}
    var_44 = 10
    var_45 = {var_5: var_44, var_6: var_8, var_7: var_44}
    var_46 = 40
    var_47 = {var_5: var_46, var_6: var_8, var_7: var_46}
    var_48 = module_0.Token(var_43)
    var_49 = module_1.String(max_length=var_13)
    var_50 = 'too_long'
    var_51 = 1
    var_52 = {var_5: var_13, var_6: var_51, var_7: var_13}
    var_53 = 13
    var_54 = {var_5: var_53, var_6: var_51, var_7: var_53}
    var_55 = module_0.Token(var_50)
    var_56 = module_2.validate_with_positions(token=var_55, validator=var_49)
    var_57 = e.messages()[0]
    var_58 = 100
    var_59 = module_1.Integer(minimum=var_44, maximum=var_58)
    var_60 = {var_5: var_8, var_6: var_8, var_7: var_8}
    var_61 = 2
    var_62 = {var_5: var_61, var_6: var_8, var_7: var_61}
    var_63 = module_0.Token(var_37)
    var_64 = module_2.validate_with_positions(token=var_63, validator=var_59)
    assert var_64 == 50



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = 5
    var_8 = var_2 * var_7
    var_9 = {var_0: var_8, var_1: var_3}
    var_10 = module_0.Token(var_9)
    var_11 = error.messages()[0]
    var_12 = {var_0: var_2}
    var_13 = module_0.Token(var_12)
    var_14 = var_2 * var_7
    var_15 = -5
    var_16 = {var_0: var_14, var_1: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = module_1.String(max_length=var_7)
    var_19 = 'too long string'
    var_20 = module_0.Token(var_19)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_18)
    var_22 = 'inner'
    var_23 = var_2 * var_7
    var_24 = -5
    var_25 = {var_21: var_23, var_1: var_24}
    var_26 = {var_22: var_25}
    var_27 = module_0.Token(var_26)
    var_28 = 'field1'
    var_29 = 'field2'
    var_30 = 'field3'
    var_31 = 'aa'
    var_32 = 'bb'
    var_33 = 'cc'
    var_34 = {var_28: var_31, var_29: var_32, var_30: var_33}
    var_35 = module_0.Token(var_34)



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = 0
    var_3 = 4
    var_4 = '{"name": "test"}'
    var_5 = module_0.ScalarToken(var_1, var_2, var_3, var_4)
    var_6 = {var_0: var_5}
    var_7 = 20
    var_8 = 'age'
    var_9 = 'toolongname'
    var_10 = 17
    var_11 = 28
    var_12 = '{"name": "toolongname"}'
    var_13 = module_0.ScalarToken(var_9, var_10, var_11, var_12)
    var_14 = -5
    var_15 = 35
    var_16 = 37
    var_17 = '{"age": -5}'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_0: var_13, var_8: var_18}
    var_20 = 45
    var_21 = '{"name": "toolongname", "age": -5}'
    var_22 = 25
    var_23 = 10
    var_24 = 12
    var_25 = '{"age": 25}'
    var_26 = module_0.ScalarToken(var_22, var_23, var_24, var_25)
    var_27 = {var_8: var_26}
    var_28 = 'items'
    var_29 = 'id'
    var_30 = 'value'
    var_31 = 1
    var_32 = 21
    var_33 = '[{"id": 1}]'
    var_34 = module_0.ScalarToken(var_31, var_7, var_32, var_33)
    var_35 = 23
    var_36 = 29
    var_37 = '[{"value": "test"}]'
    var_38 = module_0.ScalarToken(var_1, var_35, var_36, var_37)
    var_39 = {var_29: var_34, var_30: var_38}
    var_40 = 15
    var_41 = '[{"id": 1, "value": "test"}]'
    var_42 = 2
    var_43 = 46
    var_44 = '[{"id": 2}]'
    var_45 = module_0.ScalarToken(var_42, var_20, var_43, var_44)
    var_46 = {var_29: var_45}
    var_47 = 40
    var_48 = 55
    var_49 = 60
    var_50 = '{"items": [{"id": 1, "value": "test"}, {"id": 2}]}'
    var_51 = 70
    var_52 = 'toolongvalue'
    var_53 = '"toolongvalue"'
    var_54 = module_0.ScalarToken(var_52, var_2, var_24, var_53)
    var_55 = 5
    var_56 = module_1.String(max_length=var_55)
    var_57 = module_2.validate_with_positions(token=var_54, validator=var_56)
    var_58 = 'third'
    var_59 = 'first'
    var_60 = 'second'
    var_61 = 'value3'
    var_62 = 50
    var_63 = 56
    var_64 = '{"third": "value3"}'
    var_65 = module_0.ScalarToken(var_61, var_62, var_63, var_64)
    var_66 = 'value1'
    var_67 = 16
    var_68 = '{"first": "value1"}'
    var_69 = module_0.ScalarToken(var_66, var_23, var_67, var_68)
    var_70 = 'value2'
    var_71 = 30
    var_72 = 36
    var_73 = '{"second": "value2"}'
    var_74 = module_0.ScalarToken(var_70, var_71, var_72, var_73)
    var_75 = {var_58: var_65, var_59: var_69, var_60: var_74}
    var_76 = 65
    var_77 = '{"first": "value1", "second": "value2", "third": "value3"}'
    var_78 = module_0.ScalarToken(var_61, var_62, var_63, var_64)
    var_79 = {var_58: var_78}
    var_80 = module_2.validate_with_positions(token=var_54, validator=var_55)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 4
    var_5 = module_0.ScalarToken(var_2, var_3, var_4)
    var_6 = 25
    var_7 = 10
    var_8 = 12
    var_9 = module_0.ScalarToken(var_6, var_7, var_8)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = 20
    var_12 = module_0.ScalarToken(var_0, var_3, var_4)
    var_13 = module_0.ScalarToken(var_1, var_7, var_8)
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = module_0.ScalarToken(var_6, var_7, var_8)
    var_16 = {var_1: var_15}
    var_17 = module_0.ScalarToken(var_1, var_7, var_8)
    var_18 = {var_1: var_17}
    var_19 = error.messages()[0]
    var_20 = 'data'
    var_21 = {}
    var_22 = 15
    var_23 = module_0.ScalarToken(var_20, var_3, var_4)
    var_24 = {var_20: var_23}
    var_25 = error.messages()[0]
    var_26 = 'Johnathan'
    var_27 = 19
    var_28 = module_0.ScalarToken(var_26, var_7, var_27)
    var_29 = 27
    var_30 = module_0.ScalarToken(var_22, var_6, var_29)
    var_31 = {var_0: var_28, var_1: var_30}
    var_32 = 30
    var_33 = module_0.ScalarToken(var_0, var_3, var_4)
    var_34 = 23
    var_35 = module_0.ScalarToken(var_1, var_11, var_34)
    var_36 = {var_0: var_33, var_1: var_35}
    var_37 = 5
    var_38 = module_1.String(max_length=var_37)
    var_39 = 'test'
    var_40 = module_0.ScalarToken(var_39, var_3, var_4)
    var_41 = module_2.validate_with_positions(token=var_40, validator=var_38)
    assert var_41 == 'test'
    var_42 = 3
    var_43 = module_1.String(max_length=var_42)
    var_44 = module_0.ScalarToken(var_39, var_3, var_4)
    var_45 = module_2.validate_with_positions(token=var_44, validator=var_43)
    var_46 = error.messages()[0]
    var_47 = 'items'
    var_48 = 'id'
    var_49 = 1
    var_50 = 21
    var_51 = module_0.ScalarToken(var_49, var_11, var_50)
    var_52 = {var_48: var_51}
    var_53 = {}
    var_54 = 32
    var_55 = 35
    var_56 = 40
    var_57 = module_0.ScalarToken(var_47, var_3, var_37)
    var_58 = {var_47: var_57}
    var_59 = error.messages()[0]



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'test'
    var_3 = None
    var_4 = module_1.Token(var_2)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_1)
    assert var_5 == 'test'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'John'
    var_9 = 25
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_1.Token(var_8)
    var_12 = module_1.Token(var_9)
    var_13 = {var_6: var_11, var_7: var_12}
    var_14 = 3
    var_15 = module_0.String(max_length=var_14)
    var_16 = 'toolong'
    var_17 = module_1.Token(var_16)
    var_18 = module_2.validate_with_positions(token=var_17, validator=var_15)
    var_19 = error.messages()[0]
    var_20 = {var_6: var_8}
    var_21 = module_1.Token(var_8)
    var_22 = {var_6: var_21}
    var_23 = 'Jonathan'
    var_24 = 15
    var_25 = {var_6: var_23, var_7: var_24}
    var_26 = module_1.Token(var_23)
    var_27 = module_1.Token(var_24)
    var_28 = {var_6: var_26, var_7: var_27}
    var_29 = {}
    var_30 = 'field1'
    var_31 = 'field2'
    var_32 = 'obj'
    var_33 = 'char_index'
    var_34 = 10
    var_35 = {var_33: var_34}
    var_36 = {var_33: var_18}
    var_37 = 'outer_field'
    var_38 = {}
    var_39 = {var_37: var_38}
    var_40 = {}
    var_41 = {}



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 'inner'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = 'email'
    var_16 = 'invalid-email'
    var_17 = {var_15: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = error.messages()[0]
    var_20 = 5
    var_21 = module_1.String(max_length=var_20)
    var_22 = 'toolong'
    var_23 = module_0.Token(var_22)
    var_24 = module_2.validate_with_positions(token=var_23, validator=var_21)
    var_25 = error.messages()[0]
    var_26 = ''
    var_27 = 15
    var_28 = {var_24: var_26, var_1: var_27}
    var_29 = module_0.Token(var_28)
    var_30 = module_0.Token(var_5)
    var_31 = 0
    var_32 = 100
    var_33 = module_1.Integer(minimum=var_31, maximum=var_32)
    var_34 = 50
    var_35 = module_0.Token(var_34)
    var_36 = module_2.validate_with_positions(token=var_35, validator=var_33)
    assert var_36 == 50



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = 'VeryLongName'
    var_9 = -5
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = 30
    var_12 = "{'name': 'VeryLongName', 'age': -5}"
    var_13 = 'start_position'
    var_14 = 'end_position'
    var_15 = {var_13: var_14}
    var_16 = 15
    var_17 = "{'name': 'John'}"
    var_18 = 'items'
    var_19 = []
    var_20 = {var_18: var_19}
    var_21 = 10
    var_22 = "{'items': []}"
    var_23 = 5
    var_24 = module_0.String(max_length=var_23)
    var_25 = 'TooLong'
    var_26 = 7
    var_27 = module_1.Token(var_25, var_5, var_26, var_25)
    var_28 = module_2.validate_with_positions(token=var_27, validator=var_24)
    var_29 = 'address'
    var_30 = 'street'
    var_31 = 'Main St'
    var_32 = {var_30: var_31}
    var_33 = {var_28: var_14, var_29: var_32}
    var_34 = 40
    var_35 = "{'name': 'John', 'address': {'street': 'Main St'}}"



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.base as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = 3
    var_8 = var_2 * var_7
    var_9 = {var_0: var_8, var_1: var_3}
    var_10 = module_0.Token(var_9)
    var_11 = 0
    var_12 = exc_info.value.messages()[var_11]
    var_13 = 'maximum'
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = 5
    var_17 = module_1.String(max_length=var_16)
    var_18 = 'toolong'
    var_19 = module_0.Token(var_18)
    var_20 = module_2.validate_with_positions(token=var_19, validator=var_17)
    var_21 = 'A'
    var_22 = 20
    var_23 = var_21 * var_22
    var_24 = -5
    var_25 = {var_20: var_23, var_1: var_24}
    var_26 = module_0.Token(var_25)
    var_27 = -5
    var_28 = {var_20: var_2, var_1: var_27}
    var_29 = 1
    var_30 = module_3.Position(var_11)
    var_31 = 50
    var_32 = module_3.Position(var_31)
    var_33 = module_0.Token(var_28)
    var_34 = 10



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = -5
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = "{'name': 'John', 'age': -5}"
    var_11 = error.messages()[0]
    var_12 = {var_1: var_3}
    var_13 = 10
    var_14 = "{'age': 25}"
    var_15 = error.messages()[0]
    var_16 = 'items'
    var_17 = 1
    var_18 = 'invalid'
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_16: var_20}
    var_22 = "{'items': [1, 'invalid', 3]}"
    var_23 = 'Jonathan'
    var_24 = 15
    var_25 = {var_0: var_23, var_1: var_24}
    var_26 = "{'name': 'Jonathan', 'age': 15}"
    var_27 = module_0.String(max_length=var_19)
    var_28 = 'toolong'
    var_29 = 6
    var_30 = module_1.Token(var_28, var_5, var_29, var_28)
    var_31 = module_2.validate_with_positions(token=var_30, validator=var_27)
    var_32 = 100
    var_33 = module_0.Integer(minimum=var_5, maximum=var_32)
    var_34 = 50
    var_35 = 2
    var_36 = '50'
    var_37 = module_1.Token(var_34, var_5, var_35, var_36)
    var_38 = module_2.validate_with_positions(token=var_37, validator=var_33)
    assert var_38 == 50
    var_39 = 'data'
    var_40 = 'inner'
    var_41 = {}
    var_42 = {var_40: var_41}
    var_43 = {var_39: var_42}
    var_44 = "{'data': {'inner': {}}}"



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = module_0.Integer()
    var_9 = {var_1: var_3}
    var_10 = 10
    var_11 = "{'age': 25}"
    var_12 = error.messages()[0]
    var_13 = 'user'
    var_14 = {}
    var_15 = {var_13: var_14}
    var_16 = 15
    var_17 = "{'user': {}}"
    var_18 = 'Jonathan'
    var_19 = {var_0: var_18}
    var_20 = "{'name': 'Jonathan'}"
    var_21 = 'items'
    var_22 = -1
    var_23 = 5
    var_24 = [var_22, var_23]
    var_25 = {var_21: var_24}
    var_26 = "{'items': [-1, 5]}"
    var_27 = 'b'
    var_28 = 'c'
    var_29 = 'toolong'
    var_30 = {var_27: var_23, var_28: var_29}
    var_31 = 30
    var_32 = "{'b': 5, 'c': 'toolong'}"
    var_33 = 3
    var_34 = module_0.String(min_length=var_33)
    var_35 = 'ab'
    var_36 = 2
    var_37 = module_1.ScalarToken(var_35, var_5, var_36, var_35)
    var_38 = module_2.validate_with_positions(token=var_37, validator=var_34)
    var_39 = 100
    var_40 = module_0.Integer(minimum=var_5, maximum=var_39)
    var_41 = 50
    var_42 = '50'
    var_43 = module_1.ScalarToken(var_41, var_5, var_36, var_42)
    var_44 = module_2.validate_with_positions(token=var_43, validator=var_40)
    assert var_44 == 50



# Parsed testcases at query #5
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 4
    var_5 = '{"name": "John", "age": 25}'
    var_6 = module_0.ScalarToken(var_2, var_3, var_4, var_5)
    var_7 = 25
    var_8 = 14
    var_9 = 16
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_5)
    var_11 = {var_0: var_6, var_1: var_10}
    var_12 = 30
    var_13 = 'optional_field'
    var_14 = 'test'
    var_15 = 20
    var_16 = 24
    var_17 = '{"optional_field": "test"}'
    var_18 = module_0.ScalarToken(var_14, var_15, var_16, var_17)
    var_19 = {var_13: var_18}
    var_20 = e.messages()[0]
    var_21 = var_0.items
    var_22 = e.messages()[0]
    var_23 = 3
    var_24 = module_1.String(max_length=var_23)
    var_25 = 'toolong'
    var_26 = 7
    var_27 = '"toolong"'
    var_28 = module_0.ScalarToken(var_25, var_3, var_26, var_27)
    var_29 = module_2.validate_with_positions(token=var_28, validator=var_24)
    var_30 = e.messages()[0]
    var_31 = 'first'
    var_32 = 'second'
    var_33 = 'abc'
    var_34 = 10
    var_35 = 13
    var_36 = '{"first": "abc", "second": 5}'
    var_37 = module_0.ScalarToken(var_33, var_34, var_35, var_36)
    var_38 = 5
    var_39 = 26
    var_40 = module_0.ScalarToken(var_38, var_7, var_39, var_36)
    var_41 = {var_31: var_37, var_32: var_40}
    var_42 = 35
    var_43 = module_2.validate_with_positions(token=var_28, validator=var_29)
    var_44 = 'level3'
    var_45 = 'level2'
    var_46 = var_2.level1[var_45][var_44]
    var_47 = module_2.validate_with_positions(token=var_46, validator=var_4)
    var_48 = e.messages()[0]



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 4
    var_5 = module_1.ScalarToken(var_2, var_3, var_4)
    var_6 = module_2.validate_with_positions(token=var_5, validator=var_1)
    assert var_6 == 'hello'
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'Alice'
    var_10 = 30
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 7
    var_13 = 11
    var_14 = module_1.ScalarToken(var_9, var_12, var_13)
    var_15 = (var_7, var_14)
    var_16 = 17
    var_17 = 18
    var_18 = module_1.ScalarToken(var_10, var_16, var_17)
    var_19 = (var_8, var_18)
    var_20 = [var_15, var_19]
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = 'Bob'
    var_24 = {var_7: var_23}
    var_25 = 15
    var_26 = 9
    var_27 = module_1.ScalarToken(var_23, var_12, var_26)
    var_28 = (var_7, var_27)
    var_29 = [var_28]
    var_30 = error.messages()[var_3]
    var_31 = 'person'
    var_32 = {}
    var_33 = 10
    var_34 = []
    var_35 = 20
    var_36 = {}
    var_37 = []
    var_38 = lambda m: m.start_position.char_index
    var_39 = module_0.String(min_length=var_0)
    var_40 = 'hi'
    var_41 = 1
    var_42 = module_1.ScalarToken(var_40, var_3, var_41)
    var_43 = module_2.validate_with_positions(token=var_42, validator=var_39)
    var_44 = error.messages()[var_3]
    var_45 = 'items'
    var_46 = 'a'
    var_47 = 'b'
    var_48 = 'c'
    var_49 = [var_46, var_47, var_48]
    var_50 = module_1.ScalarToken(var_46, var_41, var_41)
    var_51 = 3
    var_52 = module_1.ScalarToken(var_47, var_51, var_51)
    var_53 = module_1.ScalarToken(var_48, var_43, var_43)
    var_54 = [var_50, var_52, var_53]
    var_55 = module_1.ListToken(var_49, var_3, var_33, var_54)
    var_56 = module_0.String(max_length=var_41)
    var_57 = module_0.Array(var_56)
    var_58 = 'bb'
    var_59 = [var_46, var_58, var_48]
    var_60 = module_1.ScalarToken(var_46, var_41, var_41)
    var_61 = module_1.ScalarToken(var_58, var_51, var_4)
    var_62 = 6
    var_63 = module_1.ScalarToken(var_48, var_62, var_62)
    var_64 = [var_60, var_61, var_63]
    var_65 = module_1.ListToken(var_59, var_3, var_33, var_64)
    var_66 = module_2.validate_with_positions(token=var_65, validator=var_57)
    var_67 = error.messages()[var_3]
    var_68 = 'first'
    var_69 = 'second'
    var_70 = 'toolong'
    var_71 = {var_68: var_70, var_69: var_35}
    var_72 = module_1.ScalarToken(var_70, var_26, var_25)
    var_73 = (var_68, var_72)
    var_74 = 24
    var_75 = 25
    var_76 = module_1.ScalarToken(var_35, var_74, var_75)
    var_77 = (var_69, var_76)
    var_78 = [var_73, var_77]
    var_79 = 'level2'
    var_80 = 'level3'
    var_81 = module_0.String()
    var_82 = {var_80: var_81}
    var_83 = {var_79: var_82}
    var_84 = 'level1'
    var_85 = 'level2'
    var_86 = {}
    var_87 = []
    var_88 = {}
    var_89 = []
    var_90 = 35
    var_91 = {}
    var_92 = []
    var_93 = {}
    var_94 = []
    var_95 = error.messages()[var_3]



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = {var_0: var_2}
    var_9 = 15
    var_10 = "{'name': 'John'}"
    var_11 = error.messages()[0]
    var_12 = 5
    var_13 = var_2 * var_12
    var_14 = {var_0: var_13, var_1: var_3}
    var_15 = 30
    var_16 = "{'name': 'JohnJohnJohnJohnJohn', 'age': 25}"
    var_17 = error.messages()[0]
    var_18 = 'user'
    var_19 = {var_0: var_2}
    var_20 = {var_18: var_19}
    var_21 = "{'user': {'name': 'John'}}"
    var_22 = error.messages()[0]
    var_23 = {}
    var_24 = 2
    var_25 = '{}'
    var_26 = 'value'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = "{'value': 'test'}"
    var_30 = 3
    var_31 = module_0.String(max_length=var_30)
    var_32 = error.messages()[0]
    var_33 = module_0.String()
    var_34 = 'item1'
    var_35 = 'item2'
    var_36 = 'item'
    var_37 = 10
    var_38 = var_36 * var_37
    var_39 = [var_34, var_35, var_38]
    var_40 = 50
    var_41 = "['item1', 'item2', 'itemitemitemitemitemitemitemitemitemitem']"
    var_42 = module_1.ListToken(var_39, var_5, var_40, var_41)
    var_43 = 'address'
    var_44 = 'Alice'
    var_45 = 'street'
    var_46 = 'city'
    var_47 = '123 Main'
    var_48 = 'Town'
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = {var_0: var_44, var_43: var_49}
    var_51 = "{'name': 'Alice', 'address': {'street': '123 Main', 'city': 'Town'}}"



# Parsed testcases at query #8
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = 5
    var_8 = var_2 * var_7
    var_9 = {var_0: var_8, var_1: var_3}
    var_10 = module_0.Token(var_9)
    var_11 = error.messages()[0]
    var_12 = {var_0: var_2}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = var_2 * var_7
    var_16 = -5
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = 'person'
    var_20 = var_2 * var_7
    var_21 = -5
    var_22 = {var_0: var_20, var_1: var_21}
    var_23 = {var_19: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.String(max_length=var_7)
    var_26 = 'toolong'
    var_27 = module_0.Token(var_26)
    var_28 = module_2.validate_with_positions(token=var_27, validator=var_25)
    var_29 = error.messages()[0]
    var_30 = 'short'
    var_31 = module_0.Token(var_30)
    var_32 = module_2.validate_with_positions(token=var_31, validator=var_25)
    assert var_32 == 'short'



# Parsed testcases at query #9
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = module_1.validate_with_positions(token=var_8, validator=var_0)
    var_10 = error.messages()[0]
    var_11 = 'Jonathan'
    var_12 = {var_0: var_11}
    var_13 = module_0.Token(var_12)
    var_14 = module_1.validate_with_positions(token=var_13, validator=var_0)
    var_15 = error.messages()[0]
    var_16 = 'inner'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = module_1.validate_with_positions(token=var_19, validator=var_0)
    var_21 = error.messages()[0]
    var_22 = 15
    var_23 = {var_20: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.validate_with_positions(token=var_24, validator=var_0)
    var_26 = 5
    var_27 = module_2.String(max_length=var_26)
    var_28 = 'Hello World'
    var_29 = module_0.Token(var_28)
    var_30 = module_1.validate_with_positions(token=var_29, validator=var_27)
    var_31 = error.messages()[0]



# Parsed testcases at query #10
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 5
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = 25
    var_7 = 7
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_1)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = 11
    var_12 = module_0.DictToken()
    var_13 = 'Jonathan'
    var_14 = 9
    var_15 = module_0.ScalarToken(var_13, var_3, var_14, var_0)
    var_16 = {var_0: var_15}
    var_17 = module_0.DictToken()
    var_18 = module_1.validate_with_positions(token=var_17, validator=var_0)
    var_19 = error.messages()[0]
    var_20 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_21 = {var_0: var_20}
    var_22 = 6
    var_23 = module_0.DictToken()
    var_24 = module_1.validate_with_positions(token=var_23, validator=var_0)
    var_25 = error.messages()[0]
    var_26 = 'nested'
    var_27 = 'value'
    var_28 = 'toolong'
    var_29 = 17
    var_30 = module_0.ScalarToken(var_28, var_8, var_29, var_27)
    var_31 = {var_27: var_30}
    var_32 = 8
    var_33 = 18
    var_34 = module_0.DictToken()
    var_35 = {var_26: var_34}
    var_36 = 19
    var_37 = module_0.DictToken()
    var_38 = module_1.validate_with_positions(token=var_37, validator=var_0)
    var_39 = error.messages()[0]
    var_40 = 'first'
    var_41 = 'second'
    var_42 = 20
    var_43 = 27
    var_44 = module_0.ScalarToken(var_28, var_42, var_43, var_40)
    var_45 = 'alsobad'
    var_46 = 30
    var_47 = 37
    var_48 = module_0.ScalarToken(var_45, var_46, var_47, var_41)
    var_49 = {var_40: var_44, var_41: var_48}
    var_50 = 38
    var_51 = module_0.DictToken()
    var_52 = module_1.validate_with_positions(token=var_51, validator=var_0)
    var_53 = 3
    var_54 = module_2.String(max_length=var_53)
    var_55 = module_0.ScalarToken(var_28, var_3, var_7, var_27)
    var_56 = module_1.validate_with_positions(token=var_55, validator=var_54)
    var_57 = error.messages()[0]
    var_58 = module_2.String(max_length=var_53)
    var_59 = module_1.validate_with_positions(token=var_56, validator=var_58)
    var_60 = error.messages()[0]



# Parsed testcases at query #11
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_1: var_3}
    var_8 = module_0.Token(var_7)
    var_9 = error.messages()[0]
    var_10 = 5
    var_11 = var_2 * var_10
    var_12 = {var_0: var_11, var_1: var_3}
    var_13 = module_0.Token(var_12)
    var_14 = error.messages()[0]
    var_15 = var_2 * var_10
    var_16 = -5
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = 'inner'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = {}
    var_24 = module_0.Token(var_23)
    var_25 = error.messages()[0]
    var_26 = module_1.String(max_length=var_10)
    var_27 = 'hello'
    var_28 = module_0.Token(var_27)
    var_29 = module_2.validate_with_positions(token=var_28, validator=var_26)
    assert var_29 == 'hello'
    var_30 = 'too long'
    var_31 = module_0.Token(var_30)
    var_32 = module_2.validate_with_positions(token=var_31, validator=var_26)
    var_33 = error.messages()[0]
    var_34 = {}
    var_35 = module_0.Token(var_34)
    var_36 = 'address'
    var_37 = 'street'
    var_38 = '123 Main St'
    var_39 = {var_37: var_38}
    var_40 = {var_32: var_2, var_36: var_39}
    var_41 = module_0.Token(var_40)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2
import typesystem.base as module_3

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 0
    var_10 = exc_info.value.messages()[var_9]
    var_11 = 5
    var_12 = var_2 * var_11
    var_13 = -5
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = 'maximum length'
    var_17 = 'minimum'
    var_18 = 'user'
    var_19 = 'Alice'
    var_20 = {var_0: var_19}
    var_21 = {var_18: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = ''
    var_24 = module_0.Token(var_23)
    var_25 = True
    var_26 = module_1.String()
    var_27 = module_2.validate_with_positions(token=var_24, validator=var_26)
    var_28 = 'Bob'
    var_29 = 30
    var_30 = {var_25: var_28, var_26: var_29}
    var_31 = {var_18: var_30}
    var_32 = module_0.Token(var_31)
    var_33 = {var_25: var_23}
    var_34 = 1
    var_35 = module_3.Position(var_9)
    var_36 = 10
    var_37 = module_3.Position(var_36)
    var_38 = module_0.Token(var_33)
    var_39 = 'start_position'
    var_40 = 'end_position'



# Parsed testcases at query #13
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 20
    var_7 = "{'name': 'John', 'age': 25}"
    var_8 = 'optional_field'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = "{'optional_field': 'test'}"
    var_12 = error.messages()[var_5]
    var_13 = 'short'
    var_14 = 'toolongvalue'
    var_15 = [var_13, var_14]
    var_16 = 30
    var_17 = "['short', 'toolongvalue']"
    var_18 = module_0.ListToken(var_15, var_5, var_16, var_17)
    var_19 = error.messages()[var_5]
    var_20 = 'toolong'
    var_21 = 15
    var_22 = {var_0: var_20, var_1: var_21}
    var_23 = "{'name': 'toolong', 'age': 15}"
    var_24 = lambda m: m.start_position.char_index
    var_25 = 'data'
    var_26 = 'value'
    var_27 = 'ok'
    var_28 = {var_26: var_27}
    var_29 = 'too_long'
    var_30 = {var_26: var_29}
    var_31 = [var_28, var_30]
    var_32 = {var_25: var_31}
    var_33 = 40
    var_34 = "{'data': [{'value': 'ok'}, {'value': 'too_long'}]}"
    var_35 = error.messages()[var_5]
    var_36 = 5
    var_37 = module_1.String(max_length=var_36)
    var_38 = 12
    var_39 = "'toolongvalue'"
    var_40 = module_0.Token(var_14, var_5, var_38, var_39)
    var_41 = module_2.validate_with_positions(token=var_40, validator=var_37)
    var_42 = error.messages()[var_5]
    var_43 = {}
    var_44 = 2
    var_45 = '{}'
    var_46 = error.messages()[var_5]



# Parsed testcases at query #14
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = 0
    var_3 = 4
    var_4 = '{"name": "test"}'
    var_5 = module_0.ScalarToken(var_1, var_2, var_3, var_4)
    var_6 = {var_0: var_5}
    var_7 = 20
    var_8 = module_1.Integer()
    var_9 = 'age'
    var_10 = 25
    var_11 = 10
    var_12 = 12
    var_13 = '{"age": 25}'
    var_14 = module_0.ScalarToken(var_10, var_11, var_12, var_13)
    var_15 = {var_9: var_14}
    var_16 = 15
    var_17 = error.messages()[0]
    var_18 = 'data'
    var_19 = {}
    var_20 = 8
    var_21 = '{"data": {}}'
    var_22 = error.messages()[0]
    var_23 = 'toolong'
    var_24 = 9
    var_25 = 16
    var_26 = '{"name": "toolong"}'
    var_27 = module_0.ScalarToken(var_23, var_24, var_25, var_26)
    var_28 = {var_0: var_27}
    var_29 = error.messages()[0]
    var_30 = ''
    var_31 = '{"name": ""}'
    var_32 = module_0.ScalarToken(var_30, var_20, var_11, var_31)
    var_33 = 22
    var_34 = '{"name": "", "age": 15}'
    var_35 = module_0.ScalarToken(var_16, var_7, var_33, var_34)
    var_36 = {var_0: var_32, var_9: var_35}
    var_37 = 30
    var_38 = True
    var_39 = 5
    var_40 = module_1.String(max_length=var_39)
    var_41 = 'toolongvalue'
    var_42 = 13
    var_43 = '"toolongvalue"'
    var_44 = module_0.ScalarToken(var_41, var_2, var_42, var_43)
    var_45 = module_2.validate_with_positions(token=var_44, validator=var_40)
    var_46 = error.messages()[0]
    var_47 = 'address'
    var_48 = 'John'
    var_49 = '{"name": "John"}'
    var_50 = module_0.ScalarToken(var_48, var_24, var_16, var_49)
    var_51 = 'street'
    var_52 = 'city'
    var_53 = 'Main St'
    var_54 = 39
    var_55 = '"street": "Main St"'
    var_56 = module_0.ScalarToken(var_53, var_37, var_54, var_55)
    var_57 = 'City'
    var_58 = 45
    var_59 = 51
    var_60 = '"city": "City"'
    var_61 = module_0.ScalarToken(var_57, var_58, var_59, var_60)
    var_62 = {var_51: var_56, var_52: var_61}
    var_63 = 55
    var_64 = '"address": {...}'
    var_65 = 60
    var_66 = '{"name": "John", "address": {...}}'
    var_67 = 'items'
    var_68 = 'ok'
    var_69 = '"ok"'
    var_70 = module_0.ScalarToken(var_68, var_12, var_25, var_69)
    var_71 = 18
    var_72 = 27
    var_73 = '"toolong"'
    var_74 = module_0.ScalarToken(var_23, var_71, var_72, var_73)
    var_75 = [var_70, var_74]
    var_76 = '"items": [...]'
    var_77 = module_0.ListToken(var_75, var_20, var_37, var_76)
    var_78 = {var_67: var_77}
    var_79 = 35
    var_80 = '{"items": ["ok", "toolong"]}'
    var_81 = module_2.validate_with_positions(token=var_44, validator=var_45)
    var_82 = error.messages()[0]



# Parsed testcases at query #15
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 0
    var_4 = 5
    var_5 = module_0.ScalarToken(var_2, var_3, var_4)
    var_6 = 25
    var_7 = 7
    var_8 = 9
    var_9 = module_0.ScalarToken(var_6, var_7, var_8)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = 3
    var_12 = var_2 * var_11
    var_13 = 15
    var_14 = module_0.ScalarToken(var_12, var_3, var_13)
    var_15 = -5
    var_16 = 17
    var_17 = 19
    var_18 = module_0.ScalarToken(var_15, var_16, var_17)
    var_19 = {var_0: var_14, var_1: var_18}
    var_20 = module_0.ScalarToken(var_2, var_3, var_4)
    var_21 = {var_0: var_20}
    var_22 = module_1.String(max_length=var_4)
    var_23 = 'toolong'
    var_24 = module_0.ScalarToken(var_23, var_3, var_7)
    var_25 = module_2.validate_with_positions(token=var_24, validator=var_22)
    var_26 = module_1.Integer()
    var_27 = 'id'
    var_28 = 'tags'
    var_29 = 1
    var_30 = 2
    var_31 = module_0.ScalarToken(var_29, var_30, var_11)
    var_32 = 'a'
    var_33 = 10
    var_34 = 11
    var_35 = module_0.ScalarToken(var_32, var_33, var_34)
    var_36 = 'b'
    var_37 = 13
    var_38 = 14
    var_39 = module_0.ScalarToken(var_36, var_37, var_38)
    var_40 = [var_35, var_39]
    var_41 = module_0.ListToken(var_40, var_8, var_13)
    var_42 = {var_27: var_31, var_28: var_41}
    var_43 = 'z'
    var_44 = 'error1'
    var_45 = 20
    var_46 = 26
    var_47 = module_0.ScalarToken(var_44, var_45, var_46)
    var_48 = 'error2'
    var_49 = 6
    var_50 = module_0.ScalarToken(var_48, var_3, var_49)
    var_51 = {var_43: var_47, var_32: var_50}
    var_52 = module_2.validate_with_positions(token=var_24, validator=var_25)



# Parsed testcases at query #16
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.positional_validation as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 5
    var_5 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = 25
    var_7 = 7
    var_8 = 10
    var_9 = module_0.ScalarToken(var_6, var_7, var_8, var_1)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = 11
    var_12 = module_0.DictToken()
    var_13 = 'Johnathan'
    var_14 = module_0.ScalarToken(var_13, var_3, var_8, var_0)
    var_15 = 12
    var_16 = 15
    var_17 = module_0.ScalarToken(var_6, var_15, var_16, var_1)
    var_18 = {var_0: var_14, var_1: var_17}
    var_19 = 16
    var_20 = module_0.DictToken()
    var_21 = module_1.validate_with_positions(token=var_20, validator=var_0)
    var_22 = error.messages()[0]
    var_23 = module_0.ScalarToken(var_6, var_7, var_8, var_21)
    var_24 = {var_21: var_23}
    var_25 = module_0.DictToken()
    var_26 = module_1.validate_with_positions(token=var_25, validator=var_0)
    var_27 = error.messages()[0]
    var_28 = module_0.ScalarToken(var_13, var_3, var_8, var_0)
    var_29 = -5
    var_30 = module_0.ScalarToken(var_29, var_15, var_16, var_26)
    var_31 = {var_0: var_28, var_26: var_30}
    var_32 = module_0.DictToken()
    var_33 = module_1.validate_with_positions(token=var_32, validator=var_0)
    var_34 = module_2.String(max_length=var_4)
    var_35 = 'toolong'
    var_36 = 'test'
    var_37 = module_0.ScalarToken(var_35, var_3, var_7, var_36)
    var_38 = module_1.validate_with_positions(token=var_37, validator=var_34)
    var_39 = error.messages()[0]
    var_40 = 'items'
    var_41 = 23
    var_42 = module_1.validate_with_positions(token=var_37, validator=var_38)
    var_43 = 'first'
    var_44 = 'second'
    var_45 = 'third'
    var_46 = 'aa'
    var_47 = 50
    var_48 = 55
    var_49 = module_0.ScalarToken(var_46, var_47, var_48, var_43)
    var_50 = 'bb'
    var_51 = 30
    var_52 = 35
    var_53 = module_0.ScalarToken(var_50, var_51, var_52, var_44)
    var_54 = 'cc'
    var_55 = module_0.ScalarToken(var_54, var_8, var_16, var_45)
    var_56 = {var_43: var_49, var_44: var_53, var_45: var_55}
    var_57 = 60
    var_58 = module_0.DictToken()
    var_59 = module_1.validate_with_positions(token=var_58, validator=var_38)



# Parsed testcases at query #17
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'hello'
    var_1 = 1
    var_2 = 0
    var_3 = (var_1, var_2, var_2)
    var_4 = 5
    var_5 = (var_1, var_4, var_4)
    var_6 = module_0.Token(var_0)
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'Alice'
    var_10 = 30
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_1, var_2, var_2)
    var_13 = 20
    var_14 = (var_1, var_13, var_13)
    var_15 = module_0.Token(var_11)
    var_16 = module_1.Integer()
    var_17 = {var_8: var_10}
    var_18 = (var_1, var_2, var_2)
    var_19 = 10
    var_20 = (var_1, var_19, var_19)
    var_21 = module_0.Token(var_17)
    var_22 = 'person'
    var_23 = {var_8: var_10}
    var_24 = {var_22: var_23}
    var_25 = (var_1, var_2, var_2)
    var_26 = (var_1, var_13, var_13)
    var_27 = module_0.Token(var_24)
    var_28 = 'wrong'
    var_29 = 2
    var_30 = 25
    var_31 = (var_29, var_2, var_30)
    var_32 = (var_29, var_4, var_10)
    var_33 = module_0.Token(var_28)
    var_34 = 'TooLongName'
    var_35 = 15
    var_36 = {var_7: var_34, var_8: var_35}
    var_37 = 3
    var_38 = 35
    var_39 = (var_37, var_2, var_38)
    var_40 = 60
    var_41 = (var_37, var_30, var_40)
    var_42 = module_0.Token(var_36)
    var_43 = None
    var_44 = 4
    var_45 = 65
    var_46 = (var_44, var_2, var_45)
    var_47 = (var_44, var_2, var_45)
    var_48 = module_0.Token(var_43)
    var_49 = module_1.String()
    var_50 = module_2.validate_with_positions(token=var_48, validator=var_49)
    var_51 = 'nested'
    var_52 = 'value'
    var_53 = {var_51: var_52}
    var_54 = 70
    var_55 = (var_4, var_2, var_54)
    var_56 = 85
    var_57 = (var_4, var_35, var_56)
    var_58 = module_0.Token(var_53)
    var_59 = module_2.validate_with_positions(token=var_58, validator=var_49)



# Parsed testcases at query #18
#--------------------------


import typesystem.tokenize.tokens as module_0
import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 0
    var_4 = 4
    var_5 = module_0.ScalarToken(var_2, var_3, var_4)
    var_6 = 25
    var_7 = 6
    var_8 = 8
    var_9 = module_0.ScalarToken(var_6, var_7, var_8)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.ScalarToken(var_2, var_3, var_4)
    var_12 = {var_0: var_11}
    var_13 = 'Jonathan'
    var_14 = module_0.ScalarToken(var_13, var_3, var_8)
    var_15 = 15
    var_16 = 10
    var_17 = 12
    var_18 = module_0.ScalarToken(var_15, var_16, var_17)
    var_19 = {var_0: var_14, var_1: var_18}
    var_20 = 'inner'
    var_21 = {}
    var_22 = 'items'
    var_23 = 'abc'
    var_24 = 13
    var_25 = module_0.ScalarToken(var_23, var_16, var_24)
    var_26 = 'abcd'
    var_27 = 19
    var_28 = module_0.ScalarToken(var_26, var_15, var_27)
    var_29 = [var_25, var_28]
    var_30 = 20
    var_31 = module_0.ListToken(var_29, var_8, var_30)
    var_32 = {var_22: var_31}
    var_33 = 5
    var_34 = module_1.String(max_length=var_33)
    var_35 = 'test'
    var_36 = module_0.ScalarToken(var_35, var_3, var_4)
    var_37 = module_2.validate_with_positions(token=var_36, validator=var_34)
    assert var_37 == 'test'
    var_38 = 3
    var_39 = module_1.String(max_length=var_38)
    var_40 = module_0.ScalarToken(var_35, var_3, var_4)
    var_41 = module_2.validate_with_positions(token=var_40, validator=var_39)
    var_42 = {}
    var_43 = module_2.validate_with_positions(token=var_40, validator=var_41)



