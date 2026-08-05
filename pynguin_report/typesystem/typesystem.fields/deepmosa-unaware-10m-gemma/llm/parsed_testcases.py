####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = None
    var_4 = excinfo.value.messages()[var_2]
    var_5 = var_4.code
    assert var_5 == 'null'
    var_6 = 'test'
    var_7 = 'test'
    var_8 = excinfo.value.messages()[var_2]
    var_9 = var_8.code
    assert var_9 == 'union'
    var_10 = 'constraint_violation'
    var_11 = 'test'
    var_12 = excinfo.value.messages()[var_2]
    var_13 = var_12.code
    assert var_13 == 'constraint_violation'
    var_14 = 'type'
    var_15 = 'test'
    var_16 = excinfo.value.messages()[var_2]
    var_17 = var_16.code
    assert var_17 == 'type'



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Message'
    var_1 = 'ValidationError'
    var_2 = 'Name'
    var_3 = module_0.String()
    var_4 = 'Age'
    var_5 = module_0.Integer()
    var_6 = 'Active'
    var_7 = module_0.Boolean()
    var_8 = 'name'
    var_9 = 'age'
    var_10 = {var_8: var_3, var_9: var_5}
    var_11 = [var_8]
    var_12 = True
    var_13 = module_0.Object(properties=var_10, additional_properties=var_12, required=var_11)
    var_14 = 'extra'
    var_15 = 'Alice'
    var_16 = 30
    var_17 = 'info'
    var_18 = {var_8: var_15, var_9: var_16, var_14: var_17}
    var_19 = var_13.validate(var_18)
    var_20 = None
    var_21 = var_13.validate(var_20)
    var_22 = 'messages'
    var_23 = 'May not be null'
    var_24 = 'not'
    var_25 = 'a'
    var_26 = 'dict'
    var_27 = [var_24, var_25, var_26]
    var_28 = var_13.validate(var_27)
    var_29 = 'Must be an object'
    var_30 = 'age'
    var_31 = 30
    var_32 = {var_30: var_31}
    var_33 = var_13.validate(var_32)
    var_34 = 'This field is required'
    var_35 = [var_8]
    var_36 = 'name'
    var_37 = 'age'
    var_38 = 123
    var_39 = 30
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = var_13.validate(var_40)
    var_42 = 'Must be a string'
    var_43 = [var_8]
    var_44 = 'a'
    var_45 = {var_44: var_3}
    var_46 = 2
    var_47 = module_0.Object(properties=var_45, min_properties=var_46)
    var_48 = 'a'
    var_49 = 'val'
    var_50 = {var_48: var_49}
    var_51 = var_47.validate(var_50)
    var_52 = 'Must have at least 2 properties'
    var_53 = {var_8: var_3}
    var_54 = False
    var_55 = module_0.Object(properties=var_53, additional_properties=var_54)
    var_56 = 'name'
    var_57 = 'age'
    var_58 = 'Alice'
    var_59 = 30
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = var_55.validate(var_60)
    var_62 = 'Invalid property name'
    var_63 = '^user_\\d+$'
    var_64 = {var_63: var_3}
    var_65 = module_0.Object(pattern_properties=var_64, additional_properties=var_54)
    var_66 = 'user_123'
    var_67 = 'Bob'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)
    var_70 = 'user_abc'
    var_71 = 'Bob'
    var_72 = {var_70: var_71}
    var_73 = var_65.validate(var_72)
    var_74 = 'Default'
    var_75 = 'Guest'
    var_76 = module_0.String()
    var_77 = 'username'
    var_78 = {var_77: var_76}
    var_79 = module_0.Object(properties=var_78)
    var_80 = 'other'
    var_81 = 'val'
    var_82 = {var_80: var_81}
    var_83 = var_79.validate(var_82)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Age'
    var_1 = 18
    var_2 = module_0.Integer()
    var_3 = 'Name'
    var_4 = module_0.String()
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {var_5: var_4, var_6: var_2}
    var_8 = [var_5]
    var_9 = module_0.Object(properties=var_7, required=var_8)
    var_10 = 'John'
    var_11 = 25
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = 'Jane'
    var_15 = {var_5: var_14}
    var_16 = var_9.validate(var_15)
    var_17 = 'age'
    var_18 = 30
    var_19 = {var_17: var_18}
    var_20 = var_9.validate(var_19)
    var_21 = 'required'
    var_22 = [var_20]
    var_23 = 'not'
    var_24 = 'a'
    var_25 = 'dict'
    var_26 = [var_23, var_24, var_25]
    var_27 = var_9.validate(var_26)
    var_28 = 'type'
    var_29 = 3
    var_30 = module_0.String(max_length=var_29)
    var_31 = 'a'
    var_32 = module_0.String()
    var_33 = {var_31: var_32}
    var_34 = module_0.Object(properties=var_33, property_names=var_30)
    var_35 = 'abcde'
    var_36 = 'val'
    var_37 = {var_35: var_36}
    var_38 = var_34.validate(var_37)
    var_39 = 'invalid_property'
    var_40 = 'abcde'
    var_41 = [var_40]
    var_42 = module_0.String()
    var_43 = {var_31: var_42}
    var_44 = False
    var_45 = module_0.Object(properties=var_43, additional_properties=var_44)
    var_46 = 'a'
    var_47 = 'b'
    var_48 = 'val'
    var_49 = 'unexpected'
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = var_45.validate(var_50)
    var_52 = 'b'
    var_53 = module_0.Integer()
    var_54 = module_0.String()
    var_55 = {var_31: var_54}
    var_56 = module_0.Object(properties=var_55, additional_properties=var_53)
    var_57 = 'extra'
    var_58 = 'val'
    var_59 = 10
    var_60 = {var_31: var_58, var_57: var_59}
    var_61 = var_56.validate(var_60)
    var_62 = 'a'
    var_63 = 'extra'
    var_64 = 'val'
    var_65 = 'not_an_int'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_56.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_31: var_68}
    var_70 = 2
    var_71 = module_0.Object(properties=var_69, min_properties=var_70, max_properties=var_70)
    var_72 = 'a'
    var_73 = 'val'
    var_74 = {var_72: var_73}
    var_75 = var_71.validate(var_74)
    var_76 = 'min_properties'
    var_77 = 'a'
    var_78 = 'b'
    var_79 = 'c'
    var_80 = 'val'
    var_81 = {var_77: var_80, var_78: var_80, var_79: var_80}
    var_82 = var_71.validate(var_81)
    var_83 = 'max_properties'
    var_84 = 'fixed'
    var_85 = module_0.String()
    var_86 = {var_84: var_85}
    var_87 = '^dyn_.*'
    var_88 = module_0.String()
    var_89 = {var_87: var_88}
    var_90 = module_0.Object(properties=var_86, pattern_properties=var_89)
    var_91 = 'dyn_key'
    var_92 = 'static'
    var_93 = 'dynamic_value'
    var_94 = {var_84: var_92, var_91: var_93}
    var_95 = var_90.validate(var_94)
    var_96 = True
    var_97 = module_0.String()
    var_98 = {var_31: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = None
    var_101 = {var_31: var_100}
    var_102 = var_99.validate(var_101)
    var_103 = 'a'
    var_104 = None
    var_105 = {var_103: var_104}
    var_106 = var_99.validate(var_105)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = module_0.Integer()
    var_6 = False
    var_7 = module_0.Array(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Integer()
    var_11 = module_0.Array(var_10)
    var_12 = 'not a list'
    var_13 = var_11.validate(var_12)
    var_14 = 'type'
    var_15 = any(var_3)
    var_16 = module_0.Integer()
    var_17 = module_0.Array(var_16, min_items=var_14)
    var_18 = []
    var_19 = var_17.validate(var_18)
    var_20 = 'empty'
    var_21 = module_0.Integer()
    var_22 = 2
    var_23 = module_0.Array(var_21, exact_items=var_22)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_23.validate(var_25)
    var_27 = 'exact_items'
    var_28 = module_0.Integer()
    var_29 = module_0.Array(var_28, max_items=var_25)
    var_30 = 1
    var_31 = 2
    var_32 = [var_30, var_31]
    var_33 = var_29.validate(var_32)
    var_34 = 'max_items'
    var_35 = [var_31]
    var_36 = 'type'
    var_37 = [var_6]
    var_38 = 1
    var_39 = [var_38]
    var_40 = 'a'
    var_41 = 'b'
    var_42 = [var_40, var_41]
    var_43 = (var_39, var_32)
    var_44 = (var_39, var_32)
    var_45 = 1
    var_46 = [var_45, var_45]
    var_47 = 'unique_items'
    var_48 = 99
    var_49 = [var_46, var_22]
    var_50 = 1
    var_51 = 2
    var_52 = 3
    var_53 = [var_50, var_51, var_52]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = None



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 2
    var_3 = 10
    var_4 = module_0.Array(min_items=var_2, max_items=var_3)
    var_5 = True
    var_6 = module_0.Array(unique_items=var_5)
    var_7 = 'not a field'
    var_8 = module_0.Array(var_7)
    var_9 = 123
    var_10 = module_0.Array(additional_items=var_9)
    var_11 = 'two'
    var_12 = module_0.Array(min_items=var_11)
    var_13 = None
    var_14 = module_0.Array(max_items=var_13)
    var_15 = 2.5
    var_16 = module_0.Array(max_items=var_15)
    var_17 = 'yes'
    var_18 = module_0.Array(unique_items=var_17)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = module_0.String()
    var_3 = True
    var_4 = module_0.String(allow_blank=var_3)
    var_5 = 10
    var_6 = 5
    var_7 = module_0.String(max_length=var_5, min_length=var_6)
    var_8 = 'not_an_int'
    var_9 = module_0.String(max_length=var_8)
    var_10 = 'not_an_int'
    var_11 = module_0.String(min_length=var_10)
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(pattern=var_12)
    var_14 = 'abc'
    var_15 = module_1.match(var_14)
    var_16 = '\\d+'
    var_17 = module_1.compile(var_16)
    var_18 = module_0.String(pattern=var_17)
    var_19 = 123
    var_20 = module_0.String(pattern=var_19)
    var_21 = 'email'
    var_22 = module_0.String(format=var_21)
    var_23 = 123
    var_24 = module_0.String(format=var_23)
    var_25 = False
    var_26 = module_0.String(trim_whitespace=var_25)
    var_27 = module_0.String(coerce_types=var_25)
    var_28 = 'K'
    var_29 = 'D'
    var_30 = module_0.String()



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.Const(var_0)
    var_3 = 10
    var_4 = module_0.Const(var_3)
    var_5 = var_4.validate(var_3)
    assert var_5 == 10
    var_6 = 'hello'
    var_7 = module_0.Const(var_6)
    var_8 = var_7.validate(var_6)
    assert var_8 == 'hello'
    var_9 = None
    var_10 = module_0.Const(var_9)
    var_11 = var_10.validate(var_9)
    assert var_11 is None
    var_12 = 'expected'
    var_13 = module_0.Const(var_12)
    var_14 = 'actual'
    var_15 = var_13.validate(var_14)
    var_16 = None
    var_17 = module_0.Const(var_16)
    var_18 = 'not_none'
    var_19 = var_17.validate(var_18)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'No Default'
    var_1 = 'Test'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = var_2.get_default_value()
    var_4 = 'Static'
    var_5 = 'hello'
    var_6 = module_0.Field(title=var_4, description=var_1, default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 'hello'
    var_8 = 'Callable'
    var_9 = 42
    var_10 = lambda : var_9
    var_11 = module_0.Field(title=var_8, description=var_1, default=var_10)
    var_12 = var_11.get_default_value()
    assert var_12 == 42
    var_13 = 'None'
    var_14 = None
    var_15 = True
    var_16 = module_0.Field(title=var_13, description=var_1, default=var_14, allow_null=var_15)
    var_17 = var_16.get_default_value()
    assert var_17 is None
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = 'Complex'
    var_22 = module_0.Field(title=var_21, description=var_1, default=var_20)
    var_23 = var_22.get_default_value()
    var_24 = 'Bool'
    var_25 = module_0.Field(title=var_24, description=var_1, default=var_15)
    var_26 = var_25.get_default_value()
    assert var_26 is True



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    var_2 = 'hello'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'hello'
    var_5 = True
    var_6 = module_0.Field(allow_null=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 is None
    var_8 = 42
    var_9 = module_0.Field(default=var_8)
    var_10 = var_9.get_default_value()
    assert var_10 == 42



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 123
    var_5 = module_0.Const(var_4)
    var_6 = True
    var_7 = module_0.Const(var_6)
    var_8 = 'test'
    var_9 = True
    var_10 = module_0.Const(var_8)
    var_11 = var_1.validate(var_8)
    assert var_11 == 'hello'
    var_12 = 'world'
    var_13 = var_1.validate(var_12)
    var_14 = 'not none'
    var_15 = var_3.validate(var_14)
    var_16 = var_3.validate(var_15)
    assert var_16 is None



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'true'
    var_6 = var_0.validate(var_5)
    assert var_6 is True
    var_7 = 'TRUE'
    var_8 = var_0.validate(var_7)
    assert var_8 is True
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = '1'
    var_12 = var_0.validate(var_11)
    assert var_12 is True
    var_13 = 'false'
    var_14 = var_0.validate(var_13)
    assert var_14 is False
    var_15 = 'off'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = '0'
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = var_0.validate(var_1)
    assert var_19 is True
    var_20 = var_0.validate(var_3)
    assert var_20 is False
    var_21 = ''
    var_22 = var_0.validate(var_21)
    assert var_22 is False
    var_23 = None
    var_24 = var_0.validate(var_23)
    var_25 = module_0.Boolean()
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = 'null'
    var_29 = var_25.validate(var_28)
    assert var_29 is None
    var_30 = 'none'
    var_31 = var_25.validate(var_30)
    assert var_31 is None
    var_32 = var_25.validate(var_21)
    assert var_32 is None
    var_33 = 'not-a-boolean'
    var_34 = var_0.validate(var_33)
    var_35 = 2
    var_36 = var_0.validate(var_35)
    var_37 = module_0.Boolean(coerce_types=var_3)
    var_38 = var_37.validate(var_35)
    assert var_38 is True
    var_39 = 'true'
    var_40 = var_37.validate(var_39)
    var_41 = module_0.Boolean()



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.Const(var_0)
    var_3 = 10
    var_4 = module_0.Const(var_3)
    var_5 = var_4.validate(var_3)
    assert var_5 == 10
    var_6 = 5
    var_7 = var_4.validate(var_6)
    var_8 = '10'
    var_9 = var_4.validate(var_8)
    var_10 = None
    var_11 = module_0.Const(var_10)
    var_12 = var_11.validate(var_10)
    assert var_12 is None
    var_13 = 1
    var_14 = var_11.validate(var_13)
    var_15 = 'hello'
    var_16 = module_0.Const(var_15)
    var_17 = var_16.validate(var_15)
    assert var_17 == 'hello'
    var_18 = 'world'
    var_19 = var_16.validate(var_18)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = None
    var_2 = 'text'
    var_3 = None
    var_4 = 'code'
    var_5 = 'type'
    var_6 = {var_4: var_5}
    var_7 = {var_4: var_5}
    var_8 = 'not_matching'
    var_9 = 'index'
    var_10 = 'not_a_type_error'
    var_11 = 0
    var_12 = [var_11]
    var_13 = {var_4: var_10, var_9: var_12}
    var_14 = 'some_value'



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = {}
    var_2 = 'Name'
    var_3 = False
    var_4 = module_0.String(allow_blank=var_3)
    var_5 = 'Age'
    var_6 = 18
    var_7 = module_0.Integer()
    var_8 = 'name'
    var_9 = 'age'
    var_10 = {var_8: var_4, var_9: var_7}
    var_11 = [var_8]
    var_12 = True
    var_13 = module_0.Object(properties=var_10, additional_properties=var_12, required=var_11)
    var_14 = 'extra'
    var_15 = 'Alice'
    var_16 = 25
    var_17 = 'info'
    var_18 = {var_8: var_15, var_9: var_16, var_14: var_17}
    var_19 = var_13.validate(var_18)
    var_20 = 'age'
    var_21 = 30
    var_22 = {var_20: var_21}
    var_23 = var_13.validate(var_22)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Alpha'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Beta'
    var_5 = (var_3, var_4)
    var_6 = 'c'
    var_7 = [var_2, var_5, var_6]
    var_8 = [var_0, var_3]
    var_9 = False
    var_10 = module_0.Choice(choices=var_8)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'
    var_12 = var_10.validate(var_3)
    assert var_12 == 'b'
    var_13 = 'x'
    var_14 = 'X-ray'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = var_17.validate(var_13)
    assert var_18 == 'x'
    var_19 = [var_0, var_3]
    var_20 = module_0.Choice(choices=var_19)
    var_21 = 'z'
    var_22 = var_20.validate(var_21)
    var_23 = [var_21, var_3]
    var_24 = module_0.Choice(choices=var_23)
    var_25 = [var_21, var_3]
    var_26 = True
    var_27 = module_0.Choice(choices=var_25)
    var_28 = None
    var_29 = var_27.validate(var_28)
    assert var_29 is None
    var_30 = [var_21, var_3]
    var_31 = module_0.Choice(choices=var_30, coerce_types=var_26)
    var_32 = ''
    var_33 = var_31.validate(var_32)
    var_34 = [var_32, var_3]
    var_35 = module_0.Choice(choices=var_34, coerce_types=var_26)
    var_36 = ''
    var_37 = var_35.validate(var_36)
    assert var_37 is None



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 3
    var_2 = 5
    var_3 = module_0.String(max_length=var_2, min_length=var_1)
    var_4 = 'abc'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'abc'
    var_6 = 'abcde'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'abcde'
    var_8 = 'ab'
    var_9 = var_3.validate(var_8)
    var_10 = 'abcdef'
    var_11 = var_3.validate(var_10)
    var_12 = True
    var_13 = module_0.String(trim_whitespace=var_12)
    var_14 = '  hello  '
    var_15 = var_13.validate(var_14)
    assert var_15 == 'hello'
    var_16 = False
    var_17 = module_0.String(trim_whitespace=var_16)
    var_18 = var_17.validate(var_14)
    assert var_18 == '  hello  '
    var_19 = module_0.String()
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = module_0.String()
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = module_0.String(allow_blank=var_16)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_0.Array(var_2)
    var_5 = 1
    var_6 = 'two'
    var_7 = 3.0
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.serialize(var_8)
    var_10 = module_0.Integer()
    var_11 = module_0.Array(var_10)
    var_12 = '1'
    var_13 = '2'
    var_14 = '3'
    var_15 = [var_12, var_13, var_14]
    var_16 = var_11.serialize(var_15)
    var_17 = 100
    var_18 = 200
    var_19 = [var_17, var_18]
    var_20 = module_0.Decimal()
    var_21 = module_0.Array(var_20)
    var_22 = '1.5'
    var_23 = '2.7'
    var_24 = module_0.Boolean()
    var_25 = module_0.Array(var_24)
    var_26 = True
    var_27 = False
    var_28 = [var_26, var_27, var_12]
    var_29 = var_25.serialize(var_28)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'Option C'
    var_4 = (var_2, var_3)
    var_5 = [var_0, var_1, var_4]
    var_6 = False
    var_7 = module_0.Choice(choices=var_5)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_1)
    assert var_9 == 'b'
    var_10 = var_7.validate(var_2)
    assert var_10 == 'c'
    var_11 = 'z'
    var_12 = var_7.validate(var_11)
    var_13 = None
    var_14 = var_7.validate(var_13)
    var_15 = True
    var_16 = module_0.Choice(choices=var_5)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = ''
    var_20 = var_7.validate(var_19)
    var_21 = module_0.Choice(choices=var_5, coerce_types=var_15)
    var_22 = ''
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = []
    var_25 = module_0.Choice(choices=var_24)
    var_26 = 'a'
    var_27 = var_25.validate(var_26)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_0.Array(var_2)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.serialize(var_8)
    var_10 = [var_5, var_6, var_7]
    var_11 = 4
    var_12 = 6
    var_13 = [var_6, var_11, var_12]
    var_14 = 10
    var_15 = 20
    var_16 = [var_14, var_15]
    var_17 = 'val1'
    var_18 = 'valron2'
    var_19 = [var_17, var_18]
    var_20 = [var_14, var_15]
    var_21 = module_0.Integer()
    var_22 = module_0.Float()
    var_23 = [var_21, var_22]
    var_24 = module_0.Array(var_23)
    var_25 = 2.5
    var_26 = [var_5, var_25]
    var_27 = var_24.serialize(var_26)
    var_28 = module_0.Array(var_2)
    var_29 = 'a'
    var_30 = {var_29: var_5}
    var_31 = [var_5, var_6]
    var_32 = [var_30, var_31]
    var_33 = var_28.serialize(var_32)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'a'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'b'
    var_7 = 'v1'
    var_8 = 'Display 1'
    var_9 = (var_7, var_8)
    var_10 = 'v2'
    var_11 = 'Display 2'
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = module_0.Choice(choices=var_13)
    var_15 = var_14.validate(var_7)
    assert var_15 == 'v1'
    var_16 = var_14.validate(var_10)
    assert var_16 == 'v2'
    var_17 = 'c'
    var_18 = var_4.validate(var_17)
    var_19 = [var_17]
    var_20 = module_0.Choice(choices=var_19)
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = [var_21]
    var_24 = True
    var_25 = module_0.Choice(choices=var_23)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = [var_21]
    var_29 = module_0.Choice(choices=var_28, coerce_types=var_24)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    assert var_31 is None
    var_32 = [var_21]
    var_33 = module_0.Choice(choices=var_32, coerce_types=var_24)
    var_34 = ''
    var_35 = var_33.validate(var_34)
    var_36 = [var_34]
    var_37 = module_0.Choice(choices=var_36, coerce_types=var_3)
    var_38 = ''
    var_39 = var_37.validate(var_38)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = None
    var_2 = 10
    var_3 = 'string'
    var_4 = None
    var_5 = 'union'
    var_6 = 'invalid'
    var_7 = 'not_a_type_error'
    var_8 = 'invalid'



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    var_2 = 42
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 42
    var_5 = 'hello'
    var_6 = module_0.Field(default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 'hello'
    var_8 = None
    var_9 = module_0.Field(default=var_8)
    var_10 = var_9.get_default_value()
    assert var_10 is None
    var_11 = 'dynamic'
    var_12 = lambda : var_11
    var_13 = module_0.Field(default=var_12)
    var_14 = var_13.get_default_value()
    assert var_14 == 'dynamic'



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'ValidationError'
    var_1 = 'Message'
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'Name'
    var_5 = module_0.String()
    var_6 = 'Age'
    var_7 = module_0.Integer()
    var_8 = {var_2: var_5, var_3: var_7}
    var_9 = [var_2]
    var_10 = module_0.Object(properties=var_8, required=var_9)
    var_11 = 'extra'
    var_12 = 'Alice'
    var_13 = 30
    var_14 = 'allowed'
    var_15 = {var_2: var_12, var_3: var_13, var_11: var_14}
    var_16 = var_10.validate(var_15)
    var_17 = {var_3: var_13}
    var_18 = var_10.validate(var_17)
    var_19 = 'required'
    var_20 = [var_2]
    var_21 = 'bad_key'
    var_22 = 123
    var_23 = {var_21: var_22}
    var_24 = 'invalid_property'
    var_25 = 'bad_key'
    var_26 = [var_25]
    var_27 = 1
    var_28 = module_0.Object(max_properties=var_27)
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 1
    var_32 = 2
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = var_28.validate(var_33)
    var_35 = 'max_properties'
    var_36 = module_0.Object(min_properties=var_27)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'empty'
    var_40 = 'a'
    var_41 = module_0.String()
    var_42 = {var_40: var_41}
    var_43 = False
    var_44 = module_0.Object(properties=var_42, additional_properties=var_43)
    var_45 = 'a'
    var_46 = 'b'
    var_47 = 'val'
    var_48 = 'forbidden'
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = var_44.validate(var_49)
    var_51 = 'b'
    var_52 = '^id_.*'
    var_53 = module_0.Integer()
    var_54 = {var_52: var_53}
    var_55 = module_0.Object(pattern_properties=var_54)
    var_56 = 'id_123'
    var_57 = 456
    var_58 = {var_56: var_57}
    var_59 = var_55.validate(var_58)
    var_60 = 'id_123'
    var_61 = 'not_an_int'
    var_62 = {var_60: var_61}
    var_63 = var_55.validate(var_62)
    var_64 = [var_56]
    var_65 = module_0.Object()
    var_66 = 123
    var_67 = 'value'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)
    var_70 = 'invalid_key'
    var_71 = 123
    var_72 = [var_71]



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = '\n    Comprehensive test function as requested by the signature.\n    Note: This implementation assumes the presence of ValidationError and Uniqueness \n    in the global scope as per the provided snippet.\n    '
    var_1 = module_0.Integer()
    var_2 = module_0.Array(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = module_0.Array(min_items=var_4)
    var_9 = 1
    var_10 = [var_9]
    var_11 = var_8.validate(var_10)
    var_12 = module_0.Array(max_items=var_11)
    var_13 = 1
    var_14 = 2
    var_15 = [var_13, var_14]
    var_16 = var_12.validate(var_15)
    var_17 = module_0.Array()
    var_18 = 'not a list'
    var_19 = var_17.validate(var_18)
    var_20 = True
    var_21 = module_0.Array()
    var_22 = None
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = 1
    var_25 = [var_24]



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    var_2 = 'hello'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'hello'
    var_5 = True
    var_6 = module_0.Field(allow_null=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 is None
    var_8 = 'key'
    var_9 = 2
    var_10 = 3
    var_11 = [var_5, var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = module_0.Field(default=var_12)
    var_14 = var_13.get_default_value()



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = module_0.String()
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'
    var_4 = '  hello  '
    var_5 = var_1.validate(var_4)
    assert var_5 == 'hello'
    var_6 = True
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.String()
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String(allow_blank=var_6, coerce_types=var_6)
    var_15 = var_14.validate(var_8)
    assert var_15 == ''
    var_16 = ''
    var_17 = var_14.validate(var_16)
    assert var_17 == ''
    var_18 = module_0.String(allow_blank=var_10)
    var_19 = ''
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = 123
    var_23 = var_21.validate(var_22)
    var_24 = 3
    var_25 = 5
    var_26 = module_0.String(max_length=var_25, min_length=var_24)
    var_27 = 'abc'
    var_28 = var_26.validate(var_27)
    assert var_28 == 'abc'
    var_29 = 'abcd'
    var_30 = var_26.validate(var_29)
    assert var_30 == 'abcd'
    var_31 = 'abcde'
    var_32 = var_26.validate(var_31)
    assert var_32 == 'abcde'
    var_33 = 'ab'
    var_34 = var_26.validate(var_33)
    var_35 = 'abcdef'
    var_36 = var_26.validate(var_35)
    var_37 = '^[a-z]+$'
    var_38 = module_0.String(pattern=var_37)
    var_39 = var_38.validate(var_27)
    assert var_39 == 'abc'
    var_40 = 'ABC'
    var_41 = var_38.validate(var_40)
    var_42 = '\\d+'
    var_43 = module_1.compile(var_42)
    var_44 = module_0.String(pattern=var_43)
    var_45 = '123'
    var_46 = var_44.validate(var_45)
    assert var_46 == '123'
    var_47 = 'abc'
    var_48 = var_44.validate(var_47)
    var_49 = module_0.String()
    var_50 = 'a\x00b'
    var_51 = var_49.validate(var_50)
    assert var_51 == 'ab'
    var_52 = 'email'
    var_53 = module_0.String(format=var_52)
    var_54 = 'test@example.com'
    var_55 = var_53.validate(var_54)
    assert var_55 == 'test@example.com'
    var_56 = module_0.String(allow_blank=var_6, coerce_types=var_6)
    var_57 = var_56.validate(var_8)
    assert var_57 == ''
    var_58 = module_0.String(trim_whitespace=var_10)
    var_59 = var_58.validate(var_4)
    assert var_59 == '  hello  '



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'Option B'
    var_3 = (var_1, var_2)
    var_4 = 'C'
    var_5 = [var_0, var_3, var_4]
    var_6 = False
    var_7 = module_0.Choice(choices=var_5)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'A'
    var_9 = var_7.validate(var_1)
    assert var_9 == 'B'
    var_10 = var_7.validate(var_4)
    assert var_10 == 'C'
    var_11 = 'D'
    var_12 = var_7.validate(var_11)
    var_13 = None
    var_14 = var_7.validate(var_13)
    var_15 = True
    var_16 = module_0.Choice(choices=var_5)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = [var_13]
    var_20 = module_0.Choice(choices=var_19, coerce_types=var_15)
    var_21 = ''
    var_22 = var_20.validate(var_21)
    assert var_22 is None
    var_23 = [var_13]
    var_24 = module_0.Choice(choices=var_23)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    var_27 = []
    var_28 = module_0.Choice(choices=var_27)
    var_29 = 'A'
    var_30 = var_28.validate(var_29)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 1
    var_5 = True
    var_6 = module_0.Const(var_4)
    var_7 = 'hello'
    var_8 = module_0.Const(var_7)
    var_9 = var_8.validate(var_7)
    assert var_9 == 'hello'
    var_10 = 'world'
    var_11 = var_8.validate(var_10)
    var_12 = None
    var_13 = module_0.Const(var_12)
    var_14 = 'not_null'
    var_15 = var_13.validate(var_14)
    var_16 = 'only_null'
    var_17 = 5
    var_18 = module_0.Const(var_17)
    var_19 = 10
    var_20 = var_18.validate(var_19)
    var_21 = 'const'



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = module_0.String()
    var_2 = 'Age'
    var_3 = 18
    var_4 = module_0.Integer()
    var_5 = 'Active'
    var_6 = module_0.Boolean()
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'active'
    var_10 = {var_7: var_1, var_8: var_4, var_9: var_6}
    var_11 = [var_7, var_9]
    var_12 = module_0.Object(properties=var_10, required=var_11)
    var_13 = 'John Doe'
    var_14 = 25
    var_15 = True
    var_16 = {var_7: var_13, var_8: var_14, var_9: var_15}
    var_17 = var_12.validate(var_16)
    var_18 = 'Jane'
    var_19 = False
    var_20 = {var_7: var_18, var_9: var_19}
    var_21 = var_12.validate(var_20)
    var_22 = 30
    var_23 = {var_8: var_22}
    var_24 = var_12.validate(var_23)
    var_25 = 'not'
    var_26 = 'a'
    var_27 = 'dict'
    var_28 = [var_25, var_26, var_27]
    var_29 = var_12.validate(var_28)
    var_30 = 123
    var_31 = {var_29: var_30, var_9: var_15}
    var_32 = var_12.validate(var_31)
    var_33 = [var_29]
    var_34 = {var_29: var_1}
    var_35 = module_0.Object(properties=var_34, additional_properties=var_19)
    var_36 = 'unknown'
    var_37 = 'Test'
    var_38 = 'value'
    var_39 = {var_29: var_37, var_36: var_38}
    var_40 = var_35.validate(var_39)
    var_41 = 'invalid_property'
    var_42 = {var_29: var_1}
    var_43 = module_0.Integer()
    var_44 = module_0.Object(properties=var_42, additional_properties=var_43)
    var_45 = 'score'
    var_46 = 100
    var_47 = {var_29: var_37, var_45: var_46}
    var_48 = 'not_a_number'
    var_49 = {var_29: var_37, var_45: var_48}
    var_50 = var_44.validate(var_47)
    var_51 = var_44.validate(var_49)
    var_52 = {var_29: var_1}
    var_53 = 2
    var_54 = module_0.Object(properties=var_52, min_properties=var_53, max_properties=var_53)
    var_55 = 'name'
    var_56 = 'Too few'
    var_57 = {var_55: var_56}
    var_58 = var_54.validate(var_57)
    var_59 = 'min_properties'
    var_60 = 'empty'
    var_61 = 'name'
    var_62 = 'b'
    var_63 = 'c'
    var_64 = 'A'
    var_65 = 'B'
    var_66 = 'C'
    var_67 = {var_61: var_64, var_62: var_65, var_63: var_66}
    var_68 = var_54.validate(var_67)
    var_69 = 'max_properties'
    var_70 = '^attr_\\d+$'
    var_71 = module_0.Integer()
    var_72 = {var_70: var_71}
    var_73 = {var_65: var_1}
    var_74 = module_0.Object(properties=var_73, pattern_properties=var_72)
    var_75 = 'attr_1'
    var_76 = 50
    var_77 = {var_65: var_37, var_75: var_76}
    var_78 = 'not_int'
    var_79 = {var_65: var_37, var_75: var_78}
    var_80 = var_74.validate(var_77)
    var_81 = var_74.validate(var_79)
    var_82 = module_0.String()
    var_83 = {var_65: var_82}
    var_84 = module_0.Object(properties=var_83)
    var_85 = None
    var_86 = {var_65: var_85}
    var_87 = var_84.validate(var_86)
    var_88 = None
    var_89 = var_84.validate(var_88)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = True
    var_6 = False
    var_7 = None
    var_8 = 'not a list'
    var_9 = 1
    var_10 = [var_9]
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = [var_11, var_12, var_13]
    var_15 = 1
    var_16 = [var_15]
    var_17 = [var_5, var_13]
    var_18 = []
    var_19 = True
    var_20 = 1
    var_21 = [var_20, var_20]
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_22, var_23]
    var_25 = 99
    var_26 = [var_19, var_13]
    var_27 = 1
    var_28 = [var_27]



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 10
    var_2 = var_0.validate(var_1)
    assert var_2 == 10
    var_3 = 10.5
    var_4 = var_0.validate(var_3)
    var_5 = '10.5'
    var_6 = var_0.validate(var_5)
    var_7 = True
    var_8 = module_0.Number()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Number()
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = ''
    var_16 = var_8.validate(var_15)
    assert var_16 is None
    var_17 = True
    var_18 = var_0.validate(var_17)
    var_19 = 10.5
    var_20 = module_0.Number(coerce_types=var_11)
    var_21 = '10'
    var_22 = var_20.validate(var_21)
    var_23 = var_0.validate(var_21)
    var_24 = 'nan'
    var_25 = float(var_24)
    var_26 = var_0.validate(var_25)
    var_27 = 5
    var_28 = 2
    var_29 = module_0.Number(minimum=var_27, exclusive_minimum=var_28)
    var_30 = var_29.validate(var_27)
    assert var_30 == 5
    var_31 = 2.1
    var_32 = var_29.validate(var_31)
    var_33 = 4
    var_34 = var_29.validate(var_33)
    var_35 = 2
    var_36 = var_29.validate(var_35)
    var_37 = 15
    var_38 = module_0.Number(maximum=var_35, exclusive_maximum=var_37)
    var_39 = var_38.validate(var_35)
    assert var_39 == 10
    var_40 = 11
    var_41 = var_38.validate(var_40)
    var_42 = 15
    var_43 = var_38.validate(var_42)
    var_44 = module_0.Number(multiple_of=var_27)
    var_45 = var_44.validate(var_42)
    assert var_45 == 10



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    assert var_1 is True
    assert var_1 is False
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'true'
    var_6 = 'TRUE'
    var_7 = 'on'
    var_8 = '1'
    var_9 = [var_5, var_6, var_7, var_8, var_1]
    var_10 = 'false'
    var_11 = 'FALSE'
    var_12 = 'off'
    var_13 = '0'
    var_14 = ''
    var_15 = [var_10, var_11, var_12, var_13, var_3, var_14]
    var_16 = None
    var_17 = var_0.validate(var_16)
    var_18 = module_0.Boolean()
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 'null'
    var_22 = var_18.validate(var_21)
    assert var_22 is None
    var_23 = 'none'
    var_24 = var_18.validate(var_23)
    assert var_24 is None
    var_25 = var_18.validate(var_14)
    assert var_25 is None
    var_26 = module_0.Boolean(coerce_types=var_3)
    var_27 = 'true'
    var_28 = var_26.validate(var_27)
    var_29 = 'maybe'
    var_30 = var_0.validate(var_29)
    var_31 = 123
    var_32 = var_0.validate(var_31)



