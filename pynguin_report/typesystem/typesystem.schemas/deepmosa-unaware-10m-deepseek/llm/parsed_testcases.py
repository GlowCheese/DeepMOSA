####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'active'
    var_7 = module_1.String()
    var_8 = module_1.Integer()
    var_9 = module_1.Boolean()
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = module_0.Schema(var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = True
    var_15 = {var_4: var_12, var_5: var_13, var_6: var_14}
    var_16 = var_11.serialize(var_15)
    var_17 = 'Jane'
    var_18 = 25
    var_19 = False
    var_20 = 'Bob'
    var_21 = {var_4: var_20}
    var_22 = var_11.serialize(var_21)
    var_23 = 'Alice'
    var_24 = 'street'
    var_25 = 'city'
    var_26 = module_1.String()
    var_27 = module_1.String()
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = 'address'
    var_31 = module_1.String()
    var_32 = {var_4: var_31, var_30: var_29}
    var_33 = module_0.Schema(var_32)
    var_34 = 'Test'
    var_35 = 'Main St'
    var_36 = 'Metropolis'
    var_37 = {var_24: var_35, var_25: var_36}
    var_38 = {var_4: var_34, var_30: var_37}
    var_39 = var_33.serialize(var_38)
    var_40 = 'data'
    var_41 = 'value'
    var_42 = {var_40: var_41}
    var_43 = var_11.serialize(var_42)
    var_44 = {}
    var_45 = module_0.Schema(var_44)
    var_46 = {}
    var_47 = var_45.serialize(var_46)
    var_48 = 'extra'
    var_49 = module_1.String()
    var_50 = module_1.String()
    var_51 = {var_4: var_49, var_48: var_50}
    var_52 = module_0.Schema(var_51)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_1.Schema(var_5)
    var_7 = 'allow_null'
    var_8 = hasattr(var_6, var_7)
    var_9 = module_0.Field(read_only=var_1)
    var_10 = 'id'
    var_11 = 'title'
    var_12 = module_0.Field()
    var_13 = {var_10: var_9, var_11: var_12}
    var_14 = module_1.Schema(var_13)
    var_15 = 'default_value'
    var_16 = module_0.Field(default=var_15)
    var_17 = 'email'
    var_18 = module_0.Field()
    var_19 = {var_3: var_16, var_17: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 'default1'
    var_22 = module_0.Field(default=var_21)
    var_23 = module_0.Field(allow_null=var_1)
    var_24 = 'field1'
    var_25 = 'field2'
    var_26 = {var_24: var_22, var_25: var_23}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = module_1.Schema(var_28)
    var_30 = 'Test schema'
    var_31 = module_1.Schema(var_5)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = 0
    var_5 = 150
    var_6 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    var_15 = module_0.String()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 'not a dict'
    var_22 = var_8.validate(var_21)
    var_23 = 1
    var_24 = 'name'
    var_25 = 'value'
    var_26 = 'John'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = var_8.validate(var_27)
    var_29 = 'name'
    var_30 = 'John'
    var_31 = {var_29: var_30}
    var_32 = var_8.validate(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'VeryLongNameExceedsLimit'
    var_36 = 30
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = var_8.validate(var_37)
    var_39 = 'id'
    var_40 = module_0.Integer()
    var_41 = module_0.String()
    var_42 = {var_39: var_40, var_33: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = 'Alice'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)
    var_47 = 'active'
    var_48 = module_0.String()
    var_49 = module_0.Field(default=var_17)
    var_50 = {var_33: var_48, var_47: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = 'Bob'
    var_53 = {var_33: var_52}
    var_54 = var_51.validate(var_53)
    var_55 = {}
    var_56 = var_8.validate(var_55)
    var_57 = 'address'
    var_58 = 'street'
    var_59 = 'city'
    var_60 = module_0.String()
    var_61 = module_0.String()
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_1.Schema(var_62)
    var_64 = {var_57: var_63}
    var_65 = module_1.Schema(var_64)
    var_66 = '123 Main'
    var_67 = 'Boston'
    var_68 = {var_58: var_66, var_59: var_67}
    var_69 = {var_57: var_68}
    var_70 = var_65.validate(var_69)
    var_71 = 'address'
    var_72 = 'street'
    var_73 = 123
    var_74 = {var_72: var_73}
    var_75 = {var_71: var_74}
    var_76 = var_65.validate(var_75)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    var_8 = 'id'
    var_9 = {var_2: var_0, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'default_value'
    var_12 = module_0.Field(default=var_11)
    var_13 = 'optional'
    var_14 = {var_2: var_0, var_13: var_12}
    var_15 = module_1.Schema(var_14)
    var_16 = 'required_field'
    var_17 = 'read_only_field'
    var_18 = 'default_field'
    var_19 = {var_16: var_0, var_17: var_7, var_18: var_12}
    var_20 = module_1.Schema(var_19)
    var_21 = module_1.Schema(var_4)
    var_22 = {}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Test schema'
    var_25 = module_1.Schema(var_4)
    var_26 = 'description'
    var_27 = hasattr(var_25, var_26)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    var_8 = 'id'
    var_9 = {var_2: var_0, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'default_value'
    var_12 = module_0.Field(default=var_11)
    var_13 = 'optional'
    var_14 = {var_2: var_0, var_13: var_12}
    var_15 = module_1.Schema(var_14)
    var_16 = 'required_field'
    var_17 = 'read_only_field'
    var_18 = 'default_field'
    var_19 = module_0.Field(read_only=var_6)
    var_20 = 0
    var_21 = module_0.Field(default=var_20)
    var_22 = {var_16: var_0, var_17: var_19, var_18: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Test schema'
    var_25 = module_1.Schema(var_4)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = module_1.Field()
    var_17 = 'error_field'
    var_18 = module_0.Reference(var_17, var_15)
    var_19 = 'Invalid'
    var_20 = 'invalid'
    var_21 = module_2.Message(text=var_19, code=var_20)
    var_22 = ()
    var_23 = [var_21]
    var_24 = module_2.ValidationError(messages=var_23)
    var_25 = 'some_value'
    var_26 = var_18.validate(var_25)
    var_27 = module_0.Definitions()
    var_28 = 'missing'
    var_29 = module_0.Reference(var_28, var_27)
    var_30 = 'value'
    var_31 = var_29.validate(var_30)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = {var_1: var_0}
    var_8 = True
    var_9 = module_1.Schema(var_7)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = {var_1: var_0}
    var_13 = False
    var_14 = module_1.Schema(var_12)
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = {var_15: var_0}
    var_18 = module_1.Schema(var_17)
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = {var_19: var_0}
    var_22 = module_1.Schema(var_21)
    var_23 = 123
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = {var_23: var_0}
    var_28 = module_1.Schema(var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = 'default_value'
    var_32 = module_0.Field(default=var_31)
    var_33 = {var_29: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Field(read_only=var_8)
    var_38 = {var_29: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = 'ignored'
    var_41 = {var_29: var_40}
    var_42 = var_39.validate(var_41)
    var_43 = module_0.Field()
    var_44 = 'nested'
    var_45 = {var_44: var_43}
    var_46 = module_1.Schema(var_45)
    var_47 = 'nested'
    var_48 = None
    var_49 = {var_47: var_48}
    var_50 = var_46.validate(var_49)
    var_51 = module_0.Field()
    var_52 = module_0.Field()
    var_53 = 'field1'
    var_54 = 'field2'
    var_55 = {var_53: var_51, var_54: var_52}
    var_56 = module_1.Schema(var_55)
    var_57 = 123
    var_58 = 'extra'
    var_59 = 'bad key'
    var_60 = 'unexpected'
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = var_56.validate(var_61)
    var_63 = {var_57: var_0}
    var_64 = module_1.Schema(var_63)
    var_65 = {var_57: var_59}
    var_66 = var_64.validate(var_65)
    var_67 = {var_57: var_0}
    var_68 = module_1.Schema(var_67)
    var_69 = {var_57: var_59}



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'name'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'id'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 123
    var_32 = {var_28: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = 'default_value'
    var_35 = module_0.Field(default=var_34)
    var_36 = 'optional'
    var_37 = {var_36: var_35}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field()
    var_42 = {var_22: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = 'John'
    var_45 = {var_22: var_44}
    var_46 = var_43.validate(var_45)
    var_47 = module_0.Field()
    var_48 = ()
    var_49 = 'Invalid'
    var_50 = 'invalid'
    var_51 = []
    var_52 = module_2.Message(text=var_49, code=var_50, index=var_51)
    var_53 = [var_52]
    var_54 = module_2.ValidationError(messages=var_53)
    var_55 = 'nested'
    var_56 = {var_55: var_47}
    var_57 = module_1.Schema(var_56)
    var_58 = 'nested'
    var_59 = 'value'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)
    var_62 = module_0.Field()
    var_63 = 'field1'
    var_64 = 'field2'
    var_65 = {var_63: var_62, var_64: var_62}
    var_66 = module_1.Schema(var_65)
    var_67 = 2
    var_68 = 'invalid key'
    var_69 = {var_67: var_68}
    var_70 = var_66.validate(var_69)
    var_71 = module_0.Field(allow_null=var_67)
    var_72 = 'data'
    var_73 = {var_72: var_71}
    var_74 = module_1.Schema(var_73)
    var_75 = {var_72: var_69}
    var_76 = var_74.validate(var_75)
    var_77 = 'age'
    var_78 = module_0.Field()
    var_79 = {var_77: var_78}
    var_80 = module_1.Schema(var_79)
    var_81 = 'person'
    var_82 = {var_81: var_80}
    var_83 = module_1.Schema(var_82)
    var_84 = 30
    var_85 = {var_77: var_84}
    var_86 = {var_81: var_85}
    var_87 = var_83.validate(var_86)
    var_88 = module_0.Field()
    var_89 = 'Error'
    var_90 = 'error'
    var_91 = []
    var_92 = module_2.Message(text=var_89, code=var_90, index=var_91)
    var_93 = [var_92]
    var_94 = module_2.ValidationError(messages=var_93)
    var_95 = (var_69, var_94)
    var_96 = 'test'
    var_97 = {var_96: var_88}
    var_98 = module_1.Schema(var_97)
    var_99 = 'test'
    var_100 = 'value'
    var_101 = {var_99: var_100}
    var_102 = var_98.validate(var_101)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Reference(var_3, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = module_1.Field()
    var_17 = ()
    var_18 = 'Target error'
    var_19 = 'target_error'
    var_20 = module_2.Message(text=var_18, code=var_19)
    var_21 = [var_20]
    var_22 = module_2.ValidationError(messages=var_21)
    var_23 = 'target2'
    var_24 = module_0.Reference(var_23, var_15)
    var_25 = 'bad_value'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Definitions()
    var_28 = 'missing'
    var_29 = module_0.Reference(var_28, var_27)
    var_30 = 'value'
    var_31 = var_29.validate(var_30)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_1.Field()
    var_16 = 'nested'
    var_17 = module_0.Reference(var_16, var_0)
    var_18 = 'test'
    var_19 = var_17.validate(var_18)
    assert var_19 == 'TEST'
    var_20 = module_1.Field()
    var_21 = ()
    var_22 = 'Invalid'
    var_23 = 'invalid'
    var_24 = module_2.Message(text=var_22, code=var_23)
    var_25 = [var_24]
    var_26 = module_2.ValidationError(messages=var_25)
    var_27 = 'error_field'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = 'bad_value'
    var_30 = var_28.validate(var_29)
    var_31 = 'missing'
    var_32 = module_0.Reference(var_31, var_0)
    var_33 = 'value'
    var_34 = var_32.validate(var_33)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.Field(allow_null=var_7)
    var_9 = {var_2: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Field()
    var_14 = {var_12: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = 'not a dict'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Field()
    var_19 = {var_17: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 123
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = module_0.Field()
    var_26 = {var_22: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = 'default_name'
    var_31 = module_0.Field(default=var_30)
    var_32 = {var_29: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Field(read_only=var_28)
    var_37 = {var_29: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'test'
    var_40 = {var_29: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = module_0.Field()
    var_43 = {var_29: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = 'John'
    var_46 = {var_29: var_45}
    var_47 = var_44.validate(var_46)
    var_48 = module_0.Field()
    var_49 = 'age'
    var_50 = {var_49: var_48}
    var_51 = module_1.Schema(var_50)
    var_52 = 'age'
    var_53 = 'not a number'
    var_54 = {var_52: var_53}
    var_55 = var_51.validate(var_54)
    var_56 = module_0.Field()
    var_57 = module_0.Field()
    var_58 = 'field1'
    var_59 = 'field2'
    var_60 = {var_58: var_56, var_59: var_57}
    var_61 = module_1.Schema(var_60)
    var_62 = 123
    var_63 = 'extra'
    var_64 = 'invalid'
    var_65 = 'field'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_61.validate(var_66)
    var_68 = module_0.Field()
    var_69 = 'nested'
    var_70 = {var_69: var_68}
    var_71 = module_1.Schema(var_70)
    var_72 = 'data'
    var_73 = {var_72: var_71}
    var_74 = module_1.Schema(var_73)
    var_75 = 'value'
    var_76 = {var_69: var_75}
    var_77 = {var_72: var_76}
    var_78 = var_74.validate(var_77)
    var_79 = 'default1'
    var_80 = module_0.Field(default=var_79)
    var_81 = module_0.Field(read_only=var_62)
    var_82 = module_0.Field()
    var_83 = 'f1'
    var_84 = 'f2'
    var_85 = 'f3'
    var_86 = {var_83: var_80, var_84: var_81, var_85: var_82}
    var_87 = module_1.Schema(var_86)
    var_88 = 'value3'
    var_89 = {var_85: var_88}
    var_90 = var_87.validate(var_89)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = 'name'
    var_17 = module_1.Field()
    var_18 = {var_16: var_17}
    var_19 = module_0.Schema(var_18)
    var_20 = 'person'
    var_21 = module_0.Reference(var_20, var_15)
    var_22 = 'John'
    var_23 = {var_16: var_22}
    var_24 = var_21.validate(var_23)
    var_25 = module_0.Definitions()
    var_26 = False
    var_27 = module_1.Field(allow_null=var_26)
    var_28 = 'strict_field'
    var_29 = module_0.Reference(var_28, var_25)
    var_30 = None
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = module_1.Field()
    var_34 = 'inner'
    var_35 = module_0.Reference(var_34, var_32)
    var_36 = 'middle'
    var_37 = module_0.Reference(var_36, var_32)
    var_38 = 'test'
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Definitions()
    var_41 = 'missing'
    var_42 = module_0.Reference(var_41, var_40)
    var_43 = 'any'
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'valid'
    var_19 = 'value'
    var_20 = {var_17: var_19, var_18: var_19}
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Field()
    var_23 = 'required_field'
    var_24 = {var_23: var_22}
    var_25 = module_1.Schema(var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Field(read_only=var_26)
    var_29 = 'read_only'
    var_30 = {var_29: var_28}
    var_31 = module_1.Schema(var_30)
    var_32 = 'other'
    var_33 = 'value'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 'default_value'
    var_37 = module_0.Field(default=var_36)
    var_38 = 'with_default'
    var_39 = {var_38: var_37}
    var_40 = module_1.Schema(var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = module_0.Field()
    var_44 = 'name'
    var_45 = {var_44: var_43}
    var_46 = module_1.Schema(var_45)
    var_47 = 'John'
    var_48 = {var_44: var_47}
    var_49 = var_46.validate(var_48)
    var_50 = module_0.Field()
    var_51 = 'age'
    var_52 = {var_51: var_50}
    var_53 = module_1.Schema(var_52)
    var_54 = 'age'
    var_55 = None
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = module_0.Field()
    var_59 = 'required'
    var_60 = {var_59: var_58}
    var_61 = module_1.Schema(var_60)
    var_62 = 1
    var_63 = 'other'
    var_64 = 'invalid'
    var_65 = 'value'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_61.validate(var_66)
    var_68 = module_0.Field()
    var_69 = 'nested'
    var_70 = {var_69: var_68}
    var_71 = module_1.Schema(var_70)
    var_72 = 'parent'
    var_73 = {var_72: var_71}
    var_74 = module_1.Schema(var_73)
    var_75 = {var_69: var_33}
    var_76 = {var_72: var_75}
    var_77 = var_74.validate(var_76)
    var_78 = {}
    var_79 = module_1.Schema(var_78)
    var_80 = 'extra'
    var_81 = 'ignored'
    var_82 = {var_80: var_81}
    var_83 = var_79.validate(var_82)
    var_84 = module_0.Field(allow_null=var_62)
    var_85 = 'nullable'
    var_86 = {var_85: var_84}
    var_87 = module_1.Schema(var_86)
    var_88 = {var_85: var_64}
    var_89 = var_87.validate(var_88)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Definitions()
    var_15 = module_1.Field()
    var_16 = ()
    var_17 = []
    var_18 = module_2.ValidationError(messages=var_17)
    var_19 = module_0.Reference(var_13, var_14)
    var_20 = 'invalid'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Definitions()
    var_23 = module_1.Field()
    var_24 = module_0.Reference(var_21, var_22)
    var_25 = 'test'
    var_26 = var_24.validate(var_25)
    assert var_26 == 'test'
    var_27 = module_0.Definitions()
    var_28 = 'missing'
    var_29 = module_0.Reference(var_28, var_27)
    var_30 = 'value'
    var_31 = var_29.validate(var_30)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Field()
    var_11 = {var_2: var_10}
    var_12 = module_1.Schema(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Field()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = module_0.Field()
    var_22 = {var_13: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Field()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 1
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Field()
    var_34 = 'required_field'
    var_35 = {var_34: var_33}
    var_36 = module_1.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = module_0.Field(read_only=var_17)
    var_40 = module_0.Field()
    var_41 = 'read_only'
    var_42 = 'regular'
    var_43 = {var_41: var_39, var_42: var_40}
    var_44 = module_1.Schema(var_43)
    var_45 = 'value'
    var_46 = {var_42: var_45}
    var_47 = var_44.validate(var_46)
    var_48 = 'default_value'
    var_49 = module_0.Field(default=var_48)
    var_50 = 'field'
    var_51 = {var_50: var_49}
    var_52 = module_1.Schema(var_51)
    var_53 = {}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Field()
    var_56 = ()
    var_57 = 'Nested error'
    var_58 = 'nested'
    var_59 = []
    var_60 = module_2.Message(text=var_57, code=var_58, index=var_59)
    var_61 = [var_60]
    var_62 = module_2.ValidationError(messages=var_61)
    var_63 = {var_58: var_55}
    var_64 = module_1.Schema(var_63)
    var_65 = 'nested'
    var_66 = 'value'
    var_67 = {var_65: var_66}
    var_68 = var_64.validate(var_67)
    var_69 = module_0.Field()
    var_70 = module_0.Field()
    var_71 = 'field1'
    var_72 = 'field2'
    var_73 = {var_71: var_69, var_72: var_70}
    var_74 = module_1.Schema(var_73)
    var_75 = {}
    var_76 = var_74.validate(var_75)
    var_77 = {msg.code for msg in e.messages()}
    var_78 = module_0.Field()
    var_79 = {var_75: var_78}
    var_80 = module_1.Schema(var_79)
    var_81 = 'john'
    var_82 = {var_75: var_81}
    var_83 = var_80.validate(var_82)
    var_84 = module_0.Field()
    var_85 = 'key'
    var_86 = {var_85: var_84}
    var_87 = module_1.Schema(var_86)
    var_88 = {var_85: var_45}



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = module_1.Field()
    var_17 = 'error_field'
    var_18 = module_0.Reference(var_17, var_15)
    var_19 = 'Invalid'
    var_20 = 'custom'
    var_21 = []
    var_22 = module_2.Message(text=var_19, code=var_20, index=var_21)
    var_23 = ()
    var_24 = [var_22]
    var_25 = module_2.ValidationError(messages=var_24)
    var_26 = 'any_value'
    var_27 = var_18.validate(var_26)
    var_28 = module_0.Definitions()
    var_29 = 'missing'
    var_30 = module_0.Reference(var_29, var_28)
    var_31 = 'value'
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Definitions()
    var_34 = 'nested'
    var_35 = module_1.Field()
    var_36 = {var_34: var_35}
    var_37 = module_0.Schema(var_36)
    var_38 = 'schema_target'
    var_39 = module_0.Reference(var_38, var_33)
    var_40 = 'test'
    var_41 = {var_34: var_40}
    var_42 = var_39.validate(var_41)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_1.Field()
    var_15 = ()
    var_16 = 'Invalid'
    var_17 = 'invalid'
    var_18 = module_2.Message(text=var_16, code=var_17)
    var_19 = [var_18]
    var_20 = module_2.ValidationError(messages=var_19)
    var_21 = 'error_target'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 'bad_value'
    var_24 = var_22.validate(var_23)
    var_25 = 'missing'
    var_26 = module_0.Reference(var_25, var_0)
    var_27 = 1
    var_28 = var_26.validate(var_27)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = {var_2: var_0}
    var_11 = module_1.Schema(var_10)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = {var_12: var_0}
    var_15 = True
    var_16 = module_1.Schema(var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = {var_12: var_0}
    var_20 = module_1.Schema(var_19)
    var_21 = 'not a dict'
    var_22 = var_20.validate(var_21)
    var_23 = {var_21: var_0}
    var_24 = module_1.Schema(var_23)
    var_25 = 1
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.Field()
    var_30 = {var_25: var_29}
    var_31 = module_1.Schema(var_30)
    var_32 = {}
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Field(read_only=var_15)
    var_35 = module_0.Field()
    var_36 = 'id'
    var_37 = {var_36: var_34, var_32: var_35}
    var_38 = module_1.Schema(var_37)
    var_39 = {var_32: var_28}
    var_40 = var_38.validate(var_39)
    var_41 = 'default_value'
    var_42 = module_0.Field(default=var_41)
    var_43 = {var_32: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {}
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Field()
    var_48 = {var_32: var_47}
    var_49 = module_1.Schema(var_48)
    var_50 = 'name'
    var_51 = 'value'
    var_52 = {var_50: var_51}
    var_53 = var_49.validate(var_52)
    var_54 = module_0.Field()
    var_55 = module_0.Field()
    var_56 = 'field1'
    var_57 = 'field2'
    var_58 = {var_56: var_54, var_57: var_55}
    var_59 = module_1.Schema(var_58)
    var_60 = 'field1'
    var_61 = 'field2'
    var_62 = 'val1'
    var_63 = 'val2'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = var_59.validate(var_64)
    var_66 = {msg.code for msg in e.messages()}
    var_67 = module_0.Field()
    var_68 = 'valid_field'
    var_69 = {var_68: var_67}
    var_70 = module_1.Schema(var_69)
    var_71 = 1
    var_72 = 'other'
    var_73 = 'invalid'
    var_74 = 'value'
    var_75 = {var_71: var_73, var_72: var_74}
    var_76 = var_70.validate(var_75)
    var_77 = module_0.Field()
    var_78 = 'nested'
    var_79 = {var_78: var_77}
    var_80 = module_1.Schema(var_79)
    var_81 = 'value'
    var_82 = {var_78: var_81}
    var_83 = var_80.validate(var_82)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'Person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = True
    var_14 = module_0.Reference(var_7, var_0)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Reference(var_7, var_0)
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Reference(var_7, var_0)
    var_21 = 'name'
    var_22 = 'age'
    var_23 = 'John'
    var_24 = 'not_a_number'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = var_20.validate(var_25)
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = 'name'
    var_29 = 'John'
    var_30 = {var_28: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = 'street'
    var_34 = 'city'
    var_35 = module_1.String()
    var_36 = module_1.String()
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = module_0.Schema(var_37)
    var_39 = 'address'
    var_40 = module_1.String()
    var_41 = 'Address'
    var_42 = module_0.Reference(var_41, var_32)
    var_43 = {var_28: var_40, var_39: var_42}
    var_44 = module_0.Schema(var_43)
    var_45 = module_0.Reference(var_26, var_32)
    var_46 = 'Alice'
    var_47 = '123 Main St'
    var_48 = 'Boston'
    var_49 = {var_33: var_47, var_34: var_48}
    var_50 = {var_28: var_46, var_39: var_49}
    var_51 = var_45.validate(var_50)
    var_52 = module_0.Reference(var_26, var_32)
    var_53 = 'name'
    var_54 = 'address'
    var_55 = 'Alice'
    var_56 = 'street'
    var_57 = '123 Main St'
    var_58 = {var_56: var_57}
    var_59 = {var_53: var_55, var_54: var_58}
    var_60 = var_52.validate(var_59)
    var_61 = module_0.Definitions()
    var_62 = 'NonExistent'
    var_63 = module_0.Reference(var_62, var_61)
    var_64 = 'name'
    var_65 = 'John'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = module_1.Field()
    var_17 = 'target_field2'
    var_18 = module_0.Reference(var_17, var_15)
    var_19 = ()
    var_20 = 'Target error'
    var_21 = 'target_error'
    var_22 = module_2.Message(text=var_20, code=var_21)
    var_23 = [var_22]
    var_24 = module_2.ValidationError(messages=var_23)
    var_25 = 'some_value'
    var_26 = var_18.validate(var_25)
    var_27 = module_0.Definitions()
    var_28 = 'non_existent'
    var_29 = module_0.Reference(var_28, var_27)
    var_30 = 'value'
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Definitions()
    var_33 = 'name'
    var_34 = module_1.Field()
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'person_schema'
    var_38 = module_0.Reference(var_37, var_32)
    var_39 = 'John'
    var_40 = {var_33: var_39}
    var_41 = var_38.validate(var_40)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'test'
    var_5 = 'data'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_0.Reference(var_2, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Reference(var_2, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Definitions()
    var_17 = module_1.Field()
    var_18 = 'target_field2'
    var_19 = module_0.Reference(var_18, var_16)
    var_20 = 'some_value'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Definitions()
    var_23 = 'non_existent'
    var_24 = module_0.Reference(var_23, var_22)
    var_25 = 'value'
    var_26 = var_24.validate(var_25)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 'value'
    var_32 = {var_28: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = 'default_value'
    var_35 = module_0.Field(default=var_34)
    var_36 = 'with_default'
    var_37 = {var_36: var_35}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field()
    var_42 = 'failing'
    var_43 = {var_42: var_41}
    var_44 = module_1.Schema(var_43)
    var_45 = 'failing'
    var_46 = 'value'
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)
    var_49 = module_0.Field()
    var_50 = 'required'
    var_51 = {var_50: var_49}
    var_52 = module_1.Schema(var_51)
    var_53 = 2
    var_54 = 'invalid key'
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)
    var_57 = {msg.code for msg in e.messages()}
    var_58 = module_0.Field()
    var_59 = 'nested'
    var_60 = {var_59: var_58}
    var_61 = module_1.Schema(var_60)
    var_62 = 'valid_value'
    var_63 = {var_59: var_62}
    var_64 = var_61.validate(var_63)
    var_65 = 'defaulted'
    var_66 = {}
    var_67 = var_61.validate(var_66)
    var_68 = 'ignored'
    var_69 = module_0.Field(default=var_68, read_only=var_53)
    var_70 = 'complex'
    var_71 = {var_70: var_69}
    var_72 = module_1.Schema(var_71)
    var_73 = {}
    var_74 = var_72.validate(var_73)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 'other'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'default_value'
    var_36 = module_0.Field(default=var_35)
    var_37 = 'with_default'
    var_38 = {var_37: var_36}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 'name'
    var_44 = {var_43: var_42}
    var_45 = module_1.Schema(var_44)
    var_46 = 'John'
    var_47 = {var_43: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = 'failing'
    var_50 = 'failing'
    var_51 = 'any value'
    var_52 = {var_50: var_51}
    var_53 = var_45.validate(var_52)
    var_54 = module_0.Field()
    var_55 = 'required'
    var_56 = {var_55: var_54}
    var_57 = module_1.Schema(var_56)
    var_58 = 1
    var_59 = 'invalid key'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)
    var_62 = module_0.Field()
    var_63 = 'nested'
    var_64 = {var_63: var_62}
    var_65 = module_1.Schema(var_64)
    var_66 = 'parent'
    var_67 = {var_66: var_65}
    var_68 = module_1.Schema(var_67)
    var_69 = {var_63: var_32}
    var_70 = {var_66: var_69}
    var_71 = var_68.validate(var_70)
    var_72 = module_0.Field(allow_null=var_58)
    var_73 = 'nullable'
    var_74 = {var_73: var_72}
    var_75 = module_1.Schema(var_74)
    var_76 = {var_73: var_60}
    var_77 = var_75.validate(var_76)
    var_78 = module_0.Field(allow_null=var_7)
    var_79 = 'non_nullable'
    var_80 = {var_79: var_78}
    var_81 = module_1.Schema(var_80)
    var_82 = 'non_nullable'
    var_83 = None
    var_84 = {var_82: var_83}
    var_85 = var_81.validate(var_84)



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'Person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = module_0.Reference(var_7, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = True
    var_17 = module_0.Reference(var_7, var_0)
    var_18 = None
    var_19 = var_17.validate(var_18)
    assert var_19 is None
    var_20 = 'not_an_integer'
    var_21 = {var_14: var_9, var_15: var_20}
    var_22 = var_8.validate(var_21)
    var_23 = module_1.String()
    var_24 = {var_22: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = 'RequiredPerson'
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Definitions()
    var_31 = 'id'
    var_32 = module_1.Integer()
    var_33 = {var_31: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = 'data'
    var_36 = 'Inner'
    var_37 = module_0.Reference(var_36, var_30)
    var_38 = {var_35: var_37}
    var_39 = module_0.Schema(var_38)
    var_40 = 'Outer'
    var_41 = module_0.Reference(var_40, var_30)
    var_42 = 123
    var_43 = {var_31: var_42}
    var_44 = {var_35: var_43}
    var_45 = var_41.validate(var_44)
    var_46 = 'not_a_dict'
    var_47 = var_8.validate(var_46)
    var_48 = module_0.Definitions()
    var_49 = 'NonExistent'
    var_50 = module_0.Reference(var_49, var_48)
    var_51 = 'test'
    var_52 = 'data'
    var_53 = {var_51: var_52}
    var_54 = var_50.validate(var_53)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'active'
    var_3 = 10
    var_4 = module_0.String(max_length=var_3)
    var_5 = 0
    var_6 = 150
    var_7 = module_0.Integer(minimum=var_5, maximum=var_6)
    var_8 = module_0.Boolean()
    var_9 = {var_0: var_4, var_1: var_7, var_2: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = 'John'
    var_12 = 30
    var_13 = True
    var_14 = {var_0: var_11, var_1: var_12, var_2: var_13}
    var_15 = var_10.validate(var_14)
    var_16 = None
    var_17 = var_10.validate(var_16)
    var_18 = module_0.String(max_length=var_3)
    var_19 = {var_16: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = None
    var_22 = var_20.validate(var_21)
    assert var_22 is None
    var_23 = 'not a dict'
    var_24 = var_10.validate(var_23)
    var_25 = 1
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_10.validate(var_27)
    var_29 = 'name'
    var_30 = 'John'
    var_31 = {var_29: var_30}
    var_32 = var_10.validate(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'active'
    var_36 = 'VeryLongName'
    var_37 = 200
    var_38 = True
    var_39 = {var_33: var_36, var_34: var_37, var_35: var_38}
    var_40 = var_10.validate(var_39)
    var_41 = 'id'
    var_42 = module_0.Integer()
    var_43 = module_0.String()
    var_44 = {var_41: var_42, var_33: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'Alice'
    var_47 = {var_33: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = 'count'
    var_50 = module_0.String()
    var_51 = module_0.Integer()
    var_52 = {var_33: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = 'Bob'
    var_55 = {var_33: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = 'tags'
    var_58 = 'metadata'
    var_59 = module_0.String()
    var_60 = module_0.Array(var_59)
    var_61 = 'key'
    var_62 = 'value'
    var_63 = module_0.String()
    var_64 = module_0.String()
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = module_1.Schema(var_65)
    var_67 = {var_57: var_60, var_58: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = 'tag1'
    var_70 = 'tag2'
    var_71 = [var_69, var_70]
    var_72 = 'color'
    var_73 = 'blue'
    var_74 = {var_61: var_72, var_62: var_73}
    var_75 = {var_57: var_71, var_58: var_74}
    var_76 = var_68.validate(var_75)
    var_77 = 'tags'
    var_78 = 'metadata'
    var_79 = 'tag1'
    var_80 = 123
    var_81 = [var_79, var_80]
    var_82 = 'key'
    var_83 = 'value'
    var_84 = 'color'
    var_85 = 'blue'
    var_86 = {var_82: var_84, var_83: var_85}
    var_87 = {var_77: var_81, var_78: var_86}
    var_88 = var_68.validate(var_87)
    var_89 = 'name'
    var_90 = 'age'
    var_91 = 'VeryLongNameThatExceedsLimit'
    var_92 = -5
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = var_10.validate(var_93)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = {var_89: var_95, var_90: var_96}
    var_98 = module_1.Schema(var_97)
    var_99 = 25
    var_100 = {var_89: var_21, var_90: var_99}
    var_101 = var_98.validate(var_100)
    var_102 = {}
    var_103 = module_1.Schema(var_102)
    var_104 = {}
    var_105 = var_103.validate(var_104)
    var_106 = (var_89, var_87)
    var_107 = (var_90, var_88)
    var_108 = [var_106, var_107]
    var_109 = dict(var_105)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = 0
    var_5 = 150
    var_6 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    var_15 = module_0.String()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 'not a dict'
    var_22 = var_8.validate(var_21)
    var_23 = 1
    var_24 = 'name'
    var_25 = 'value'
    var_26 = 'John'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = var_8.validate(var_27)
    var_29 = 'name'
    var_30 = 'John'
    var_31 = {var_29: var_30}
    var_32 = var_8.validate(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'John'
    var_36 = 5
    var_37 = var_35 * var_36
    var_38 = 200
    var_39 = {var_33: var_37, var_34: var_38}
    var_40 = var_8.validate(var_39)
    var_41 = 'id'
    var_42 = module_0.Integer()
    var_43 = module_0.String()
    var_44 = {var_41: var_42, var_33: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'Alice'
    var_47 = {var_33: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = 'active'
    var_50 = module_0.String()
    var_51 = module_0.Boolean()
    var_52 = {var_33: var_50, var_49: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = 'Bob'
    var_55 = {var_33: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = 'profile'
    var_58 = 'email'
    var_59 = module_0.String(format=var_58)
    var_60 = {var_58: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = {var_57: var_61}
    var_63 = module_1.Schema(var_62)
    var_64 = 'profile'
    var_65 = 'email'
    var_66 = 'invalid-email'
    var_67 = {var_65: var_66}
    var_68 = {var_64: var_67}
    var_69 = var_63.validate(var_68)
    var_70 = 'required1'
    var_71 = 'required2'
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = module_0.String(format=var_58)
    var_75 = {var_70: var_72, var_71: var_73, var_58: var_74}
    var_76 = module_1.Schema(var_75)
    var_77 = 'email'
    var_78 = 'invalid_key'
    var_79 = 'not-an-email'
    var_80 = 123
    var_81 = {var_77: var_79, var_78: var_80}
    var_82 = var_76.validate(var_81)
    var_83 = (var_77, var_9)
    var_84 = 25
    var_85 = (var_78, var_84)
    var_86 = [var_83, var_85]



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'failing'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'anything'
    var_17 = var_15.validate(var_16)
    var_18 = 'nonexistent'
    var_19 = module_0.Reference(var_18, var_0)
    var_20 = 'value'
    var_21 = var_19.validate(var_20)
    var_22 = 'complex'
    var_23 = module_0.Reference(var_22, var_0)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_0.Reference(var_2, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Definitions()
    var_16 = module_1.Field()
    var_17 = 'error_ref'
    var_18 = module_0.Reference(var_17, var_15)
    var_19 = 'invalid_value'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Definitions()
    var_22 = 'missing_ref'
    var_23 = module_0.Reference(var_22, var_21)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)
    var_28 = module_0.Definitions()
    var_29 = 'name'
    var_30 = module_1.Field()
    var_31 = {var_29: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'person_ref'
    var_34 = module_0.Reference(var_33, var_28)
    var_35 = 'John'
    var_36 = {var_29: var_35}
    var_37 = {var_29: var_35}
    var_38 = var_34.validate(var_37)



# Parsed testcases at query #29
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'error_target'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'anything'
    var_17 = var_15.validate(var_16)
    var_18 = 'nonexistent'
    var_19 = module_0.Reference(var_18, var_0)
    var_20 = 5
    var_21 = var_19.validate(var_20)
    var_22 = 'complex'
    var_23 = module_0.Reference(var_22, var_0)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'active'
    var_7 = module_1.String()
    var_8 = module_1.Integer()
    var_9 = module_1.Boolean()
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = module_0.Schema(var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = True
    var_15 = {var_4: var_12, var_5: var_13, var_6: var_14}
    var_16 = var_11.serialize(var_15)
    var_17 = 'Alice'
    var_18 = 25
    var_19 = False
    var_20 = 'Bob'
    var_21 = {var_4: var_20}
    var_22 = var_11.serialize(var_21)
    var_23 = 'nested'
    var_24 = module_1.String()
    var_25 = {var_23: var_24}
    var_26 = module_0.Schema(var_25)
    var_27 = 'value'
    var_28 = {var_23: var_27}
    var_29 = var_26.serialize(var_28)
    var_30 = 'id'
    var_31 = module_1.String()
    var_32 = module_1.String()
    var_33 = {var_4: var_31, var_30: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = 'Test'
    var_36 = '123'
    var_37 = {var_4: var_35, var_30: var_36}
    var_38 = var_34.serialize(var_37)
    var_39 = {}
    var_40 = module_0.Schema(var_39)
    var_41 = {}
    var_42 = var_40.serialize(var_41)
    var_43 = 'data'
    var_44 = 'test'
    var_45 = {var_43: var_44}
    var_46 = var_40.serialize(var_45)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    var_8 = 'id'
    var_9 = {var_2: var_0, var_8: var_7}
    var_10 = module_1.Schema(var_9)
    var_11 = 'default_value'
    var_12 = module_0.Field(default=var_11)
    var_13 = 'optional'
    var_14 = {var_2: var_0, var_13: var_12}
    var_15 = module_1.Schema(var_14)
    var_16 = 'required_field'
    var_17 = 'read_only_field'
    var_18 = 'default_field'
    var_19 = module_0.Field(read_only=var_6)
    var_20 = 0
    var_21 = module_0.Field(default=var_20)
    var_22 = {var_16: var_0, var_17: var_19, var_18: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'Test schema'
    var_25 = module_1.Schema(var_4)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Field()
    var_11 = {var_2: var_10}
    var_12 = module_1.Schema(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Field()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = module_0.Field()
    var_22 = {var_13: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = 'not a dict'
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Field()
    var_27 = {var_24: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 1
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Field()
    var_34 = {var_29: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = {}
    var_37 = var_35.validate(var_36)
    var_38 = 'default_value'
    var_39 = module_0.Field(default=var_38)
    var_40 = {var_36: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Field(read_only=var_17)
    var_45 = 'id'
    var_46 = module_0.Field()
    var_47 = {var_36: var_46, var_45: var_44}
    var_48 = module_1.Schema(var_47)
    var_49 = 123
    var_50 = {var_36: var_32, var_45: var_49}
    var_51 = var_48.validate(var_50)
    var_52 = module_0.Field()
    var_53 = 'data'
    var_54 = {var_53: var_52}
    var_55 = module_1.Schema(var_54)
    var_56 = 'data'
    var_57 = 'some value'
    var_58 = {var_56: var_57}
    var_59 = var_55.validate(var_58)
    var_60 = module_0.Field()
    var_61 = module_0.Field()
    var_62 = 'field1'
    var_63 = 'field2'
    var_64 = {var_62: var_60, var_63: var_61}
    var_65 = module_1.Schema(var_64)
    var_66 = 1
    var_67 = 'invalid key'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)
    var_70 = module_0.Field()
    var_71 = 'child'
    var_72 = {var_71: var_70}
    var_73 = module_1.Schema(var_72)
    var_74 = 'child'
    var_75 = 'invalid'
    var_76 = {var_74: var_75}
    var_77 = var_73.validate(var_76)
    var_78 = module_0.Field()
    var_79 = module_0.Field()
    var_80 = module_0.Field()
    var_81 = 'email'
    var_82 = {var_74: var_78, var_75: var_79, var_81: var_80}
    var_83 = module_1.Schema(var_82)
    var_84 = 'Alice'
    var_85 = 25
    var_86 = 'alice@example.com'
    var_87 = {var_74: var_84, var_75: var_85, var_81: var_86}
    var_88 = var_83.validate(var_87)
    var_89 = 'field'
    var_90 = {}
    var_91 = var_83.validate(var_90)
    var_92 = module_0.Field()
    var_93 = module_0.Field()
    var_94 = 'valid'
    var_95 = 'invalid'
    var_96 = {var_94: var_92, var_95: var_93}
    var_97 = module_1.Schema(var_96)
    var_98 = 'valid'
    var_99 = 'invalid'
    var_100 = 'ok'
    var_101 = 'anything'
    var_102 = {var_98: var_100, var_99: var_101}
    var_103 = var_97.validate(var_102)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = {}
    var_5 = module_0.Schema(var_4)
    var_6 = {}
    var_7 = var_5.serialize(var_6)
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'active'
    var_11 = module_1.String()
    var_12 = module_1.Integer()
    var_13 = module_1.Boolean()
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = module_0.Schema(var_14)
    var_16 = 'John'
    var_17 = 30
    var_18 = True
    var_19 = {var_8: var_16, var_9: var_17, var_10: var_18}
    var_20 = var_15.serialize(var_19)
    var_21 = module_1.String()
    var_22 = module_1.Integer()
    var_23 = module_1.Boolean()
    var_24 = {var_8: var_21, var_9: var_22, var_10: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = 'Jane'
    var_27 = {var_8: var_26}
    var_28 = var_25.serialize(var_27)
    var_29 = module_1.String()
    var_30 = module_1.Integer()
    var_31 = {var_8: var_29, var_9: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'extra'
    var_34 = 'Bob'
    var_35 = 25
    var_36 = 'ignored'
    var_37 = {var_8: var_34, var_9: var_35, var_33: var_36}
    var_38 = var_32.serialize(var_37)
    var_39 = module_1.String()
    var_40 = module_1.Integer()
    var_41 = {var_8: var_39, var_9: var_40}
    var_42 = module_0.Schema(var_41)
    var_43 = 'Alice'
    var_44 = 28
    var_45 = module_1.String()
    var_46 = module_1.Integer()
    var_47 = {var_8: var_45, var_9: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = 'Charlie'
    var_50 = 'street'
    var_51 = 'city'
    var_52 = module_1.String()
    var_53 = module_1.String()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_0.Schema(var_54)
    var_56 = 'address'
    var_57 = module_1.String()
    var_58 = {var_8: var_57, var_56: var_55}
    var_59 = module_0.Schema(var_58)
    var_60 = 'David'
    var_61 = '123 Main'
    var_62 = 'Boston'
    var_63 = {var_50: var_61, var_51: var_62}
    var_64 = {var_8: var_60, var_56: var_63}
    var_65 = var_59.serialize(var_64)
    var_66 = 'id'
    var_67 = module_1.String()
    var_68 = module_1.String()
    var_69 = {var_8: var_67, var_66: var_68}
    var_70 = module_0.Schema(var_69)
    var_71 = 'Eve'
    var_72 = '12345'
    var_73 = {var_8: var_71, var_66: var_72}
    var_74 = var_70.serialize(var_73)
    var_75 = 'data'
    var_76 = 'value'
    var_77 = {var_75: var_76}
    var_78 = var_70.serialize(var_77)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 'other'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'default_value'
    var_36 = module_0.Field(default=var_35)
    var_37 = 'with_default'
    var_38 = {var_37: var_36}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 'name'
    var_44 = {var_43: var_42}
    var_45 = module_1.Schema(var_44)
    var_46 = 'John'
    var_47 = {var_43: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = module_0.Field()
    var_50 = 'nested'
    var_51 = {var_50: var_49}
    var_52 = module_1.Schema(var_51)
    var_53 = 'nested'
    var_54 = None
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)
    var_57 = module_0.Field()
    var_58 = 'req1'
    var_59 = 'req2'
    var_60 = {var_58: var_57, var_59: var_57}
    var_61 = module_1.Schema(var_60)
    var_62 = 2
    var_63 = 'invalid key'
    var_64 = {var_62: var_63}
    var_65 = var_61.validate(var_64)
    var_66 = 'default'
    var_67 = module_0.Field(default=var_66, read_only=var_62)
    var_68 = 'complex'
    var_69 = {var_68: var_67}
    var_70 = module_1.Schema(var_69)
    var_71 = {}
    var_72 = var_70.validate(var_71)
    var_73 = module_0.Field()
    var_74 = module_0.Field()
    var_75 = 'age'
    var_76 = {var_43: var_73, var_75: var_74}
    var_77 = module_1.Schema(var_76)
    var_78 = 'Alice'
    var_79 = 30
    var_80 = {var_43: var_78, var_75: var_79}
    var_81 = var_77.validate(var_80)
    var_82 = module_0.Field()
    var_83 = 'count'
    var_84 = {var_83: var_82}
    var_85 = module_1.Schema(var_84)
    var_86 = 42
    var_87 = {var_83: var_86}
    var_88 = var_85.validate(var_87)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = {}
    var_5 = module_0.Schema(var_4)
    var_6 = {}
    var_7 = var_5.serialize(var_6)
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'active'
    var_11 = module_1.String()
    var_12 = module_1.Integer()
    var_13 = module_1.Boolean()
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = module_0.Schema(var_14)
    var_16 = 'John'
    var_17 = 30
    var_18 = True
    var_19 = {var_8: var_16, var_9: var_17, var_10: var_18}
    var_20 = var_15.serialize(var_19)
    var_21 = module_1.String()
    var_22 = module_1.Integer()
    var_23 = module_1.Boolean()
    var_24 = {var_8: var_21, var_9: var_22, var_10: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = {var_8: var_16, var_9: var_17}
    var_27 = var_25.serialize(var_26)
    var_28 = module_1.String()
    var_29 = module_1.Integer()
    var_30 = module_1.Boolean()
    var_31 = {var_8: var_28, var_9: var_29, var_10: var_30}
    var_32 = module_0.Schema(var_31)
    var_33 = 'Alice'
    var_34 = 25
    var_35 = False
    var_36 = 'street'
    var_37 = 'city'
    var_38 = module_1.String()
    var_39 = module_1.String()
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = 'address'
    var_43 = module_1.String()
    var_44 = {var_8: var_43, var_42: var_41}
    var_45 = module_0.Schema(var_44)
    var_46 = 'Bob'
    var_47 = '123 Main St'
    var_48 = 'Boston'
    var_49 = {var_36: var_47, var_37: var_48}
    var_50 = {var_8: var_46, var_42: var_49}
    var_51 = var_45.serialize(var_50)
    var_52 = 'id'
    var_53 = module_1.String()
    var_54 = module_1.Integer()
    var_55 = module_1.Integer()
    var_56 = {var_8: var_53, var_52: var_54, var_9: var_55}
    var_57 = module_0.Schema(var_56)
    var_58 = 'Charlie'
    var_59 = 123
    var_60 = 40
    var_61 = {var_8: var_58, var_52: var_59, var_9: var_60}
    var_62 = var_57.serialize(var_61)
    var_63 = 'code'
    var_64 = module_1.String()
    var_65 = 'test'
    var_66 = 'abc'
    var_67 = {var_8: var_65, var_63: var_66}
    var_68 = var_57.serialize(var_67)
    var_69 = module_1.String()
    var_70 = module_1.Integer()
    var_71 = {var_8: var_69, var_9: var_70}
    var_72 = module_0.Schema(var_71)
    var_73 = 'David'
    var_74 = module_1.String()
    var_75 = module_1.Integer()
    var_76 = {var_8: var_74, var_9: var_75}
    var_77 = module_0.Schema(var_76)
    var_78 = {}
    var_79 = var_77.serialize(var_78)



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Person'
    var_7 = module_0.Reference(var_6, var_0)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = module_0.Reference(var_6, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = True
    var_16 = module_0.Reference(var_6, var_0)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = 'not_an_integer'
    var_20 = {var_13: var_8, var_14: var_19}
    var_21 = var_7.validate(var_20)
    var_22 = 'street'
    var_23 = 'city'
    var_24 = module_1.String()
    var_25 = module_1.String()
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = 'Address'
    var_28 = module_0.Reference(var_27, var_0)
    var_29 = '123 Main St'
    var_30 = 'Anytown'
    var_31 = {var_22: var_29, var_23: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'NonExistent'
    var_34 = module_0.Reference(var_33, var_0)
    var_35 = 'test'
    var_36 = 'data'
    var_37 = {var_35: var_36}
    var_38 = var_34.validate(var_37)
    var_39 = 'employees'
    var_40 = module_1.String()
    var_41 = module_1.Integer(minimum=var_15)
    var_42 = {var_35: var_40, var_39: var_41}
    var_43 = 'Company'
    var_44 = module_0.Reference(var_43, var_0)
    var_45 = 'Tech Corp'
    var_46 = 100
    var_47 = {var_35: var_45, var_39: var_46}
    var_48 = var_44.validate(var_47)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only_field'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'default_value'
    var_34 = module_0.Field(default=var_33)
    var_35 = 'field_with_default'
    var_36 = {var_35: var_34}
    var_37 = module_1.Schema(var_36)
    var_38 = {}
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Field()
    var_41 = 'nested'
    var_42 = {var_41: var_40}
    var_43 = module_1.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_41: var_44}
    var_46 = var_43.validate(var_45)
    var_47 = module_0.Field()
    var_48 = {var_41: var_47}
    var_49 = module_1.Schema(var_48)
    var_50 = 'nested'
    var_51 = None
    var_52 = {var_50: var_51}
    var_53 = var_49.validate(var_52)
    var_54 = module_0.Field()
    var_55 = module_0.Field()
    var_56 = 'field1'
    var_57 = 'field2'
    var_58 = {var_56: var_54, var_57: var_55}
    var_59 = module_1.Schema(var_58)
    var_60 = 3
    var_61 = 'field1'
    var_62 = 'invalid'
    var_63 = None
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = var_59.validate(var_64)
    var_66 = 'inner_field'
    var_67 = module_0.Field()
    var_68 = {var_66: var_67}
    var_69 = module_1.Schema(var_68)
    var_70 = 'outer_field'
    var_71 = {var_70: var_69}
    var_72 = module_1.Schema(var_71)
    var_73 = {var_66: var_44}
    var_74 = {var_70: var_73}
    var_75 = var_72.validate(var_74)
    var_76 = module_0.Field(read_only=var_60)
    var_77 = module_0.Field()
    var_78 = 'read_only'
    var_79 = 'regular'
    var_80 = {var_78: var_76, var_79: var_77}
    var_81 = module_1.Schema(var_80)
    var_82 = {var_79: var_44}
    var_83 = var_81.validate(var_82)
    var_84 = 'default1'
    var_85 = module_0.Field(default=var_84)
    var_86 = module_0.Field()
    var_87 = module_0.Field()
    var_88 = 'field3'
    var_89 = {var_56: var_85, var_57: var_86, var_88: var_87}
    var_90 = module_1.Schema(var_89)
    var_91 = 'field2'
    var_92 = 'field3'
    var_93 = 'valid'
    var_94 = None
    var_95 = {var_91: var_93, var_92: var_94}
    var_96 = var_90.validate(var_95)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = 0
    var_5 = 150
    var_6 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    var_15 = module_0.String()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 'not a dict'
    var_22 = var_8.validate(var_21)
    var_23 = 1
    var_24 = 'name'
    var_25 = 'value'
    var_26 = 'John'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = var_8.validate(var_27)
    var_29 = 'name'
    var_30 = 'John'
    var_31 = {var_29: var_30}
    var_32 = var_8.validate(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'John'
    var_36 = -5
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = var_8.validate(var_37)
    var_39 = module_0.String()
    var_40 = 25
    var_41 = module_0.Integer()
    var_42 = {var_33: var_39, var_34: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = {var_33: var_9}
    var_45 = var_43.validate(var_44)
    var_46 = module_0.String()
    var_47 = 'id'
    var_48 = module_0.String()
    var_49 = {var_33: var_48, var_47: var_46}
    var_50 = module_1.Schema(var_49)
    var_51 = '123'
    var_52 = {var_33: var_9, var_47: var_51}
    var_53 = var_50.validate(var_52)
    var_54 = {}
    var_55 = var_8.validate(var_54)
    var_56 = 'person'
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_54: var_57, var_55: var_58}
    var_60 = module_1.Schema(var_59)
    var_61 = {var_56: var_60}
    var_62 = module_1.Schema(var_61)
    var_63 = 'person'
    var_64 = 'age'
    var_65 = 'not a number'
    var_66 = {var_64: var_65}
    var_67 = {var_63: var_66}
    var_68 = var_62.validate(var_67)
    var_69 = 'Alice'
    var_70 = {var_63: var_69, var_64: var_40}
    var_71 = {var_56: var_70}
    var_72 = var_62.validate(var_71)
    var_73 = 'field'
    var_74 = {}
    var_75 = {}
    var_76 = module_1.Schema(var_75)
    var_77 = 'extra'
    var_78 = {var_77: var_73}
    var_79 = var_76.validate(var_78)
    var_80 = 'a'
    var_81 = 'b'
    var_82 = 'c'
    var_83 = module_0.Integer(minimum=var_65)
    var_84 = module_0.String()
    var_85 = 5
    var_86 = module_0.Integer(maximum=var_85)
    var_87 = {var_80: var_83, var_81: var_84, var_82: var_86}
    var_88 = module_1.Schema(var_87)
    var_89 = 'a'
    var_90 = 'c'
    var_91 = 5
    var_92 = 10
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = var_88.validate(var_93)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'active'
    var_7 = module_1.String()
    var_8 = module_1.Integer()
    var_9 = module_1.Boolean()
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = module_0.Schema(var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = True
    var_15 = {var_4: var_12, var_5: var_13, var_6: var_14}
    var_16 = var_11.serialize(var_15)
    var_17 = 'Alice'
    var_18 = 25
    var_19 = False
    var_20 = 'city'
    var_21 = module_1.String()
    var_22 = module_1.Integer()
    var_23 = module_1.String()
    var_24 = {var_4: var_21, var_5: var_22, var_20: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = 'Bob'
    var_27 = 35
    var_28 = {var_4: var_26, var_5: var_27}
    var_29 = var_25.serialize(var_28)
    var_30 = 'Charlie'
    var_31 = 'street'
    var_32 = 'zipcode'
    var_33 = module_1.String()
    var_34 = module_1.String()
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'address'
    var_38 = module_1.String()
    var_39 = {var_4: var_38, var_37: var_36}
    var_40 = module_0.Schema(var_39)
    var_41 = 'David'
    var_42 = 'Main St'
    var_43 = '12345'
    var_44 = {var_31: var_42, var_32: var_43}
    var_45 = {var_4: var_41, var_37: var_44}
    var_46 = var_40.serialize(var_45)
    var_47 = 'id'
    var_48 = module_1.Integer()
    var_49 = module_1.String()
    var_50 = {var_47: var_48, var_4: var_49}
    var_51 = module_0.Schema(var_50)
    var_52 = 'Eve'
    var_53 = {var_47: var_14, var_4: var_52}
    var_54 = var_51.serialize(var_53)
    var_55 = {}
    var_56 = module_0.Schema(var_55)
    var_57 = {}
    var_58 = var_56.serialize(var_57)
    var_59 = module_1.String()
    var_60 = {var_4: var_59}
    var_61 = module_0.Schema(var_60)
    var_62 = 'extra'
    var_63 = 'Frank'
    var_64 = 'field'
    var_65 = {var_4: var_63, var_62: var_64}
    var_66 = var_61.serialize(var_65)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 'value'
    var_32 = {var_28: var_31}
    var_33 = var_30.validate(var_32)
    var_34 = 'default_value'
    var_35 = module_0.Field(default=var_34)
    var_36 = 'field_with_default'
    var_37 = {var_36: var_35}
    var_38 = module_1.Schema(var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Field()
    var_42 = 'name'
    var_43 = {var_42: var_41}
    var_44 = module_1.Schema(var_43)
    var_45 = 'John'
    var_46 = {var_42: var_45}
    var_47 = var_44.validate(var_46)
    var_48 = module_0.Field()
    var_49 = 'Invalid'
    var_50 = 'invalid'
    var_51 = module_2.Message(text=var_49, code=var_50)
    var_52 = [var_51]
    var_53 = module_2.ValidationError(messages=var_52)
    var_54 = (var_19, var_53)
    var_55 = 'failing'
    var_56 = {var_55: var_48}
    var_57 = module_1.Schema(var_56)
    var_58 = 'failing'
    var_59 = 'value'
    var_60 = {var_58: var_59}
    var_61 = var_57.validate(var_60)
    var_62 = module_0.Field()
    var_63 = module_0.Field()
    var_64 = 'required'
    var_65 = 'another'
    var_66 = {var_64: var_62, var_65: var_63}
    var_67 = module_1.Schema(var_66)
    var_68 = 'another'
    var_69 = 123
    var_70 = {var_68: var_69}
    var_71 = var_67.validate(var_70)
    var_72 = module_0.Field()
    var_73 = 'nested'
    var_74 = {var_73: var_72}
    var_75 = module_1.Schema(var_74)
    var_76 = 'outer'
    var_77 = {var_76: var_75}
    var_78 = module_1.Schema(var_77)
    var_79 = {var_73: var_31}
    var_80 = {var_76: var_79}
    var_81 = var_78.validate(var_80)
    var_82 = module_0.Field(allow_null=var_68)
    var_83 = 'nullable'
    var_84 = {var_83: var_82}
    var_85 = module_1.Schema(var_84)
    var_86 = {var_83: var_70}
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Field()
    var_89 = module_0.Field()
    var_90 = 'field1'
    var_91 = 'field2'
    var_92 = {var_90: var_88, var_91: var_89}
    var_93 = module_1.Schema(var_92)
    var_94 = 'value1'
    var_95 = 'value2'
    var_96 = {var_90: var_94, var_91: var_95}
    var_97 = var_93.validate(var_96)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'test_ref'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_0.Reference(var_2, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_3.validate(var_12)
    var_14 = module_0.Definitions()
    var_15 = module_1.Field()
    var_16 = 'failing_ref'
    var_17 = module_0.Reference(var_16, var_14)
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = 'non_existent'
    var_23 = module_0.Reference(var_22, var_0)
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = module_0.String()
    var_13 = {var_10: var_12}
    var_14 = True
    var_15 = module_1.Schema(var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = 'not a dict'
    var_19 = var_5.validate(var_18)
    var_20 = 1
    var_21 = 'name'
    var_22 = 'value'
    var_23 = 'John'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = var_5.validate(var_24)
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = {var_20: var_26, var_21: var_27}
    var_29 = module_1.Schema(var_28)
    var_30 = 'name'
    var_31 = 'John'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = 'default_name'
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = {var_30: var_35, var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = {var_31: var_7}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = {var_30: var_41, var_31: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = {var_31: var_7}
    var_46 = var_44.validate(var_45)
    var_47 = 5
    var_48 = module_0.String(max_length=var_47)
    var_49 = 0
    var_50 = module_0.Integer(minimum=var_49)
    var_51 = {var_30: var_48, var_31: var_50}
    var_52 = module_1.Schema(var_51)
    var_53 = 'name'
    var_54 = 'age'
    var_55 = 'Too Long Name'
    var_56 = -5
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)
    var_59 = 'street'
    var_60 = 'city'
    var_61 = module_0.String()
    var_62 = module_0.String()
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = module_1.Schema(var_63)
    var_65 = 'address'
    var_66 = module_0.String()
    var_67 = {var_53: var_66, var_65: var_64}
    var_68 = module_1.Schema(var_67)
    var_69 = '123 Main St'
    var_70 = 'Anytown'
    var_71 = {var_59: var_69, var_60: var_70}
    var_72 = {var_53: var_58, var_65: var_71}
    var_73 = var_68.validate(var_72)
    var_74 = 'unknown'
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = {var_53: var_75, var_54: var_76}
    var_78 = module_1.Schema(var_77)
    var_79 = {}
    var_80 = var_78.validate(var_79)
    var_81 = 'name'
    var_82 = 'address'
    var_83 = 'John'
    var_84 = 'street'
    var_85 = 'city'
    var_86 = 123
    var_87 = 'Anytown'
    var_88 = {var_84: var_86, var_85: var_87}
    var_89 = {var_81: var_83, var_82: var_88}
    var_90 = var_68.validate(var_89)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'Person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = True
    var_14 = module_0.Reference(var_7, var_0)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Reference(var_7, var_0)
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = 'name'
    var_21 = 'age'
    var_22 = 'John'
    var_23 = 'not_a_number'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = var_8.validate(var_24)
    var_26 = 'name'
    var_27 = 'John'
    var_28 = {var_26: var_27}
    var_29 = var_8.validate(var_28)
    var_30 = module_0.Definitions()
    var_31 = 'address'
    var_32 = 'city'
    var_33 = module_1.String()
    var_34 = module_1.String()
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = module_1.String()
    var_38 = 'Address'
    var_39 = module_0.Reference(var_38, var_30)
    var_40 = {var_26: var_37, var_31: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = module_0.Reference(var_25, var_30)
    var_43 = 'Alice'
    var_44 = '123 Main St'
    var_45 = 'Boston'
    var_46 = {var_31: var_44, var_32: var_45}
    var_47 = {var_26: var_43, var_31: var_46}
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Definitions()
    var_50 = 'NonExistent'
    var_51 = module_0.Reference(var_50, var_49)
    var_52 = 'name'
    var_53 = 'John'
    var_54 = {var_52: var_53}
    var_55 = var_51.validate(var_54)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = 0
    var_5 = 150
    var_6 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    var_15 = module_0.String()
    var_16 = {var_13: var_15}
    var_17 = True
    var_18 = module_1.Schema(var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 'not a dict'
    var_22 = var_8.validate(var_21)
    var_23 = 1
    var_24 = 'name'
    var_25 = 'value'
    var_26 = 'John'
    var_27 = {var_23: var_25, var_24: var_26}
    var_28 = var_8.validate(var_27)
    var_29 = 'name'
    var_30 = 'John'
    var_31 = {var_29: var_30}
    var_32 = var_8.validate(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'VeryLongName'
    var_36 = 200
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = var_8.validate(var_37)
    var_39 = 'id'
    var_40 = module_0.Integer()
    var_41 = module_0.String()
    var_42 = {var_39: var_40, var_33: var_41}
    var_43 = module_1.Schema(var_42)
    var_44 = 'Alice'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)
    var_47 = 'Unknown'
    var_48 = module_0.String()
    var_49 = module_0.Integer()
    var_50 = {var_33: var_48, var_34: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = 25
    var_53 = {var_34: var_52}
    var_54 = var_51.validate(var_53)
    var_55 = 'address'
    var_56 = 'street'
    var_57 = 'city'
    var_58 = module_0.String()
    var_59 = module_0.String()
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = module_1.Schema(var_60)
    var_62 = {var_55: var_61}
    var_63 = module_1.Schema(var_62)
    var_64 = 'address'
    var_65 = 'city'
    var_66 = 'NYC'
    var_67 = {var_65: var_66}
    var_68 = {var_64: var_67}
    var_69 = var_63.validate(var_68)
    var_70 = 'email'
    var_71 = module_0.String()
    var_72 = module_0.String()
    var_73 = 18
    var_74 = module_0.Integer(minimum=var_73)
    var_75 = {var_64: var_71, var_70: var_72, var_65: var_74}
    var_76 = module_1.Schema(var_75)
    var_77 = 'age'
    var_78 = 15
    var_79 = {var_77: var_78}
    var_80 = var_76.validate(var_79)
    var_81 = 'optional'
    var_82 = module_0.String()
    var_83 = {var_81: var_82}
    var_84 = module_1.Schema(var_83)
    var_85 = {}
    var_86 = var_84.validate(var_85)
    var_87 = 'Bob'
    var_88 = 40
    var_89 = {var_77: var_87, var_78: var_88}



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Definitions()
    var_15 = module_1.Field()
    var_16 = ()
    var_17 = 'Invalid'
    var_18 = 'invalid'
    var_19 = module_2.Message(text=var_17, code=var_18)
    var_20 = [var_19]
    var_21 = module_2.ValidationError(messages=var_20)
    var_22 = module_0.Reference(var_13, var_14)
    var_23 = 'bad_value'
    var_24 = var_22.validate(var_23)
    var_25 = module_0.Definitions()
    var_26 = 'missing'
    var_27 = module_0.Reference(var_26, var_25)
    var_28 = 42
    var_29 = var_27.validate(var_28)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 'target_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = var_3.validate(var_6)
    var_9 = True
    var_10 = module_0.Reference(var_2, var_0)
    var_11 = var_10.validate(var_7)
    assert var_11 is None
    var_12 = module_0.Reference(var_2, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 'name'
    var_16 = module_1.Field()
    var_17 = {var_15: var_16}
    var_18 = module_0.Schema(var_17)
    var_19 = 'person'
    var_20 = module_0.Reference(var_19, var_0)
    var_21 = 'John'
    var_22 = {var_15: var_21}
    var_23 = var_20.validate(var_22)
    var_24 = 'strict_field'
    var_25 = module_0.Reference(var_24, var_0)
    var_26 = 'invalid'
    var_27 = var_25.validate(var_26)
    var_28 = 'missing'
    var_29 = module_0.Reference(var_28, var_0)
    var_30 = 'anything'
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Reference(var_30, var_0)
    var_33 = 'not null'
    var_34 = var_32.validate(var_33)
    assert var_34 == 'not null'



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = {}
    var_7 = False
    var_8 = module_1.Schema(var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = {}
    var_12 = module_1.Schema(var_11)
    var_13 = 'not a dict'
    var_14 = var_12.validate(var_13)
    var_15 = {}
    var_16 = module_1.Schema(var_15)
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Field()
    var_22 = 'required_field'
    var_23 = {var_22: var_21}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Field(read_only=var_25)
    var_28 = 'read_only'
    var_29 = {var_28: var_27}
    var_30 = module_1.Schema(var_29)
    var_31 = 'other'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'default_value'
    var_36 = module_0.Field(default=var_35)
    var_37 = 'field_with_default'
    var_38 = {var_37: var_36}
    var_39 = module_1.Schema(var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Field()
    var_43 = 'name'
    var_44 = {var_43: var_42}
    var_45 = module_1.Schema(var_44)
    var_46 = 'John'
    var_47 = {var_43: var_46}
    var_48 = var_45.validate(var_47)
    var_49 = module_0.Field()
    var_50 = 'age'
    var_51 = {var_50: var_49}
    var_52 = module_1.Schema(var_51)
    var_53 = 'age'
    var_54 = None
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)
    var_57 = module_0.Field()
    var_58 = 'required'
    var_59 = {var_58: var_57}
    var_60 = module_1.Schema(var_59)
    var_61 = 1
    var_62 = 'invalid key'
    var_63 = {var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = [msg.code for msg in e.messages()]
    var_66 = module_0.Field()
    var_67 = 'nested'
    var_68 = {var_67: var_66}
    var_69 = module_1.Schema(var_68)
    var_70 = 'outer'
    var_71 = {var_70: var_69}
    var_72 = module_1.Schema(var_71)
    var_73 = {var_67: var_32}
    var_74 = {var_70: var_73}
    var_75 = var_72.validate(var_74)
    var_76 = module_0.Field()
    var_77 = 'default'
    var_78 = module_0.Field(default=var_77)
    var_79 = module_0.Field(read_only=var_61)
    var_80 = 'with_default'
    var_81 = {var_58: var_76, var_80: var_78, var_28: var_79}
    var_82 = module_1.Schema(var_81)
    var_83 = {var_58: var_32}
    var_84 = var_82.validate(var_83)
    var_85 = {}
    var_86 = module_1.Schema(var_85)
    var_87 = 'extra'
    var_88 = 'should be ignored'
    var_89 = {var_87: var_88}
    var_90 = var_86.validate(var_89)
    var_91 = module_0.Field(allow_null=var_61)
    var_92 = 'nullable'
    var_93 = {var_92: var_91}
    var_94 = module_1.Schema(var_93)
    var_95 = {var_92: var_63}
    var_96 = var_94.validate(var_95)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = 0
    var_5 = 150
    var_6 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_1.Schema(var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = None
    var_14 = var_8.validate(var_13)
    var_15 = module_0.String(max_length=var_2)
    var_16 = module_0.Integer(minimum=var_4, maximum=var_5)
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = True
    var_19 = module_1.Schema(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None
    var_22 = 'not an object'
    var_23 = var_8.validate(var_22)
    var_24 = 1
    var_25 = 'name'
    var_26 = 'value'
    var_27 = 'John'
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = var_8.validate(var_28)
    var_30 = 'name'
    var_31 = 'John'
    var_32 = {var_30: var_31}
    var_33 = var_8.validate(var_32)
    var_34 = 'name'
    var_35 = 'age'
    var_36 = 'John'
    var_37 = 10
    var_38 = var_36 * var_37
    var_39 = 200
    var_40 = {var_34: var_38, var_35: var_39}
    var_41 = var_8.validate(var_40)
    var_42 = 'id'
    var_43 = module_0.String(max_length=var_36)
    var_44 = module_0.Integer()
    var_45 = module_0.Integer(minimum=var_38)
    var_46 = {var_34: var_43, var_42: var_44, var_35: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = 123
    var_49 = {var_34: var_9, var_42: var_48, var_35: var_10}
    var_50 = var_47.validate(var_49)
    var_51 = 'active'
    var_52 = module_0.String(max_length=var_36)
    var_53 = 25
    var_54 = module_0.Integer()
    var_55 = 'yes'
    var_56 = module_0.String()
    var_57 = {var_34: var_52, var_35: var_54, var_51: var_56}
    var_58 = module_1.Schema(var_57)
    var_59 = {var_34: var_9}
    var_60 = var_58.validate(var_59)
    var_61 = 'address'
    var_62 = module_0.String(max_length=var_36)
    var_63 = 'street'
    var_64 = 'city'
    var_65 = module_0.String()
    var_66 = module_0.String()
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = {var_34: var_62, var_61: var_68}
    var_70 = module_1.Schema(var_69)
    var_71 = 'name'
    var_72 = 'address'
    var_73 = 'John'
    var_74 = 'street'
    var_75 = 'Main St'
    var_76 = {var_74: var_75}
    var_77 = {var_71: var_73, var_72: var_76}
    var_78 = var_70.validate(var_77)
    var_79 = 'Main St'
    var_80 = 'Springfield'
    var_81 = {var_63: var_79, var_64: var_80}
    var_82 = {var_71: var_9, var_61: var_81}
    var_83 = var_70.validate(var_82)
    var_84 = 'Jane'
    var_85 = {var_71: var_84, var_72: var_53}
    var_86 = 1
    var_87 = 2.5
    var_88 = 'name'
    var_89 = 'value'
    var_90 = 'value2'
    var_91 = 'John'
    var_92 = {var_86: var_89, var_87: var_90, var_88: var_91}
    var_93 = var_8.validate(var_92)
    var_94 = 'field1'
    var_95 = 'field2'
    var_96 = 'field3'
    var_97 = module_0.String()
    var_98 = 20
    var_99 = module_0.Integer(minimum=var_88, maximum=var_98)
    var_100 = 5
    var_101 = module_0.String(max_length=var_100)
    var_102 = {var_94: var_97, var_95: var_99, var_96: var_101}
    var_103 = module_1.Schema(var_102)
    var_104 = 'field2'
    var_105 = 'field3'
    var_106 = 5
    var_107 = 'too long value'
    var_108 = {var_104: var_106, var_105: var_107}
    var_109 = var_103.validate(var_108)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'test_value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test_value'
    var_5 = True
    var_6 = module_0.Reference(var_1, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = None
    var_10 = var_2.validate(var_9)
    var_11 = 'other_ref'
    var_12 = module_0.Reference(var_11, var_0)
    var_13 = var_12.validate(var_7)
    assert var_13 is None
    var_14 = 'missing_ref'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'value'
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = True
    var_14 = module_1.Schema(var_12)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = 'not a dict'
    var_18 = var_5.validate(var_17)
    var_19 = 1
    var_20 = 'name'
    var_21 = 'value'
    var_22 = 'John'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = var_5.validate(var_23)
    var_25 = module_0.Field()
    var_26 = 'required_field'
    var_27 = {var_26: var_25}
    var_28 = module_1.Schema(var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Field(read_only=var_13)
    var_32 = 'read_only'
    var_33 = 'normal'
    var_34 = {var_32: var_31, var_33: var_0}
    var_35 = module_1.Schema(var_34)
    var_36 = 'value'
    var_37 = {var_33: var_36}
    var_38 = var_35.validate(var_37)
    var_39 = 'default_value'
    var_40 = module_0.Field(default=var_39)
    var_41 = 'field'
    var_42 = {var_41: var_40}
    var_43 = module_1.Schema(var_42)
    var_44 = {}
    var_45 = var_43.validate(var_44)
    var_46 = module_0.Field()
    var_47 = 'nested'
    var_48 = {var_47: var_46}
    var_49 = module_1.Schema(var_48)
    var_50 = 'error_field'
    var_51 = 'error_field'
    var_52 = 'bad_value'
    var_53 = {var_51: var_52}
    var_54 = module_0.Field()
    var_55 = module_0.Field()
    var_56 = 'field1'
    var_57 = 'field2'
    var_58 = {var_56: var_54, var_57: var_55}
    var_59 = module_1.Schema(var_58)
    var_60 = {}
    var_61 = var_59.validate(var_60)
    var_62 = {var_60: var_22, var_61: var_23}



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test_ref'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'test_value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'validated_test_value'
    var_5 = True
    var_6 = module_0.Reference(var_1, var_0)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = module_0.Reference(var_1, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = 'invalid'
    var_13 = var_2.validate(var_12)
    var_14 = 'nonexistent'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'test'
    var_17 = var_15.validate(var_16)
    var_18 = 'name'
    var_19 = module_1.Field()
    var_20 = {var_18: var_19}
    var_21 = 'nested'
    var_22 = module_0.Reference(var_21, var_0)
    var_23 = 'test'
    var_24 = {var_18: var_23}
    var_25 = var_22.validate(var_24)
    var_26 = 'custom'
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = 5
    var_29 = var_27.validate(var_28)
    assert var_29 == 10
    var_30 = 'not_int'
    var_31 = var_27.validate(var_30)



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'Person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = module_0.Reference(var_7, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = True
    var_17 = module_0.Reference(var_7, var_0)
    var_18 = None
    var_19 = var_17.validate(var_18)
    assert var_19 is None
    var_20 = 123
    var_21 = 'thirty'
    var_22 = {var_14: var_20, var_15: var_21}
    var_23 = var_8.validate(var_22)
    var_24 = 'street'
    var_25 = 'city'
    var_26 = module_1.String()
    var_27 = module_1.String()
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = 'Address'
    var_31 = module_0.Reference(var_30, var_0)
    var_32 = 'Main St'
    var_33 = 'New York'
    var_34 = {var_24: var_32, var_25: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 'NonExistent'
    var_37 = module_0.Reference(var_36, var_0)
    var_38 = 'test'
    var_39 = 'data'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'id'
    var_43 = 'person'
    var_44 = module_1.Integer()
    var_45 = module_0.Reference(var_7, var_0)
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = module_0.Schema(var_46)
    var_48 = 'Complex'
    var_49 = module_0.Reference(var_48, var_0)
    var_50 = 'Alice'
    var_51 = 25
    var_52 = {var_38: var_50, var_39: var_51}
    var_53 = {var_42: var_16, var_43: var_52}
    var_54 = var_49.validate(var_53)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = {var_2: var_0}
    var_11 = module_1.Schema(var_10)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = {var_12: var_0}
    var_15 = True
    var_16 = module_1.Schema(var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = {var_12: var_0}
    var_20 = module_1.Schema(var_19)
    var_21 = 'not a dict'
    var_22 = var_20.validate(var_21)
    var_23 = {var_21: var_0}
    var_24 = module_1.Schema(var_23)
    var_25 = 1
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.Field()
    var_30 = 'optional'
    var_31 = module_0.Field()
    var_32 = {var_25: var_29, var_30: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = 'optional'
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = var_33.validate(var_36)
    var_38 = module_0.Field(read_only=var_15)
    var_39 = 'id'
    var_40 = module_0.Field()
    var_41 = {var_34: var_40, var_39: var_38}
    var_42 = module_1.Schema(var_41)
    var_43 = 123
    var_44 = {var_34: var_37, var_39: var_43}
    var_45 = var_42.validate(var_44)
    var_46 = 'default_value'
    var_47 = module_0.Field(default=var_46)
    var_48 = {var_34: var_47}
    var_49 = module_1.Schema(var_48)
    var_50 = {}
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Field()
    var_53 = ()
    var_54 = 'Invalid'
    var_55 = 'invalid'
    var_56 = []
    var_57 = module_2.Message(text=var_54, code=var_55, index=var_56)
    var_58 = [var_57]
    var_59 = module_2.ValidationError(messages=var_58)
    var_60 = 'nested'
    var_61 = {var_60: var_52}
    var_62 = module_1.Schema(var_61)
    var_63 = 'nested'
    var_64 = 'value'
    var_65 = {var_63: var_64}
    var_66 = var_62.validate(var_65)
    var_67 = module_0.Field()
    var_68 = module_0.Field()
    var_69 = 'field1'
    var_70 = 'field2'
    var_71 = {var_69: var_67, var_70: var_68}
    var_72 = module_1.Schema(var_71)
    var_73 = ()
    var_74 = 'Error 1'
    var_75 = 'error1'
    var_76 = []
    var_77 = module_2.Message(text=var_74, code=var_75, index=var_76)
    var_78 = [var_77]
    var_79 = module_2.ValidationError(messages=var_78)
    var_80 = ()
    var_81 = 'Error 2'
    var_82 = 'error2'
    var_83 = []
    var_84 = module_2.Message(text=var_81, code=var_82, index=var_83)
    var_85 = [var_84]
    var_86 = module_2.ValidationError(messages=var_85)
    var_87 = 'field1'
    var_88 = 'field2'
    var_89 = 'val1'
    var_90 = 'val2'
    var_91 = {var_87: var_89, var_88: var_90}
    var_92 = var_72.validate(var_91)
    var_93 = module_0.Field()
    var_94 = {var_87: var_93}
    var_95 = module_1.Schema(var_94)
    var_96 = {var_87: var_90}



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Person'
    var_7 = module_0.Reference(var_6, var_0)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = module_0.Reference(var_6, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = True
    var_16 = module_0.Reference(var_6, var_0)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = 'not_an_integer'
    var_20 = {var_13: var_8, var_14: var_19}
    var_21 = var_7.validate(var_20)
    var_22 = 'street'
    var_23 = 'city'
    var_24 = 'zip'
    var_25 = module_1.String()
    var_26 = module_1.String()
    var_27 = module_1.String()
    var_28 = {var_22: var_25, var_23: var_26, var_24: var_27}
    var_29 = 'Address'
    var_30 = module_0.Reference(var_29, var_0)
    var_31 = '123 Main'
    var_32 = 'Anytown'
    var_33 = '12345'
    var_34 = {var_22: var_31, var_23: var_32, var_24: var_33}
    var_35 = var_30.validate(var_34)
    var_36 = 'NonExistent'
    var_37 = module_0.Reference(var_36, var_0)
    var_38 = 'test'
    var_39 = 'data'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'id'
    var_43 = module_1.Integer()
    var_44 = module_1.String()
    var_45 = {var_42: var_43, var_38: var_44}
    var_46 = 'RequiredFields'
    var_47 = module_0.Reference(var_46, var_0)
    var_48 = {var_42: var_15, var_38: var_17}
    var_49 = var_47.validate(var_48)
    var_50 = 'Test'
    var_51 = {var_38: var_50}
    var_52 = var_47.validate(var_51)



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.Field()
    var_2 = 2
    var_3 = 'target'
    var_4 = module_0.Reference(var_3, var_0)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    assert var_6 == 10
    var_7 = True
    var_8 = module_0.Reference(var_3, var_0)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Reference(var_3, var_0)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 'failing'
    var_15 = module_0.Reference(var_14, var_0)
    var_16 = 'anything'
    var_17 = var_15.validate(var_16)
    var_18 = 'identity'
    var_19 = module_0.Reference(var_18, var_0)
    var_20 = 'key'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = 'missing'
    var_25 = module_0.Reference(var_24, var_0)
    var_26 = 'test'
    var_27 = var_25.validate(var_26)



# Parsed testcases at query #27
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Schema(var_5)
    var_7 = 'Person'
    var_8 = module_0.Reference(var_7, var_0)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = module_0.Reference(var_7, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = True
    var_17 = module_0.Reference(var_7, var_0)
    var_18 = None
    var_19 = var_17.validate(var_18)
    assert var_19 is None
    var_20 = 'name'
    var_21 = 'age'
    var_22 = 'John'
    var_23 = 'not_a_number'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = var_8.validate(var_24)
    var_26 = 'NonExistent'
    var_27 = module_0.Reference(var_26, var_0)
    var_28 = 'name'
    var_29 = 'John'
    var_30 = {var_28: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = 'street'
    var_33 = 'city'
    var_34 = module_1.String()
    var_35 = module_1.String()
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.Schema(var_36)
    var_38 = 'Address'
    var_39 = module_0.Reference(var_38, var_0)
    var_40 = '123 Main St'
    var_41 = 'Springfield'
    var_42 = {var_32: var_40, var_33: var_41}
    var_43 = var_39.validate(var_42)



# Parsed testcases at query #28
#--------------------------


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_1.String()
    var_4 = module_1.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Person'
    var_7 = module_0.Reference(var_6, var_0)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = module_0.Reference(var_6, var_0)
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = True
    var_16 = module_0.Reference(var_6, var_0)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = 'name'
    var_20 = 'age'
    var_21 = 'John'
    var_22 = 'thirty'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = var_7.validate(var_23)
    var_25 = 'street'
    var_26 = 'city'
    var_27 = module_1.String()
    var_28 = module_1.String()
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = 'Address'
    var_31 = module_0.Reference(var_30, var_0)
    var_32 = 'Main St'
    var_33 = 'Boston'
    var_34 = {var_25: var_32, var_26: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 'NonExistent'
    var_37 = module_0.Reference(var_36, var_0)
    var_38 = 'test'
    var_39 = 'value'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 'person'
    var_43 = 'address'
    var_44 = module_0.Reference(var_24, var_0)
    var_45 = module_0.Reference(var_30, var_0)
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = 'Profile'
    var_48 = module_0.Reference(var_47, var_0)
    var_49 = 'Alice'
    var_50 = 25
    var_51 = {var_38: var_49, var_39: var_50}
    var_52 = 'Oak Ave'
    var_53 = 'Seattle'
    var_54 = {var_25: var_52, var_26: var_53}
    var_55 = {var_42: var_51, var_43: var_54}
    var_56 = var_48.validate(var_55)



